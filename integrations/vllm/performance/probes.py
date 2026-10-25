"""Bounded, falsifiable layout / old TMA-switch / eager / bandwidth probes."""
import argparse
import ctypes as C
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from hygemm_vllm.native import DualLinear,Problem,library,pack,check,POLICIES
from hygemm_vllm.offline_weights import logical_scales,physical_scales
from tune import read,write,measure,cfg,decode_mx


def timing(fn,x,cold=False,eager=False,minimum_nodes=1):
    fn();torch.cuda.synchronize()
    if not eager:
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            for _ in range(1 if cold else 8):fn()
        cudart=C.CDLL('/usr/local/cuda/lib64/libcudart.so')
        cudart.cudaGraphGetNodes.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_size_t)]
        node_count=C.c_size_t()
        assert cudart.cudaGraphGetNodes(graph.raw_cuda_graph(),None,C.byref(node_count))==0
        assert node_count.value>=minimum_nodes,('Empty/incomplete captured probe',node_count.value)
        def run():graph.replay()
        count=1 if cold else 8
    else:
        count=32
        def run():
            for _ in range(count):fn()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
    for _ in range(5):
        if cold:measure.eviction.add_(1);x.mul_(1)
        a.record();run();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/count)
    return {'median_us':statistics.median(samples),'samples_us':samples,
            'cv':statistics.pstdev(samples)/statistics.mean(samples),'cold':cold,'eager_includes_submit_gaps':eager,
            'graph_nodes':node_count.value if not eager else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
    stage=args.stage;out=stage/'probes';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    meta=read(stage/'inputs.json')['7B'];cases=read(meta['shape_manifest'])['layers']
    plans=read(stage/'7B/tune_old/plans.json')['hygemm']
    representatives={}
    for c in cases:representatives.setdefault((c['out'],c['k']),[]).append(c)
    representatives=[sorted(cs,key=lambda c:c['high'])[len(cs)//2] for cs in representatives.values()]
    measure.eviction=torch.zeros(256*1024**2,device='cuda',dtype=torch.uint8)
    fused=C.CDLL(str(stage/'fused_build/libfused_probe.so'));fused.hg_fused_error.restype=C.c_char_p
    fused.hg_fused_create.argtypes=[C.c_int,C.POINTER(Problem),C.POINTER(Problem),C.c_void_p,C.POINTER(C.c_void_p)]
    fused.hg_fused_run.argtypes=[C.c_void_p,C.c_void_p];fused.hg_fused_destroy.argtypes=[C.c_void_p]
    fused.hg_fused_info.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
    records=[]
    for c in representatives:
        o,k,h=c['out'],c['k'],c['high'];w=(torch.randn(o,k,device='cuda')*.04).half()
        r=DualLinear(w,torch.tensor(c['high_ids'],device='cuda'));rw=torch.empty(o,k,device='cuda')
        if r.lo.numel():rw[r.lo.long()]=decode_mx(r.w4,r.sf4,k,True)
        if r.hi.numel():rw[r.hi.long()]=decode_mx(r.w8,r.sf8,k,False)
        originals={n:getattr(r,n) for n in ['lo','hi','w4','w8','sf4','sf8']}
        for t in [1,8,64]:
            config=next(p for p in plans if (p['out'],p['k'],p['high'],p['tokens'])==(o,k,h,t))
            r.low_id,r.high_id,r.concurrent=config['low_id'],config['high_id'],POLICIES[config['policy']]
            x=torch.randn(t,k,device='cuda',dtype=torch.float16);q,sf=pack(x);expected=(decode_mx(q,sf,k,False)@rw.t()).half()
            baseline=measure(r,x,cfg(config),expected,cold=True)
            for order in ['reverse','block_reverse']:
                r.close()
                for part,wn,sn in [('lo','w4','sf4'),('hi','w8','sf8')]:
                    ids=originals[part];n=len(ids)
                    if not n:continue
                    if order=='reverse':permutation=torch.arange(n-1,-1,-1,device='cuda')
                    else:
                        chunks=list(torch.arange(n,device='cuda').split(32));permutation=torch.cat(chunks[::-1])
                    setattr(r,part,ids[permutation]);setattr(r,wn,originals[wn][permutation].contiguous())
                    setattr(r,sn,physical_scales(logical_scales(originals[sn],n,k)[permutation].contiguous(),k))
                score=measure(r,x,cfg(config),expected,cold=True)
                records.append({'probe':'partition_row_order','shape':[o,k,h,t],'order':order,'baseline':baseline,'candidate':score,
                                'same_payload_and_scale_bytes':True,'offline_conversion_not_timed':True})
                r.close()
                for n,v in originals.items():setattr(r,n,v)
            # Fused TMA probe keeps q/sf/y addresses fixed for its lifetime.
            y=torch.empty(t,o,device='cuda',dtype=torch.float16);stream=torch.cuda.current_stream().cuda_stream
            low=Problem(len(r.lo),t,k,o,r.w4.data_ptr(),q.data_ptr(),r.sf4.data_ptr(),sf.data_ptr(),r.lo.data_ptr(),y.data_ptr())
            high=Problem(len(r.hi),t,k,o,r.w8.data_ptr(),q.data_ptr(),r.sf8.data_ptr(),sf.data_ptr(),r.hi.data_ptr(),y.data_ptr())
            for variant in range(8):
                handle=C.c_void_p();status=fused.hg_fused_create(variant,C.byref(low),C.byref(high),stream,C.byref(handle))
                assert status==0,fused.hg_fused_error()
                def launch():
                    # torch.cuda.graph selects a capture stream. Never retain
                    # the handle-creation stream for subsequent graph launches.
                    current=torch.cuda.current_stream().cuda_stream
                    check(library().hg_pack(x.data_ptr(),q.data_ptr(),sf.data_ptr(),t,k,0,current))
                    status=fused.hg_fused_run(handle,current)
                    if status:raise RuntimeError(fused.hg_fused_error().decode())
                launch();torch.testing.assert_close(y,expected,atol=.01,rtol=.003)
                values=(C.c_int*11)();fused.hg_fused_info(handle,values)
                score=timing(launch,x,cold=True,minimum_nodes=2)
                records.append({'probe':'old_tma_switch','shape':[o,k,h,t],'variant':variant,'info':list(values),
                                'baseline':baseline,'candidate':score,'includes_online_pack':True})
                fused.hg_fused_destroy(handle)
            if t in [1,8]:
                for policy in ['serial','concurrent','high_first','high_main','gemv_pair1']:
                    r.concurrent=POLICIES[policy]
                    y=r(x);torch.testing.assert_close(y,expected,atol=.01,rtol=.003)
                    records.append({'probe':'eager_vs_graph','shape':[o,k,h,t],'policy':policy,
                        'graph':timing(lambda:r(x),x),'eager':timing(lambda:r(x),x,eager=True)})
            write(out/'results.json',records)
        r.close();del r,rw,w,originals;gc.collect();torch.cuda.empty_cache()
    # Streaming memcpy's two-way traffic is an empirical optimistic BW reference,
    # not a claim that mixed-weight gather can attain the same rate.
    n=1024**3;a=torch.empty(n,device='cuda',dtype=torch.uint8);b=torch.empty_like(a);a.fill_(1)
    score=timing(lambda:b.copy_(a),a)
    bandwidth=2*n/(score['median_us']*1e-6)
    write(out/'bandwidth.json',{'copy_bytes_each_direction':n,'read_plus_write_bytes':2*n,
        'measured_bytes_per_second':bandwidth,'timing':score,
        'optimistic_quantized_weight_stream_ms':meta['budget']['persistent_quantized_bytes']/bandwidth*1000,
        'scope':'weights/scales/IDs only; actual decode also reads activations, attention/KV, embedding/lm-head and writes outputs'})


if __name__=='__main__':main()
