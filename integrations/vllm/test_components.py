"""Validate pack/layout, pointer rebinding, native kernels and graph replay."""
import argparse
import json
from pathlib import Path
import torch
from hygemm_vllm.native import DualLinear, pack
from hygemm_vllm.mixllm import MixLinear, load_library


def decode_mx(q, sf, k, fp4):
    rows = q.shape[0]
    if not rows:
        return torch.empty((0,k), device=q.device, dtype=torch.float32)
    if fp4:
        codes = torch.stack((q & 15, q >> 4), dim=-1).flatten(1).long()
        lut = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], device=q.device)
        values = lut[codes]
    else:
        values = q.view(torch.float8_e4m3fn).float()
    r = torch.arange(rows, device=q.device)[:,None]
    g = torch.arange(k//32, device=q.device)[None,:]
    index = ((r//128)*(k//128)+g//4)*512+(r%32)*16+((r%128)//32)*4+g%4
    scale = torch.exp2(sf[index].float()-127)
    return (values.view(rows,k//32,32)*scale[:,:,None]).reshape(rows,k)


def reference(runtime, x, kind):
    if kind == 'hygemm':
        q,sf = pack(x)
        dx = decode_mx(q,sf,x.shape[1],False)
        w = torch.empty(runtime.outputs,runtime.k,device=x.device)
        w[runtime.lo.long()] = decode_mx(runtime.w4,runtime.sf4,runtime.k,True)
        w[runtime.hi.long()] = decode_mx(runtime.w8,runtime.sf8,runtime.k,False)
    else:
        q,sf = torch.ops.kernels_mixllm.quantize(x)
        dx = (q.float().reshape(x.shape[0],-1,128)*sf[:,:x.shape[0]].t()[:,:,None]).reshape_as(x)
        w = runtime.dequant_weight()
    return (dx@w.t()).half()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['hygemm','mixllm'], required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.kind=='mixllm':load_library()
    records=[]
    outputs=384 if args.kind=='mixllm' else 257
    for k in [256,896]:
        for high in [0,26,outputs]:
            w=(torch.randn(outputs,k,device='cuda')*.04).half()
            ids=torch.randperm(outputs,device='cuda')[:high]
            r=DualLinear(w,ids) if args.kind=='hygemm' else MixLinear(w,ids)
            for tokens in [1,3,17,128]:
                print(args.kind,outputs,k,high,tokens,flush=True)
                x=torch.randn(tokens,k,device='cuda',dtype=torch.float16)
                y=r(x);ref=reference(r,x,args.kind)
                torch.testing.assert_close(y,ref,atol=.002,rtol=.002)
                err=(y.float()-ref.float()).abs().max().item()
                # New pointers, same shape: stale TMA descriptors must fail this.
                x2=(torch.randn_like(x)*.5).contiguous()
                y2=r(x2);ref2=reference(r,x2,args.kind)
                torch.testing.assert_close(y2,ref2,atol=.002,rtol=.002)
                records.append({'k':k,'high':high,'tokens':tokens,'max_abs':err,'passed':True})
            if high==26 and k==896:
                x=torch.randn(3,k,device='cuda',dtype=torch.float16)
                r(x);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=r(x)
                for seed in [111,112]:
                    torch.manual_seed(seed);x.copy_(torch.randn_like(x));graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(y,reference(r,x,args.kind),atol=.002,rtol=.002)
                records.append({'graph_changed_input_replay':True,'passed':True})
    if args.kind=='mixllm':
        w=(torch.randn(1024,256,device='cuda')*.04).half()
        r=MixLinear(w,torch.arange(102,device='cuda'))
        x=torch.randn(3,256,device='cuda',dtype=torch.float16)
        torch.testing.assert_close(r(x),reference(r,x,args.kind),atol=.002,rtol=.002)
        records.append({'original_cuda_transpose':True,'passed':True})
    args.out.write_text(json.dumps(records,indent=2))
    print(args.kind,'passed',len(records),'component/graph cases')


if __name__=='__main__':main()
