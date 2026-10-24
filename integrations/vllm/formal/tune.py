"""Bounded offline autotiling on actual model dimensions and partition counts.

Full candidate screening on a representative partition per O/K/purity, then
top-three refinement for every actual partition width and token bucket.
"""
import argparse
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
import torch
from hygemm_vllm.native import DualLinear,pack
from hygemm_vllm.mixllm import MixLinear,load_library
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_components import decode_mx

TOKENS=[1,2,4,8,16,32,64,128,256,512,1024,2048]


def make_reference(r,kind):
    indices=torch.linspace(0,r.outputs-1,min(64,r.outputs),device='cuda').long()
    special=[a[:1] for a in [r.hi,r.lo] if a.numel()]+[a[-1:] for a in [r.hi,r.lo] if a.numel()]
    indices=torch.unique(torch.cat([indices,*special]).long())
    if kind=='mixllm':w=r.dequant_weight()[indices]
    else:
        w=torch.empty(r.outputs,r.k,device='cuda')
        w[r.lo.long()]=decode_mx(r.w4,r.sf4,r.k,True)
        w[r.hi.long()]=decode_mx(r.w8,r.sf8,r.k,False)
        w=w[indices]
    return indices,w


def input_reference(x,kind):
    ids=torch.linspace(0,x.shape[0]-1,min(16,x.shape[0]),device='cuda').long()
    if kind=='mixllm':
        q,s=torch.ops.kernels_mixllm.quantize(x)
        dx=(q.float().reshape(x.shape[0],-1,128)*s[:,:x.shape[0]].t()[...,None]).reshape_as(x)
    else:
        q,s=pack(x);dx=decode_mx(q,s,x.shape[1],False)
    return ids,dx[ids]


def measure(r,x,kind,config,expected,index,token_ids,active_ids=None,repeats=3):
    if kind=='mixllm':r.forced_config=(config['stage'],config['id'])
    else:r.low_id,r.high_id,r.concurrent=config['low_id'],config['high_id'],config['policy']=='concurrent'
    y=r(x)
    cols=index
    ref=expected
    if active_ids is not None:
        mask=torch.isin(index,active_ids);cols=index[mask];ref=expected[:,mask]
    assert torch.isfinite(y[:,cols]).all()
    torch.testing.assert_close(y[token_ids[:,None],cols[None,:]],ref,atol=.01,rtol=.003)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    count=32
    with torch.cuda.graph(graph):
        for _ in range(count):y=r(x)
    graph.replay();torch.cuda.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(repeats):
        begin.record();graph.replay();end.record();end.synchronize()
        samples.append(begin.elapsed_time(end)*1000/count)
    return {**config,'us':statistics.median(samples),'samples_us':samples,
            'cv':statistics.pstdev(samples)/statistics.mean(samples),'validated':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tokens',default=','.join(map(str,TOKENS)))
    args=ap.parse_args();buckets=list(map(int,args.tokens.split(',')))
    torch.set_num_threads(4);torch.manual_seed(2029);torch.backends.cuda.matmul.allow_tf32=False
    load_library()
    cases=json.loads((args.out/'cases.json').read_text())['cases']
    groups=defaultdict(list)
    for c in cases:groups[(c['out'],c['k'],0 if not c['high'] else 1 if c['high']==c['out'] else 2)].append(c)
    prop=torch.cuda.get_device_properties(0)
    shared_limit=prop.shared_memory_per_block_optin
    inventory={};valid={}
    for rm in [False,True]:
        candidates=[];all_meta=[]
        for stage in [5,11]:
            for i in range(1,60):
                info=torch.ops.kernels_mixllm.config_info(stage,i,rm)
                if not info:continue
                ok=max(info[5:7])<=shared_limit and info[7]<=1024
                item={'stage':stage,'id':i,'info':info,'legal':ok}
                all_meta.append(item)
                if ok:candidates.append({'stage':stage,'id':i})
        inventory[str(rm)]=all_meta;valid[rm]=candidates
    (args.out/'mixllm_candidate_inventory.json').write_text(json.dumps({'shared_limit':shared_limit,'candidates':inventory},indent=2))
    result={'hygemm':[],'mixllm':[],'protocol':{'tokens':buckets,'screen':'all legal original MixLLM; HyGEMM isolated eight then top3x3 and serial/concurrent',
        'refinement':'top three whole-layer plans on every exact O/K/high width',
        'inputs':'fixed random data; actual channel maps; resident microbench, includes online pack and output layout',
        'sampled_reference':True,'global_optimality_claim':False}}
    if (args.out/'plans.json').exists():result=json.loads((args.out/'plans.json').read_text())
    done={(kind,r['out'],r['k'],r['high'],r['tokens']) for kind in ['hygemm','mixllm'] for r in result[kind]}
    raw=(args.out/'tuning_raw.jsonl').open('a',buffering=1)
    shortlist_path=args.out/'shortlists.json'
    shortlists=json.loads(shortlist_path.read_text()) if shortlist_path.exists() else {}
    shortlists={group:{t:rows for t,rows in values.items() if rows} for group,values in shortlists.items()}
    started=time.perf_counter()
    for (outputs,k,purity),group in groups.items():
        representative=sorted(group,key=lambda c:c['high'])[len(group)//2]
        weight=(torch.randn(outputs,k,device='cuda')*.04).half()
        for kind in ['hygemm','mixllm']:
            groupkey=f'{kind}_{outputs}_{k}_{purity}'
            shortlists.setdefault(groupkey,{})
            if any(str(t) not in shortlists[groupkey] for t in buckets):
                ids=torch.tensor(representative['high_ids'],device='cuda')
                r=DualLinear(weight,ids) if kind=='hygemm' else MixLinear(weight,ids)
                index,rw=make_reference(r,kind)
                for t in buckets:
                    if str(t) in shortlists[groupkey]:continue
                    x=torch.randn(t,k,device='cuda',dtype=torch.float16)
                    tids,dx=input_reference(x,kind);expected=(dx@rw.t()).half()
                    if kind=='mixllm':configs=valid[purity!=2]
                    else:
                        top={};lo,hi=r.lo,r.hi
                        for part,active in [('low',lo),('high',hi)]:
                            if not active.numel():top[part]=[0];continue
                            r.lo=lo if part=='low' else lo[:0];r.hi=hi if part=='high' else hi[:0]
                            scores=[]
                            for candidate in range(8):
                                config={'low_id':candidate if part=='low' else 0,'high_id':candidate if part=='high' else 0,'policy':'serial'}
                                score=measure(r,x,kind,config,expected,index,tids,active)
                                scores.append(score)
                                raw.write(json.dumps({'phase':'isolated','kind':kind,'out':outputs,'k':k,'high':representative['high'],'tokens':t,'part':part,**score})+'\n')
                            top[part]=[s[part+'_id'] for s in sorted(scores,key=lambda s:s['us'])[:3]]
                        r.lo,r.hi=lo,hi
                        configs=[{'low_id':li,'high_id':hi_id,'policy':policy} for li in top['low'] for hi_id in top['high']
                                 for policy in (['serial','concurrent'] if purity==2 else ['serial'])]
                    scores=[]
                    for config in configs:
                        score=measure(r,x,kind,config,expected,index,tids)
                        scores.append(score)
                        raw.write(json.dumps({'phase':'screen','kind':kind,'out':outputs,'k':k,'high':representative['high'],'tokens':t,**score})+'\n')
                    shortlists[groupkey][str(t)]=sorted(scores,key=lambda s:s['us'])[:3]
                    shortlist_path.write_text(json.dumps(shortlists,indent=2))
                    print('screen',groupkey,t,'best',shortlists[groupkey][str(t)][0],flush=True)
                if kind=='hygemm':r.close()
                del r,index,rw,x,dx,expected;gc.collect();torch.cuda.empty_cache()
            for case in group:
                if all((kind,outputs,k,case['high'],t) in done for t in buckets):continue
                ids=torch.tensor(case['high_ids'],device='cuda')
                r=DualLinear(weight,ids) if kind=='hygemm' else MixLinear(weight,ids)
                index,rw=make_reference(r,kind)
                for t in buckets:
                    key=(kind,outputs,k,case['high'],t)
                    if key in done:continue
                    x=torch.randn(t,k,device='cuda',dtype=torch.float16)
                    tids,dx=input_reference(x,kind);expected=(dx@rw.t()).half()
                    scores=[]
                    for old in shortlists[groupkey][str(t)]:
                        config={q:old[q] for q in (['low_id','high_id','policy'] if kind=='hygemm' else ['stage','id'])}
                        score=measure(r,x,kind,config,expected,index,tids,repeats=3)
                        scores.append(score)
                        raw.write(json.dumps({'phase':'refine','kind':kind,'out':outputs,'k':k,'high':case['high'],'tokens':t,**score})+'\n')
                    best=min(scores,key=lambda s:s['us'])
                    result[kind].append({'out':outputs,'k':k,'high':case['high'],'tokens':t,**best})
                    done.add(key)
                    (args.out/'plans.json').write_text(json.dumps(result,indent=2))
                print('refined',kind,outputs,k,case['high'],len(result[kind]),flush=True)
                if kind=='hygemm':r.close()
                del r,index,rw,x,dx,expected;gc.collect();torch.cuda.empty_cache()
        del weight;gc.collect();torch.cuda.empty_cache()
    assert all(len(result[k])==len(cases)*len(buckets) for k in ['hygemm','mixllm'])
    (args.out/'tuning_complete.json').write_text(json.dumps({'elapsed_s':time.perf_counter()-started,'cases':len(cases),'buckets':buckets,
         'plans_per_path':len(cases)*len(buckets),'passed':True},indent=2))


if __name__=='__main__':main()
