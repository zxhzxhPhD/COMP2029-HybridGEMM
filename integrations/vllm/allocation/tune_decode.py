"""Bounded existing-kernel tuning for changed partitions; no device edits."""
import argparse
import gc
from pathlib import Path
import sys
from alloc_utils import *
from hygemm_vllm.native import DualLinear
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'formal'))
from tune import measure


def config(row):return {k:row[k] for k in ['low_id','high_id','policy']}


def full_reference(runtime):
    weights=torch.empty(runtime.outputs,runtime.k,device='cuda',dtype=torch.float32)
    if runtime.lo.numel():weights[runtime.lo.long()]=decode_mx(runtime.w4,runtime.sf4,runtime.k,True)
    if runtime.hi.numel():weights[runtime.hi.long()]=decode_mx(runtime.w8,runtime.sf8,runtime.k,False)
    return torch.arange(runtime.outputs,device='cuda'),weights


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--view',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();configure();torch.manual_seed(2029);args.out.mkdir(parents=True,exist_ok=True)
    cases=read(args.root/'views'/args.view/'view.json')['layers']
    old={r['layer']:r for r in read(args.root/'reference_cases.json')['layers']}
    plans=read(args.root/'reference_plans.json')['hygemm']
    inherited={};unique={}
    for c in cases:
        key=(c['out'],c['k'],c['high']);unique.setdefault(key,c)
        for tokens in [1,8,64,128]:
            candidates=sorted([r for r in plans if (r['out'],r['k'],r['high'])==(c['out'],c['k'],old[c['layer']]['high'])],key=lambda r:r['tokens'])
            cfg=config(next(r for r in candidates if r['tokens']>=tokens))
            inherited.setdefault((*key,tokens),[])
            if cfg not in inherited[(*key,tokens)]:inherited[(*key,tokens)].append(cfg)
    target=args.out/'plans.json';result=read(target) if target.exists() else {'hygemm':[]}
    done={(r['out'],r['k'],r['high'],r['tokens']) for r in result['hygemm']}
    short={};raw=(args.out/'measurements.jsonl').open('a',buffering=1)
    for key,c in unique.items():
        if all((*key,t) in done for t in [1,8,64,128]):continue
        w=(torch.randn(c['out'],c['k'],device='cuda')*.04).half()
        r=DualLinear(w,torch.tensor(c['high_ids'],device='cuda'));index,rw=full_reference(r)
        for t in [1,8,64,128]:
            full=(*key,t)
            if full in done:continue
            x=torch.randn(t,c['k'],device='cuda',dtype=torch.float16)
            tids=torch.arange(t,device='cuda');dx=dq(x);expected=(dx@rw.t()).half()
            group=(c['out'],c['k'],c['high']==0,c['high']==c['out'],t)
            if group not in short:
                lo,hi=r.lo,r.hi;top={}
                for part,active in [('low',lo),('high',hi)]:
                    if not active.numel():top[part]=[0];continue
                    r.lo,r.hi=(lo,hi[:0]) if part=='low' else (lo[:0],hi)
                    scores=[]
                    for candidate in range(28 if t<=8 else 8):
                        cfg={'low_id':candidate if part=='low' else 0,'high_id':candidate if part=='high' else 0,'policy':'serial'}
                        value=measure(r,x,'hygemm',cfg,expected,index,tids,active);scores.append(value)
                        raw.write(__import__('json').dumps({'phase':'screen','key':full,'part':part,**value})+'\n')
                    top[part]=[s[part+'_id'] for s in sorted(scores,key=lambda s:s['us'])[:2]]
                r.lo,r.hi=lo,hi
                short[group]=[{'low_id':a,'high_id':b,'policy':p} for a in top['low'] for b in top['high']
                    for p in (['serial','concurrent'] if lo.numel() and hi.numel() else ['serial'])]
            configs=inherited[full]+[c for c in short[group] if c not in inherited[full]]
            scores=[measure(r,x,'hygemm',cfg,expected,index,tids) for cfg in configs]
            for value in scores:raw.write(__import__('json').dumps({'phase':'exact_shape','key':full,**value})+'\n')
            baseline=min(scores[:len(inherited[full])],key=lambda s:s['us']);best=min(scores,key=lambda s:s['us'])
            chosen=best if baseline['us']/best['us']>=1.02 else baseline
            result['hygemm'].append({'out':key[0],'k':key[1],'high':key[2],'tokens':t,**chosen})
            write(target,result);done.add(full)
        r.close();del r,rw,w,x,dx,expected;gc.collect();torch.cuda.empty_cache()
        print('Tuned',key,flush=True)
    assert len(done)==len(unique)*4
    write(args.out/'complete.json',{'passed':True,'plans':len(done),'shape_count':len(unique),
        'search':'first exact shape per O/K/purity screen: IDs0-27 at T1/8, IDs0-7 at T64/128, top2 pairs then exact-H refinement plus inherited plans',
        'correctness':'all output rows and tokens against quantized dense reference at atol .01 rtol .003 for every measurement',
        'measurement':'resident 3x32 Graph calls; whole-model confirmation required','micro_adoption_margin':1.02})


if __name__=='__main__':main()
