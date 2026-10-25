"""Version-bound actual-shape search: resident screens and cold refinement."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import torch
from hygemm_vllm.native import DualLinear, POLICIES, pack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_components import decode_mx


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    tmp=Path(p).with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cfg(r):return {k:r[k] for k in ['low_id','high_id','policy']}


def measure(r,x,c,expected,active=None,cold=False):
    r.low_id,r.high_id,r.concurrent=c['low_id'],c['high_id'],POLICIES[c['policy']]
    y=r(x)
    if active is None:actual,ref=y,expected
    else:actual,ref=y[:,active.long()],expected[:,active.long()]
    torch.testing.assert_close(actual,ref,atol=.01,rtol=.003)
    graph=torch.cuda.CUDAGraph();torch.cuda.synchronize()
    count=1 if cold else 8
    with torch.cuda.graph(graph):
        for _ in range(count):y=r(x)
    graph.replay();torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(3):
        times=[]
        for _ in range(3 if cold else 1):
            if cold:measure.eviction.add_(1);x.mul_(1)
            a.record();graph.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000/count)
        samples.append(statistics.median(times))
    return {**c,'us':statistics.median(samples),'samples_us':samples,'cv':statistics.pstdev(samples)/statistics.mean(samples),'validated_full_output':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--family',choices=['old','new'],required=True)
    ap.add_argument('--tokens',default='1,2,4,8,16,32,64,128');ap.add_argument('--screen-only',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    meta=read(args.stage/'inputs.json')[args.model];spec=read(meta['shape_manifest'])
    root=Path(meta['source_root']);oldcases={x['layer']:x for x in read(root/'reference_cases.json')['layers']}
    oldplans=read(root/'reference_plans.json')['hygemm'];tokens=list(map(int,args.tokens.split(',')))
    identity={'library_sha256':sha(os.environ['HYGEMM_LIBRARY']),'selection_sha256':meta['selection_sha256'],
              'shape_sha256':sha(meta['shape_manifest']),'source_sha256':sha(__file__),
              'layout':'MX K32 E8M0 native physical scales v1','family':args.family,'buckets':tokens,
              'pack_mode':os.environ.get('HYGEMM_PACK_MODE','0'),'reference_plans_sha256':sha(root/'reference_plans.json')}
    if (args.out/'identity.json').exists():assert read(args.out/'identity.json')==identity,'Stale tuning cache'
    write(args.out/'identity.json',identity)
    groups=defaultdict(dict);inherited=defaultdict(list)
    for c in spec['layers']:
        key=(c['out'],c['k'],c['high']);purity=0 if not c['high'] else 1 if c['high']==c['out'] else 2
        groups[(c['out'],c['k'],purity)][key]=c
        for t in tokens:
            rows=sorted([p for p in oldplans if (p['out'],p['k'],p['high'])==(c['out'],c['k'],oldcases[c['layer']]['high'])],key=lambda p:p['tokens'])
            v=cfg(next(p for p in rows if p['tokens']>=t))
            if v not in inherited[(*key,t)]:inherited[(*key,t)].append(v)
    measure.eviction=torch.zeros(256*1024**2,device='cuda',dtype=torch.uint8)
    raw=(args.out/'raw.jsonl').open('a',buffering=1)
    short=read(args.out/'shortlists.json') if (args.out/'shortlists.json').exists() else {}
    result=read(args.out/'plans.json') if (args.out/'plans.json').exists() else {'hygemm':[],'identity':identity}
    done={(p['out'],p['k'],p['high'],p['tokens']) for p in result['hygemm']}
    def log(phase,key,score):raw.write(json.dumps({'phase':phase,'key':key,**score})+'\n')
    for group,cases in groups.items():
        o,k,purity=group;representative=sorted(cases.values(),key=lambda c:c['high'])[len(cases)//2]
        w=(torch.randn(o,k,device='cuda')*.04).half();gkey=str(group)
        for c in [representative]+[c for c in cases.values() if c is not representative]:
            key=(o,k,c['high']);r=DualLinear(w,torch.tensor(c['high_ids'],device='cuda',dtype=torch.long))
            rw=torch.empty(o,k,device='cuda',dtype=torch.float32)
            if r.lo.numel():rw[r.lo.long()]=decode_mx(r.w4,r.sf4,k,True)
            if r.hi.numel():rw[r.hi.long()]=decode_mx(r.w8,r.sf8,k,False)
            for t in tokens:
                skey=gkey+'_'+str(t);full=(*key,t)
                if full in done:continue
                x=torch.randn(t,k,device='cuda',dtype=torch.float16);q,sf=pack(x)
                expected=(decode_mx(q,sf,k,False)@rw.t()).half()
                if skey not in short:
                    top={};lo,hi=r.lo,r.hi
                    candidates=list(range(28))
                    if args.family=='new':candidates+=list(range(35,45))+ (list(range(28,35)) if t<=4 else [])
                    for part,active in [('low',lo),('high',hi)]:
                        if not active.numel():top[part]=[0];continue
                        r.lo,r.hi=(lo,hi[:0]) if part=='low' else (lo[:0],hi)
                        scores=[]
                        for variant in candidates:
                            config={'low_id':variant if part=='low' else 0,'high_id':variant if part=='high' else 0,'policy':'serial'}
                            score=measure(r,x,config,expected,active);scores.append(score);log('isolated_resident_'+part,full,score)
                        # Cold-weight shortlist includes resident top three plus all distinct families' best.
                        ranked=sorted(scores,key=lambda s:s['us']);screen=ranked[:3]
                        if args.family=='new':
                            for lower,upper in [(0,8),(8,28),(28,35),(35,45)]:
                                subset=[s for s in ranked if lower<=s[part+'_id']<upper]
                                if subset and subset[0] not in screen:screen.append(subset[0])
                        cold=[]
                        for s in screen:
                            score=measure(r,x,cfg(s),expected,active,True);cold.append(score);log('isolated_cold_'+part,full,score)
                        top[part]=[s[part+'_id'] for s in sorted(cold,key=lambda s:s['us'])[:2]]
                    r.lo,r.hi=lo,hi
                    policies=['serial','concurrent'] if purity==2 else ['serial']
                    if args.family=='new' and purity==2:policies+=['high_first','high_main']
                    configs=[{'low_id':a,'high_id':b,'policy':p} for a in top['low'] for b in top['high'] for p in policies]
                    if args.family=='new' and t<=4:
                        configs += [{'low_id':28,'high_id':28,'policy':p} for p in ['gemv_pair1','gemv_pair2','gemv_pair4']]
                    configs += [a for a in inherited[full] if a not in configs]
                    scores=[]
                    for config in configs:
                        s=measure(r,x,config,expected,cold=True);scores.append(s);log('pair_cold',full,s)
                    short[skey]=sorted(scores,key=lambda s:s['us'])[:3];write(args.out/'shortlists.json',short)
                    print('screen',args.family,full,short[skey][0],flush=True)
                if args.screen_only:continue
                configs=inherited[full]+[cfg(a) for a in short[skey] if cfg(a) not in inherited[full]]
                scores=[]
                for config in configs:
                    s=measure(r,x,config,expected,cold=True);scores.append(s);log('exact_cold',full,s)
                base=min(scores[:len(inherited[full])],key=lambda s:s['us']);best=min(scores,key=lambda s:s['us'])
                chosen=best if base['us']/best['us']>=1.02 else base
                result['hygemm'].append({'out':o,'k':k,'high':c['high'],'tokens':t,**chosen})
                write(args.out/'plans.json',result);done.add(full)
            r.close();del r,rw;gc.collect();torch.cuda.empty_cache()
            if args.screen_only:break
        del w;gc.collect();torch.cuda.empty_cache()
    write(args.out/'complete.json',{'passed':True,'identity':identity,'plans':len(done),'screen_only':args.screen_only,
        'resident_replays':3,'resident_calls_per_replay':8,'cold_replays':9,'cold_group_medians':3,
        'reference':'all rows/tokens, atol .01 rtol .003','model_confirmation_required':True})


if __name__=='__main__':main()
