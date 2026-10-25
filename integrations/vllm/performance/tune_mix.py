"""Retune only original MixLLM device candidates; keep device code immutable."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import torch
from prepare import read,write,sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'formal'))
from tune import measure,make_reference,input_reference
from hygemm_vllm.mixllm import MixLinear,load_library


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    args=ap.parse_args();out=args.stage/args.model/'tune_mix';out.mkdir(parents=True,exist_ok=True)
    old=Path('/root/autodl-tmp/20261024_235427_formal_decode')/args.model
    cases=read(old/'cases.json')['cases'];original=read(old/'plans.json')['mixllm'];tokens=[1,2,4,8,16,32,64,128]
    torch.set_num_threads(4);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False;load_library()
    identity={'source_sha256':sha(__file__),'cases_sha256':sha(old/'cases.json'),'original_plans_sha256':sha(old/'plans.json'),
        'library_sha256':sha(os.environ['MIXLLM_LIBRARY']),'buckets':tokens,'device_unchanged':True}
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    write(out/'identity.json',identity)
    groups=defaultdict(list)
    for c in cases:groups[(c['out'],c['k'],c['high'] in [0,c['out']])].append(c)
    shared=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    legal={};inventory={}
    for pure in [False,True]:
        legal[pure]=[];inventory[str(pure)]=[]
        for stage in [5,11]:
            for i in range(1,60):
                info=torch.ops.kernels_mixllm.config_info(stage,i,pure)
                if not info:continue
                ok=max(info[5:7])<=shared and info[7]<=1024
                inventory[str(pure)].append({'stage':stage,'id':i,'info':info,'legal':ok})
                if ok:legal[pure].append({'stage':stage,'id':i})
    write(out/'inventory.json',inventory)
    short=read(out/'shortlists.json') if (out/'shortlists.json').exists() else {}
    result=read(out/'plans.json') if (out/'plans.json').exists() else {'mixllm':[r for r in original if r['tokens']>128],'identity':identity}
    done={(r['out'],r['k'],r['high'],r['tokens']) for r in result['mixllm']}
    raw=(out/'raw.jsonl').open('a',buffering=1)
    for group,items in groups.items():
        o,k,pure=group;items=sorted(items,key=lambda c:c['high']);rep=items[len(items)//2]
        w=(torch.randn(o,k,device='cuda')*.04).half()
        for c in [rep]+[c for c in items if c is not rep]:
            r=MixLinear(w,torch.tensor(c['high_ids'],device='cuda',dtype=torch.long));indices,rw=make_reference(r,'mixllm')
            for t in tokens:
                full=(o,k,c['high'],t);key=str((*group,t))
                if full in done:continue
                x=torch.randn(t,k,device='cuda',dtype=torch.float16);tids,dx=input_reference(x,'mixllm');expected=(dx@rw.t()).half()
                if key not in short:
                    scores=[]
                    for cfg in legal[pure]:
                        s=measure(r,x,'mixllm',cfg,expected,indices,tids);scores.append(s)
                        raw.write(json.dumps({'phase':'screen','key':full,**s})+'\n')
                    short[key]=sorted(scores,key=lambda s:s['us'])[:3];write(out/'shortlists.json',short)
                inherited=next(p for p in original if (p['out'],p['k'],p['high'],p['tokens'])==full)
                configs=[{'stage':inherited['stage'],'id':inherited['id']}]
                for p in short[key]:
                    config={'stage':p['stage'],'id':p['id']}
                    if config not in configs:configs.append(config)
                scores=[]
                for config in configs:
                    s=measure(r,x,'mixllm',config,expected,indices,tids);scores.append(s)
                    raw.write(json.dumps({'phase':'refine','key':full,**s})+'\n')
                best=min(scores,key=lambda s:s['us']);chosen=best if scores[0]['us']/best['us']>=1.02 else scores[0]
                result['mixllm'].append({'out':o,'k':k,'high':c['high'],'tokens':t,**chosen});write(out/'plans.json',result);done.add(full)
            del r,rw,x,expected,dx;gc.collect();torch.cuda.empty_cache()
        del w;gc.collect();torch.cuda.empty_cache();print('Mix tuned',group,flush=True)
    write(out/'complete.json',{'passed':True,'identity':identity,'plans':len(result['mixllm']),
        'reference':'existing validated original device candidates, sampled rows/tokens; serving model confirmation required'})


if __name__=='__main__':main()
