"""Keep useful old-kernel bucket gains even when whole-plan replacement loses."""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from prepare import read,write,sha
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so';new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=new,HYGEMM_PACK_MODE='3',
    HYGEMM_FUSE_SILU='1',HYGEMM_SILU_MATH='0',HYGEMM_ROW_ORDER='none',HYGEMM_FUSED_GATEUP='0',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
def run(model,label,cmd,extra=None):
    write(stage/'refine_status.json',{'phase':label,'model':model,'command':cmd})
    with (stage/model/(label+'.log')).open('w') as f:r=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:write(stage/'refine_status.json',{'phase':'failed','model':model,'job':label});raise SystemExit(r.returncode)
def cells(folder,label):return {(r['batch'],r['context']):r['median_ms_per_step'] for r in read(folder/label/'result.json')['cells']}
records=[]
for model in ['0.5B','1.5B','3B']:
    folder=stage/model;decision=read(folder/'selection_decision.json')
    if decision['old_retuned_adopted']:continue
    base=cells(folder,'hy_pre_f_inherited');tuned=cells(folder,'hy_pre_f_tuned');buckets=[]
    for t in [1,2,4,8,16,32,64,128]:
        ratios=[base[(t,c)]/tuned[(t,c)] for c in [128,1024]]
        if math.prod(ratios)**.5>=1.01 and min(ratios)>=1/1.02:buckets.append(t)
    if not buckets:continue
    baseline=folder/'refined_baseline_plans.json'
    run(model,'bind_refined_baseline',[sys.executable,str(src/'bind_plan.py'),'--stage',str(stage),'--model',model,
        '--proposal',str(folder/'tune_old/plans.json'),'--buckets',','.join(map(str,buckets)),'--library',old,'--out',str(baseline)])
    plan=read(folder/'combination_plans.json');oldmap={(r['out'],r['k'],r['high'],r['tokens']):r for r in read(folder/'tune_old/plans.json')['hygemm']}
    replaced=[]
    for t in buckets:
        if t not in decision['new_compute_buckets']:replaced.append(t)
    for row in plan['layer_plans']:
        if row['tokens'] in replaced:
            chosen=oldmap[(row['out'],row['k'],row['high'],row['tokens'])]
            row.update({k:chosen[k] for k in ['low_id','high_id','policy']})
    unique={}
    for row in plan['layer_plans']:unique.setdefault((row['out'],row['k'],row['high'],row['tokens']),{k:v for k,v in row.items() if k!='layer'})
    plan['hygemm']=list(unique.values());plan['binding']['refinement']={'old_compute_buckets':replaced,
        'source_old_plans_sha256':sha(folder/'tune_old/plans.json'),'previous_plan_sha256':sha(folder/'combination_plans.json')}
    finalplan=folder/'refined_plans.json';write(finalplan,plan)
    for label,plans,extra in [('refined_baseline',baseline,{'HYGEMM_LIBRARY':old,'HYGEMM_PACK_MODE':'0','HYGEMM_FUSE_SILU':'0'}),
                              ('refined_full',finalplan,{}),('refined_repeat',finalplan,{})]:
        run(model,label,[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model',model,'--label',label,'--plans',str(plans)],extra)
    run(model,'quality_refined',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model',model,'--label','quality_refined',
        '--plans',str(finalplan),'--compare',str(folder/'quality_old/quality.json')])
    control=cells(folder,'combination_full');first=cells(folder,'refined_full');second=cells(folder,'refined_repeat')
    targeted=[k for k in control if k[0] in replaced]
    gains=[math.prod(control[k]/r[k] for k in targeted)**(1/len(targeted)) for r in [first,second]] if targeted else [1,1]
    regress=[max(r[k]/control[k] for k in control) for r in [first,second]]
    accepted=all(g>1 for g in gains) and gains[0]>=1.01 and max(regress)<=1.02
    record={'model':model,'baseline_buckets':buckets,'refined_buckets':replaced,'targeted_gains':gains,
        'max_latency_ratio_vs_previous':regress,'accepted':accepted,'baseline':'refined_baseline',
        'candidate':'refined_full' if accepted else 'combination_full',
        'repeat':'refined_repeat' if accepted else 'combination_repeat',
        'quality':'quality_refined' if accepted else 'quality_combination'}
    write(folder/'refinement_decision.json',record);records.append(record)
write(stage/'refinement_summary.json',records);write(stage/'refine_status.json',{'phase':'complete'})
