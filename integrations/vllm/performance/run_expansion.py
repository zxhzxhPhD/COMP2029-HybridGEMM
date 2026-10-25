"""Apply the bounded stage-two search and all-batch gates to another model."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import read,write,sha,runtime_sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
ap.add_argument('--resume',action='store_true');args=ap.parse_args();stage=args.stage;model=args.model;folder=stage/model
src=Path(__file__).resolve().parent
old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so';new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=new,HYGEMM_PACK_MODE='0',
    HYGEMM_FUSE_SILU='0',HYGEMM_ROW_ORDER='none',HYGEMM_SILU_MATH='0',HYGEMM_FUSED_GATEUP='0',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
history_file=folder/'expansion_jobs.json';history=read(history_file) if args.resume and history_file.exists() else []
def run(label,cmd,extra=None,required=True):
    effective={**env,**(extra or {})}
    inputs={flag:sha(cmd[cmd.index(flag)+1]) for flag in ['--plans','--proposal','--compare'] if flag in cmd}
    identity={'label':label,'command':cmd,'options':{k:v for k,v in effective.items() if k.startswith(('HYGEMM_','MIXLLM_'))},
        'script_sha256':sha(cmd[1]),'runtime_sha256':runtime_sha(),
        'binary_sha256':sha(effective['HYGEMM_LIBRARY']),'mix_sha256':sha(effective['MIXLLM_LIBRARY']),
        'producer_sha256':sha(effective['HYGEMM_PRODUCER_LIBRARY']),'input_hashes':inputs,'frozen_inputs_sha256':sha(stage/'inputs.json')}
    if args.resume and any(r['identity']==identity and r['returncode']==0 for r in history):return 0
    write(stage/'expansion_status.json',{'phase':label,'model':model,'command':cmd,'updated':time.time()})
    with (folder/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env=effective,stdout=f,stderr=subprocess.STDOUT)
    history.append({'identity':identity,'returncode':ret.returncode,'completed':time.time()});write(history_file,history)
    if ret.returncode and required:
        write(stage/'expansion_status.json',{'phase':'failed','model':model,'job':label,'exit':ret.returncode});raise SystemExit(ret.returncode)
    print('Completed',model,label,ret.returncode,flush=True);return ret.returncode
def bench(label,kind='hygemm',plans=None,extra=None):
    cmd=[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model',model,'--label',label,'--kind',kind]
    if plans:cmd+=['--plans',str(plans)]
    return run(label,cmd,extra)
def quality(label,plans=None,compare=None,extra=None,required=True):
    cmd=[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model',model,'--label',label]
    if plans:cmd+=['--plans',str(plans)]
    if compare:cmd+=['--compare',str(compare)]
    return run(label,cmd,extra,required)
for family,lib in [('old',old),('new',new)]:
    run('tune_'+family,[sys.executable,str(src/'tune.py'),'--stage',str(stage),'--model',model,
        '--family',family,'--out',str(folder/('tune_'+family))],{'HYGEMM_LIBRARY':lib})
run('tune_mix',[sys.executable,str(src/'tune_mix.py'),'--stage',str(stage),'--model',model])
bench('hy_pre_f_inherited',extra={'HYGEMM_LIBRARY':old})
bench('hy_pre_f_tuned',plans=folder/'tune_old/plans.json',extra={'HYGEMM_LIBRARY':old})
bench('new_compute',plans=folder/'tune_new/plans.json')
quality('quality_old',extra={'HYGEMM_LIBRARY':old})
quality('quality_new',plans=folder/'tune_new/plans.json',compare=folder/'quality_old/quality.json',required=False)
base={(r['batch'],r['context']):r['median_ms_per_step'] for r in read(folder/'hy_pre_f_inherited/result.json')['cells']}
oldrows=read(folder/'hy_pre_f_tuned/result.json')['cells'];ratios=[base[(r['batch'],r['context'])]/r['median_ms_per_step'] for r in oldrows]
use_old_tuned=math.prod(ratios)**(1/len(ratios))>=1.01 and min(ratios)>=1/1.02
baseline_label='hy_pre_f_tuned' if use_old_tuned else 'hy_pre_f_inherited'
if use_old_tuned:base={(r['batch'],r['context']):r['median_ms_per_step'] for r in oldrows}
newrows=read(folder/'new_compute/result.json')['cells'];qpath=folder/'quality_new/quality.json'
qok=qpath.exists() and read(qpath).get('comparison',{}).get('passed',False)
quality_buckets={}
if qpath.exists():
    qnew=read(qpath);qold=read(folder/'quality_old/quality.json');delta={}
    for a,b in zip(qnew['records'],qold['records']):
        assert a['prefill_schedule_sha256']==b['prefill_schedule_sha256']
        values=delta.setdefault(a['batch'],[])
        for i,(x,y) in enumerate(zip(a['logprobs'],b['logprobs'])):
            for j in sorted(set(a['full_target_indices'][i])&set(b['full_target_indices'][i]))[:32]:values.append(abs(x[j]-y[j]))
    quality_buckets={t:{'max_abs':max(v),'mean_abs':math.fsum(v)/len(v),'passed':max(v)<=.05 and math.fsum(v)/len(v)<=.002}
                     for t,v in delta.items()}
buckets=[]
for t in [1,2,4,8,16,32,64,128]:
    if quality_buckets.get(t,{}).get('passed',False):
        ratios=[base[(t,r['context'])]/r['median_ms_per_step'] for r in newrows if r['batch']==t]
        if math.prod(ratios)**.5>=1.01 and min(ratios)>=1/1.02:buckets.append(t)
# Preserve the chosen old baseline where new compute does not pass both gates.
proposal=read(folder/'tune_new/plans.json')
if use_old_tuned:
    oldmap={(r['out'],r['k'],r['high'],r['tokens']):r for r in read(folder/'tune_old/plans.json')['hygemm']}
    proposal['hygemm']=[r if r['tokens'] in buckets else oldmap[(r['out'],r['k'],r['high'],r['tokens'])] for r in proposal['hygemm']]
    bind_buckets=[1,2,4,8,16,32,64,128]
else:bind_buckets=buckets
write(folder/'combined_proposal.json',proposal)
plan=folder/'combination_plans.json'
run('bind_combination',[sys.executable,str(src/'bind_plan.py'),'--stage',str(stage),'--model',model,
    '--proposal',str(folder/'combined_proposal.json'),'--buckets',','.join(map(str,bind_buckets)),'--library',new,'--out',str(plan)])
final_env={'HYGEMM_PACK_MODE':'3','HYGEMM_FUSE_SILU':'1'}
write(folder/'selection_decision.json',{'baseline':baseline_label,'old_retuned_adopted':use_old_tuned,
    'new_quality_passed':qok,'quality_by_bucket':quality_buckets,'new_compute_buckets':buckets,'pack_mode':3,'fused_silu':True,'plans':str(plan),
    'status':'candidate pending complete-model performance and quality gates'})
bench('combination_full',plans=plan,extra=final_env)
bench('combination_repeat',plans=plan,extra=final_env)
quality('quality_combination',plans=plan,compare=folder/'quality_old/quality.json',extra=final_env)
run('validation_ppl',[sys.executable,str(src/'ppl.py'),'--stage',str(stage),'--model',model,
    '--model-path',str(folder/'combination_full/model'),'--out',str(folder/'validation_ppl.json')],final_env)
bench('mix_original',kind='mixllm')
bench('mix_tuned',kind='mixllm',plans=folder/'tune_mix/plans.json')
bench('fp16',kind='fp16')
bench('hy_pre_i',kind='hy_pre_i',extra={'HYGEMM_LIBRARY':old})
write(stage/'expansion_status.json',{'phase':'model_complete','model':model,'updated':time.time()})
