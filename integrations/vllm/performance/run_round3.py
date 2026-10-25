"""Confirm the 7B combination before expanding the same search to other models."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent;old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'
new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=new,HYGEMM_PACK_MODE='3',
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer.so'),HYGEMM_FUSE_SILU='0')
def read(p):return json.loads(Path(p).read_text())
while True:
    file=stage/'round2_status.json';state=read(file) if file.exists() else {}
    if state.get('phase')=='complete':break
    if state.get('phase')=='failed':raise RuntimeError(state)
    time.sleep(5)
def run(label,cmd,extra=None):
    (stage/'round3_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:
        (stage/'round3_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':ret.returncode},indent=2));raise SystemExit(ret.returncode)
    print('Completed',label,flush=True)
def bench(model,label,kind='hygemm',plans=None,extra=None):
    cmd=[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model',model,'--label',label,'--kind',kind]
    if plans:cmd+=['--plans',str(plans)]
    run(model+'_'+label,cmd,extra)
base=read(stage/'7B/hy_pre_f_inherited/result.json')['cells'];base={(r['batch'],r['context']):r['median_ms_per_step'] for r in base}
candidate=read(stage/'7B/new_compute/result.json')['cells'];buckets=[]
quality_file=stage/'7B/quality_new/quality.json'
quality_ok=quality_file.exists() and read(quality_file).get('comparison',{}).get('passed',False)
if quality_ok:
    for t in [1,8,32,64,128]:
        ratios=[base[(t,r['context'])]/r['median_ms_per_step'] for r in candidate if r['batch']==t]
        if math.prod(ratios)**(1/len(ratios))>=1.01 and min(ratios)>=1/1.02:buckets.append(t)
    if 1 in buckets:buckets+= [2,4] # exploratory adjacent buckets, admitted only after full-model gates
producer_file=stage/'7B/producer_model/result.json';producer_quality=stage/'7B/quality_producer/quality.json'
if producer_file.exists() and producer_quality.exists() and read(producer_quality).get('comparison',{}).get('passed',False):
    vals=read(producer_file)['cells'];ratios=[base[(r['batch'],r['context'])]/r['median_ms_per_step'] for r in vals]
    if math.prod(ratios)**(1/len(ratios))>=1.01 and min(ratios)>=1/1.02:
        env['HYGEMM_FUSE_SILU']='1';env['HYGEMM_SILU_MATH']=str(read(stage/'checks/producer.json')['passing_math_modes'][0])
        env['HYGEMM_PRODUCER_LIBRARY']=read(stage/'checks/producer.json')['library']
decision={'new_compute_buckets':sorted(buckets),'quality_new_passed':quality_ok,'pack_mode':3,
          'fused_silu':env['HYGEMM_FUSE_SILU'],'producer_math':env.get('HYGEMM_SILU_MATH'),
          'status':'candidate only; complete-model two-group regression gates pending'}
(stage/'combination_proposal.json').write_text(json.dumps(decision,indent=2))
plan=stage/'7B/combination_plans.json'
run('bind_7B',[sys.executable,str(src/'bind_plan.py'),'--stage',str(stage),'--model','7B','--proposal',str(stage/'7B/tune_new/plans.json'),
    '--buckets',','.join(map(str,sorted(buckets))),'--library',new,'--out',str(plan)])
bench('7B','baseline_full',extra={'HYGEMM_LIBRARY':old,'HYGEMM_PACK_MODE':'0','HYGEMM_FUSE_SILU':'0'})
bench('7B','combination_full',plans=plan)
bench('7B','combination_repeat',plans=plan)
run('quality_combination',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model','7B','--label','quality_combination',
    '--plans',str(plan),'--compare',str(stage/'7B/quality_old/quality.json')])
# Deliberately stop at this gate. The subsequent model expansion must inspect
# all cells, retain negative cases and rebind any rejected bucket choices.
(stage/'round3_status.json').write_text(json.dumps({'phase':'complete_7B_gate','updated':time.time()},indent=2))
