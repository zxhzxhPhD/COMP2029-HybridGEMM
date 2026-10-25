"""Finish bucket refinement, comparable Mix gates, ablations and final profiling."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import read,write
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=str(stage/'build/libhygemm_serving.so'),
    HYGEMM_PACK_MODE='0',HYGEMM_FUSE_SILU='0',HYGEMM_SILU_MATH='0',HYGEMM_ROW_ORDER='none',HYGEMM_FUSED_GATEUP='0',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
while True:
    state=read(stage/'remaining_status.json')
    if state['phase']=='complete':break
    if state['phase']=='failed':raise RuntimeError(state)
    time.sleep(5)
def run(label,cmd,extra=None,required=True):
    write(stage/'acceptance_status.json',{'phase':label,'command':cmd,'updated':time.time()})
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode and required:
        write(stage/'acceptance_status.json',{'phase':'failed','job':label,'exit':ret.returncode});raise SystemExit(ret.returncode)
    print('Completed',label,ret.returncode,flush=True)
    return ret.returncode
def cellmap(path):return {(r['batch'],r['context']):r['median_ms_per_step'] for r in read(path)['cells']}
run('refine_tiling',[sys.executable,str(src/'refine_tiling.py'),'--stage',str(stage)])
mix_choices={}
for model in ['0.5B','1.5B','3B','7B']:
    folder=stage/model;original=cellmap(folder/'mix_original/result.json');tuned=cellmap(folder/'mix_tuned/result.json')
    ratios=[original[k]/tuned[k] for k in original];gain=math.prod(ratios)**(1/len(ratios));choice='mix_original'
    record={'first_gain':gain,'first_max_latency_ratio':1/min(ratios),'selected':choice}
    if gain>=1.01 and min(ratios)>=1/1.02:
        run(model+'_mix_repeat',[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model',model,'--kind','mixllm',
            '--label','mix_repeat','--plans',str(folder/'tune_mix/plans.json')])
        repeat=cellmap(folder/'mix_repeat/result.json');values=[original[k]/repeat[k] for k in original]
        record['repeat_gain']=math.prod(values)**(1/len(values));record['repeat_max_latency_ratio']=1/min(values)
        if record['repeat_gain']>1 and min(values)>=1/1.02:
            run(model+'_mix_quality_original',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model',model,
                '--kind','mixllm','--label','mix_quality_original'])
            ret=run(model+'_mix_quality_tuned',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model',model,
                '--kind','mixllm','--label','mix_quality_tuned','--plans',str(folder/'tune_mix/plans.json'),
                '--compare',str(folder/'mix_quality_original/quality.json')],required=False)
            if not ret:choice='mix_tuned'
    record['selected']=choice;mix_choices[model]=record;write(stage/'mix_decisions.json',mix_choices)
    cmd=[sys.executable,str(src/'freeze_model.py'),'--stage',str(stage),'--model',model,'--mix-label',choice]
    if (folder/'refinement_decision.json').exists():
        selected=read(folder/'refinement_decision.json')
        for flag,key in [('--baseline','baseline'),('--candidate','candidate'),('--repeat','repeat'),('--quality','quality')]:cmd += [flag,selected[key]]
    run(model+'_freeze',cmd)
    run(model+'_bindings',[sys.executable,str(src/'check_binding.py'),'--stage',str(stage),'--model',model])
for label,extra in [('ablation_no_silu',{'HYGEMM_PACK_MODE':'3'}),('ablation_no_pack',{'HYGEMM_FUSE_SILU':'1'})]:
    run(label,[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label',label,
        '--plans',str(stage/'7B/combination_plans.json')],extra)
run('prepare_final_profile',[sys.executable,str(src/'make_profile_view.py'),'--stage',str(stage),'--mix',mix_choices['7B']['selected']])
run('profile_final',[sys.executable,str(src.parent/'profiling/run_profiles.py'),'--formal',str(stage/'profile_final_view'),
    '--out',str(stage/'profile_final')],{'HYGEMM_PACK_MODE':'3','HYGEMM_FUSE_SILU':'1'})
run('final_tables',['/root/miniconda3/bin/python',str(src/'analyze.py'),'--stage',str(stage)])
write(stage/'acceptance_status.json',{'phase':'complete','updated':time.time()})
