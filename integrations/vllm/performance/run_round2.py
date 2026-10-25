"""Serial diagnostic queue after the first model screen; no automatic adoption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
ap.add_argument('--repair',action='store_true');args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'
new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=new,HYGEMM_PACK_MODE='0',
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer.so'))
if args.repair:env['HYGEMM_PRODUCER_LIBRARY']=str(stage/'build/libhygemm_producer_v2.so')
while True:
    state=json.loads((stage/'round1_status.json').read_text())
    if state['phase']=='complete':break
    if state['phase']=='failed':raise RuntimeError(state)
    time.sleep(5)
def run(label,cmd,extra=None,required=True):
    (stage/'round2_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:
        result=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if result.returncode and required:
        (stage/'round2_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':result.returncode},indent=2))
        raise SystemExit(result.returncode)
    print('Completed',label,result.returncode,flush=True)
    return result.returncode
run('check_producer',[sys.executable,str(src/'check_producer.py'),'--library',env['HYGEMM_PRODUCER_LIBRARY'],
                     '--out',str(stage/'checks/producer.json')])
run('mechanism_probes',[sys.executable,str(src/'probes.py'),'--stage',str(stage)])
sanitizer='/usr/local/cuda-12.8/bin/compute-sanitizer'
for tool in ([] if args.repair else ['memcheck','synccheck','racecheck']):
    run('variants_'+tool,[sanitizer,'--tool',tool,'--error-exitcode','99',sys.executable,
        str(src.parent/'optimization/check_light.py'),'--ids',','.join(map(str,range(28,45))),
        '--sanitizer','--out',str(stage/'checks'/('variants_'+tool+'.json'))])
if not args.repair:
    run('pack_policy_memcheck',[sanitizer,'--tool','memcheck','--error-exitcode','99',sys.executable,
        str(src/'check.py'),'--sanitizer','--old-library',old,'--out',str(stage/'checks/pack_policy_memcheck.json')])
run('ncu_permission_probe',['/root/autodl-tmp/cuda-13.0/bin/ncu','--target-processes','all',
    '--kernel-name-base','demangled','--kernel-name','regex:.*gemv.*','--launch-count','1','--set','basic','--export',str(stage/'checks/ncu'),
    '--force-overwrite',sys.executable,str(src.parent/'optimization/check_light.py'),'--ids','28','--smoke',
    '--out',str(stage/'checks/ncu_check.json')],required=False)
for label,lib,plans,compare in [('quality_old',old,None,None),('quality_new',new,stage/'7B/tune_new/plans.json',stage/'7B/quality_old/quality.json')]:
    cmd=[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model','7B','--label',label]
    if plans:cmd+=['--plans',str(plans)]
    if compare:cmd+=['--compare',str(compare)]
    run(label,cmd,{'HYGEMM_LIBRARY':lib},required=label=='quality_old')
passing=json.loads((stage/'checks/producer.json').read_text())['passing_math_modes']
if passing:
    math=str(passing[0])
    run('producer_model',[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label','producer_model',
        '--batches','1,8,32,64,128'],{'HYGEMM_FUSE_SILU':'1','HYGEMM_SILU_MATH':math})
    run('quality_producer',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model','7B','--label','quality_producer',
        '--compare',str(stage/'7B/quality_old/quality.json')],{'HYGEMM_FUSE_SILU':'1','HYGEMM_SILU_MATH':math},required=False)
(stage/'round2_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
