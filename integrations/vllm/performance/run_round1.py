"""Serial GPU queue: fresh profile precedes correctness and bounded tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'
new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=new,
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
while json.loads((stage/'profile_initial/status.json').read_text())['phase']!='complete':time.sleep(5)
(stage/'checks').mkdir(exist_ok=True)
jobs=[('check_pack_policy',[sys.executable,str(src/'check.py'),'--old-library',old,'--out',str(stage/'checks/pack_policy.json')],new,0),
 ('check_variants',[sys.executable,str(src.parent/'optimization/check_light.py'),'--ids',','.join(map(str,range(28,45))),
                    '--out',str(stage/'checks/variants.json')],new,0)]
jobs.append(('check_producer',[sys.executable,str(src/'check_producer.py'),'--library',str(stage/'build/libhygemm_producer.so'),
                            '--out',str(stage/'checks/producer.json')],new,0))
for family,lib in [('old',old),('new',new)]:
    cmd=[sys.executable,str(src/'tune.py'),'--stage',str(stage),'--model','7B','--out',str(stage/'7B'/('tune_'+family)),
         '--family',family]
    jobs.append(('tune_'+family,cmd,lib,0))
for label,lib,mode,plan in [('hy_pre_f_inherited',old,0,None),('hy_pre_f_tuned',old,0,'tune_old'),
                           ('pack_padding',new,1,None),('pack_vector',new,2,None),('new_compute',new,0,'tune_new')]:
    cmd=[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label',label,
         '--batches','1,8,32,64,128']
    if plan:cmd+=['--plans',str(stage/'7B'/plan/'plans.json')]
    jobs.append((label,cmd,lib,mode))
for label,cmd,lib,mode in jobs:
    (stage/'round1_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:
        result=subprocess.run(cmd,env=dict(env,HYGEMM_LIBRARY=lib,HYGEMM_PACK_MODE=str(mode)),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (stage/'round1_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':result.returncode},indent=2))
        raise SystemExit(result.returncode)
    print('Completed',label,flush=True)
(stage/'round1_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
