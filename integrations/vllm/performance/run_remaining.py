"""Serial expansion after the 7B gates; original device baselines remain intact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
ap.add_argument('--skip-cmake',action='store_true');args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=str(stage/'build/libhygemm_serving.so'),
    HYGEMM_PACK_MODE='0',HYGEMM_FUSE_SILU='0',HYGEMM_ROW_ORDER='none',HYGEMM_FUSED_GATEUP='0',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
def run(label,cmd,extra=None):
    (stage/'remaining_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:
        (stage/'remaining_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':ret.returncode},indent=2));raise SystemExit(ret.returncode)
    print('Completed',label,flush=True)
assert json.loads((stage/'fused_model_status.json').read_text())['phase']=='complete'
assert json.loads((stage/'7B/quality_combination/quality.json').read_text())['comparison']['passed']
# Verify the ordinary from-source build as well as the fast archive-reuse build.
if not args.skip_cmake:
    run('cmake_variants_smoke',[sys.executable,str(src.parent/'optimization/check_light.py'),'--ids','28,32,35,44','--smoke',
        '--out',str(stage/'checks/cmake_variants_smoke.json')],{'HYGEMM_LIBRARY':str(stage/'full_build/libhygemm_serving.so')})
    run('cmake_producer_parity',[sys.executable,str(src/'check_producer.py'),'--library',str(stage/'full_build/libhygemm_producer.so'),
        '--maths','0','--out',str(stage/'checks/cmake_producer.json')])
for model in ['0.5B','1.5B','3B']:
    run('expand_'+model,[sys.executable,str(src/'run_expansion.py'),'--stage',str(stage),'--model',model,'--resume'])
run('7B_tune_mix',[sys.executable,str(src/'tune_mix.py'),'--stage',str(stage),'--model','7B'])
for label,kind,plans,extra in [
    ('mix_original','mixllm',None,{}),('mix_tuned','mixllm',stage/'7B/tune_mix/plans.json',{}),('fp16','fp16',None,{}),
    ('hy_pre_i','hy_pre_i',None,{'HYGEMM_LIBRARY':'/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'})]:
    cmd=[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label',label,'--kind',kind]
    if plans:cmd+=['--plans',str(plans)]
    run('7B_'+label,cmd,extra)
run('7B_validation_ppl',[sys.executable,str(src/'ppl.py'),'--stage',str(stage),'--model','7B',
    '--model-path',str(stage/'7B/combination_full/model'),'--out',str(stage/'7B/validation_ppl.json')],
    {'HYGEMM_PACK_MODE':'3','HYGEMM_FUSE_SILU':'1','HYGEMM_SILU_MATH':'0'})
(stage/'remaining_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
