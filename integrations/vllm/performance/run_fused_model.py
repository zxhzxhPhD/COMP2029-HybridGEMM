"""One bounded serving trial follows the positive gate/up TMA-switch probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=str(stage/'build/libhygemm_serving.so'),
    HYGEMM_PACK_MODE='0',HYGEMM_FUSE_SILU='0',HYGEMM_ROW_ORDER='none',HYGEMM_FUSED_GATEUP='8',
    HYGEMM_FUSED_LIBRARY=str(stage/'fused_serving_build/libfused_probe.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
while True:
    file=stage/'round4_status.json';state=json.loads(file.read_text()) if file.exists() else {}
    if state.get('phase')=='complete':break
    if state.get('phase')=='failed':raise RuntimeError(state)
    time.sleep(5)
def run(label,cmd,required=True):
    (stage/'fused_model_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode and required:
        (stage/'fused_model_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':ret.returncode},indent=2));raise SystemExit(ret.returncode)
    print('Completed',label,ret.returncode,flush=True)
run('check_fused_rebind',[sys.executable,str(src/'check_fused.py'),'--out',str(stage/'checks/fused_rebind.json')])
for tool in ['memcheck','synccheck','racecheck']:
    run('fused_'+tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',sys.executable,
        str(src/'check_fused.py'),'--sanitizer','--out',str(stage/'checks'/('fused_'+tool+'.json'))])
run('fused_gateup_model',[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label','fused_gateup_model',
    '--batches','1,2,4,8,32,64,128'])
run('quality_fused_gateup',[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model','7B','--label','quality_fused_gateup',
    '--compare',str(stage/'7B/quality_old/quality.json')],required=False)
(stage/'fused_model_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
