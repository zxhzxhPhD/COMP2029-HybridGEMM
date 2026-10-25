"""Follow the isolated row-order lead and finish producer memory validation."""
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
    HYGEMM_PACK_MODE='0',HYGEMM_FUSE_SILU='0',HYGEMM_ROW_ORDER='none',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'),
    MIXLLM_LIBRARY='/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')
while True:
    file=stage/'round3_status.json';state=json.loads(file.read_text()) if file.exists() else {}
    if state.get('phase')=='complete_7B_gate':break
    if state.get('phase')=='failed':
        repair=stage/'quality_repair_status.json'
        if not repair.exists() or json.loads(repair.read_text()).get('phase')=='failed':raise RuntimeError(state)
    time.sleep(5)
def run(label,cmd,extra=None):
    (stage/'round4_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:
        (stage/'round4_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':ret.returncode},indent=2));raise SystemExit(ret.returncode)
    print('Completed',label,flush=True)
run('check_row_order',[sys.executable,str(src/'check_row_order.py'),'--out',str(stage/'checks/row_order.json')])
run('check_extremes',[sys.executable,str(src/'check_extremes.py'),'--out',str(stage/'checks/extremes.json')])
for tool in ['memcheck','synccheck','racecheck']:
    run('producer_'+tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',
        sys.executable,str(src/'check_producer.py'),'--library',env['HYGEMM_PRODUCER_LIBRARY'],'--maths','0',
        '--out',str(stage/'checks'/('producer_'+tool+'.json'))])
run('row_reverse_model',[sys.executable,str(src/'benchmark.py'),'--stage',str(stage),'--model','7B','--label','row_reverse_model',
    '--batches','1,8,32,64,128'],{'HYGEMM_ROW_ORDER':'reverse'})
(stage/'round4_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
