"""Resumeable sequential model-by-model calibration, tuning and evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--models',type=Path,required=True)
ap.add_argument('--sizes',default='0.5B,1.5B,3B,7B')
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
src=Path(__file__).resolve().parent
protocol_path=args.out/'protocol.json'
protocol=json.loads((src/'protocol.json').read_text())
if protocol_path.exists():
    assert json.loads(protocol_path.read_text())==protocol,'Output protocol differs from this runner'
else:
    protocol_path.write_text(json.dumps(protocol,indent=2))
state={'started':time.time(),'completed':[],'phase':'starting'}
status=args.out/'pipeline_status.json'
def run(size,phase,script,flags,timeout=7200):
    state.update(model=size,phase=phase,updated=time.time());status.write_text(json.dumps(state,indent=2))
    log=args.out/f'{phase}_{size}.log'
    command=[sys.executable,str(src/script),*map(str,flags)]
    with log.open('a') as f:
        f.write('\n$ '+' '.join(command)+'\n');f.flush()
        r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode:
        state.update(phase='failed',failed_phase=phase,returncode=r.returncode,log=str(log),updated=time.time())
        status.write_text(json.dumps(state,indent=2));raise RuntimeError(str(state))
    print(size,phase,'complete',flush=True)

for i,size in enumerate(args.sizes.split(',')):
    out=args.out/size;model=args.models/f'Qwen2.5-{size}'
    if not (out/'calibration/manifest.json').exists():
        run(size,'calibrate','calibrate.py',['--model',model,'--train',args.out/'data/wikitext2_train.txt',
            '--test',args.out/'data/wikitext2_test.txt','--out',out/'calibration'])
    if not (out/'cases.json').exists():
        run(size,'prepare','prepare.py',['--model',model,'--calibration',out/'calibration','--out',out])
    if not (out/'tuning_complete.json').exists():run(size,'tune','tune.py',['--out',out])
    order=['fp16','mixllm','hygemm'];order=order[i%3:]+order[:i%3]
    for kind in order:run(size,'benchmark_'+kind,'benchmark.py',['--out',out,'--kind',kind,'--resume'])
    state['completed'].append(size);status.write_text(json.dumps(state,indent=2))
state.update(phase='complete',updated=time.time());status.write_text(json.dumps(state,indent=2))
print('Matrix complete',flush=True)
