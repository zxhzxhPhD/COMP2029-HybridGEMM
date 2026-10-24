"""Frozen framework/selection, old and new HyGEMM, original MixLLM, FP16."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--old-library', type=Path, required=True)
ap.add_argument('--new-library', type=Path, required=True)
ap.add_argument('--mix-library', type=Path, required=True)
args = ap.parse_args()
benchmark = Path(__file__).resolve().parents[1] / 'formal/benchmark.py'
jobs = [('baseline', 'hygemm', args.old_library, False), ('optimized', 'hygemm', args.new_library, False),
        ('mixllm', 'mixllm', args.old_library, True), ('fp16', 'fp16', args.old_library, True)]
status = args.out / 'comparison_status.json'
for label, kind, lib, skip_ppl in jobs:
    target = args.out / label / f'result_{kind}.json'
    if target.exists() and json.loads(target.read_text()).get('completed'):
        continue
    command = [sys.executable, str(benchmark), '--out', str(args.out / label), '--kind', kind, '--resume']
    if skip_ppl:
        command.append('--skip-ppl')
    env = dict(os.environ, HYGEMM_LIBRARY=str(lib), MIXLLM_LIBRARY=str(args.mix_library))
    status.write_text(json.dumps({'phase': label, 'command': command, 'library': str(lib), 'updated': time.time()}, indent=2))
    with (args.out / f'benchmark_{label}.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        status.write_text(json.dumps({'phase': 'failed', 'failed_job': label, 'returncode': result.returncode}, indent=2))
        raise RuntimeError(label)
    print('Complete', label, flush=True)
for label, lib in [('baseline', args.old_library), ('optimized', args.new_library)]:
    target = args.out / f'decode_quality_{label}.json'
    if target.exists():
        continue
    command = [sys.executable, str(Path(__file__).with_name('decode_quality.py')),
               '--model-root', str(args.out / label), '--out', str(target)]
    status.write_text(json.dumps({'phase': 'quality_' + label, 'command': command, 'updated': time.time()}, indent=2))
    with (args.out / f'decode_quality_{label}.log').open('w') as log:
        subprocess.run(command, env=dict(os.environ, HYGEMM_LIBRARY=str(lib)), stdout=log, stderr=subprocess.STDOUT, check=True)
status.write_text(json.dumps({'phase': 'complete', 'updated': time.time()}, indent=2))
