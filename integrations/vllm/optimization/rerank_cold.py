"""Retain a completed resident-proxy attempt, then validate cold-weight plans."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--formal', type=Path, required=True)
ap.add_argument('--old-library', type=Path, required=True)
ap.add_argument('--new-library', type=Path, required=True)
ap.add_argument('--mix-library', type=Path, required=True)
args = ap.parse_args()
src = Path(__file__).resolve().parent
while True:
    state = json.loads((args.out / 'status.json').read_text())
    if state['phase'] == 'failed':
        raise RuntimeError(state)
    if state['phase'] == 'complete':
        break
    time.sleep(10)
folder = args.out / 'optimized'
archive = folder / 'resident_attempt'
if not archive.exists():
    archive.mkdir()
    for name in ['plans.json', 'shortlists.json', 'decisions.json', 'tuning_raw.jsonl', 'tuning_complete.json', 'result_hygemm.json']:
        (folder / name).rename(archive / name)
    (args.out / 'benchmark_optimized.log').rename(args.out / 'benchmark_optimized_resident.log')
    shutil.copy2(args.formal / 'plans.json', folder / 'plans.json')
env = dict(os.environ, HYGEMM_LIBRARY=str(args.new_library), MIXLLM_LIBRARY=str(args.mix_library))
if not (folder / 'tuning_complete.json').exists():
    (args.out / 'cold_status.json').write_text(json.dumps({'phase': 'tuning', 'updated': time.time()}))
    with (args.out / 'tune_cold.log').open('a') as log:
        subprocess.run([sys.executable, str(src / 'tune_small.py'), '--cache', 'cold', '--formal', str(args.formal),
                        '--out', str(folder)], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
(args.out / 'cold_status.json').write_text(json.dumps({'phase': 'comparison', 'updated': time.time()}))
with (args.out / 'comparison_cold.log').open('a') as log:
    subprocess.run([sys.executable, str(src / 'run_comparison.py'), '--out', str(args.out),
                    '--old-library', str(args.old_library), '--new-library', str(args.new_library),
                    '--mix-library', str(args.mix_library)], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
(args.out / 'cold_status.json').write_text(json.dumps({'phase': 'complete', 'updated': time.time()}))
