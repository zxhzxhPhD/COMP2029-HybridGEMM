"""Resume the validated small-batch experiment without concurrent GPU jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--formal', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--old-library', type=Path, required=True)
ap.add_argument('--new-library', type=Path, required=True)
ap.add_argument('--mix-library', type=Path, required=True)
args = ap.parse_args()
src = Path(__file__).resolve().parent
env = dict(os.environ, HYGEMM_LIBRARY=str(args.new_library), MIXLLM_LIBRARY=str(args.mix_library))
status = args.out / 'status.json'


def run(label, command):
    status.write_text(json.dumps({'phase': label, 'command': command, 'updated': time.time()}, indent=2))
    with (args.out / f'{label}.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        status.write_text(json.dumps({'phase': 'failed', 'job': label, 'code': result.returncode}, indent=2))
        raise RuntimeError(label)
    print('Complete', label, flush=True)


for suffix, ids in [('', list(range(8, 20))), ('_deeper', list(range(20, 28)))]:
    target = args.out / f'correctness{suffix}.json'
    if not target.exists() or set(int(k.split('_')[0]) for k in json.loads(target.read_text())['inventory']) != set(ids):
        run('correctness' + suffix, [sys.executable, str(src / 'check_light.py'), '--ids', ','.join(map(str, ids)), '--out', str(target)])
    for tool in ['memcheck', 'synccheck', 'racecheck']:
        target = args.out / f'{tool}{suffix}.json'
        if target.exists() and set(int(k.split('_')[0]) for k in json.loads(target.read_text())['inventory']) == set(ids):
            continue
        command = ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '3']
        if tool == 'racecheck':
            command += ['--kernel-name', 'kns=light', '--racecheck-num-workers', '4']
        command += [sys.executable, str(src / 'check_light.py'), '--sanitizer', '--ids', ','.join(map(str, ids)), '--out', str(target)]
        run(tool + suffix, command)

run('prepare', [sys.executable, str(src / 'prepare.py'), '--formal', str(args.formal), '--out', str(args.out)])
optimized = args.out / 'optimized'
complete = optimized / 'tuning_complete.json'
if complete.exists() and len(json.loads(complete.read_text())['candidates']) == 20:
    archive = optimized / 'tuning_20_candidates'
    archive.mkdir(exist_ok=True)
    for name in ['plans.json', 'shortlists.json', 'decisions.json', 'tuning_raw.jsonl', 'tuning_complete.json']:
        source, destination = optimized / name, archive / name
        assert not destination.exists(), destination
        source.rename(destination)
    shutil.copy2(args.formal / 'plans.json', optimized / 'plans.json')
if not complete.exists():
    run('tune_deeper', [sys.executable, str(src / 'tune_small.py'), '--formal', str(args.formal), '--out', str(optimized)])
if not (args.out / 'pack_correctness.json').exists():
    run('pack_correctness', [sys.executable, str(src / 'check_pack.py'), '--old-library', str(args.old_library),
                             '--out', str(args.out / 'pack_correctness.json')])
if not (args.out / 'model_search/selection.json').exists():
    if not (args.out / 'optimized/resident_attempt/plans.json').exists() and not (args.out / 'resident_proposal/tuning_complete.json').exists():
        run('tune_resident_proposal', [sys.executable, str(src / 'tune_small.py'), '--cache', 'resident',
                                      '--formal', str(args.formal), '--out', str(args.out / 'resident_proposal')])
    run('model_search', [sys.executable, str(src / 'search_model.py'), '--formal', str(args.formal),
                         '--out', str(args.out), '--new-library', str(args.new_library)])
run('comparison', [sys.executable, str(src / 'run_comparison.py'), '--out', str(args.out),
                   '--old-library', str(args.old_library), '--new-library', str(args.new_library), '--mix-library', str(args.mix_library)])
status.write_text(json.dumps({'phase': 'complete', 'updated': time.time()}, indent=2))
