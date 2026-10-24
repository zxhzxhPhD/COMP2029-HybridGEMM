"""End-to-end acceptance of shape-tuned policies with the same Graph config."""
import argparse
import copy
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
ap.add_argument('--new-library', type=Path, required=True)
args = ap.parse_args()
source = Path(__file__).resolve().parents[1] / 'profiling/profile_decode.py'
root = args.out / 'model_search'
root.mkdir(exist_ok=True)
original = json.loads((args.formal / 'plans.json').read_text())
cold = args.out / 'optimized/cold_attempt/plans.json'
if not cold.exists():
    cold = args.out / 'optimized/plans.json'
policies = {'baseline_compute': original}
resident = args.out / 'optimized/resident_attempt/plans.json'
if not resident.exists():
    resident = args.out / 'resident_proposal/plans.json'
if resident.exists():
    policies['resident_compute'] = json.loads(resident.read_text())
policies['cold_compute'] = json.loads(cold.read_text())
uniform = copy.deepcopy(original)
for row in uniform['hygemm']:
    if row['tokens'] <= 32:
        row.update(low_id=21, high_id=21, policy='concurrent' if 0 < row['high'] < row['out'] else 'serial')
policies['uniform_deep32'] = uniform
env = dict(os.environ, HYGEMM_LIBRARY=str(args.new_library))
for name, plans in policies.items():
    folder = root / name
    folder.mkdir(exist_ok=True)
    (folder / 'plans.json').write_text(json.dumps(plans, indent=2))
    if not (folder / 'calibration').exists():
        (folder / 'calibration').symlink_to(args.formal / 'calibration', target_is_directory=True)
    model = folder / 'models/hygemm'
    model.mkdir(parents=True, exist_ok=True)
    for f in (args.formal / 'models/hygemm').iterdir():
        if f.name != 'config.json' and f.is_file() and not (model / f.name).exists():
            (model / f.name).symlink_to(f.resolve())
    cfg = json.loads((args.formal / 'models/hygemm/config.json').read_text())
    cfg['quantization_config']['tuning_plan'] = str(folder / 'plans.json')
    (model / 'config.json').write_text(json.dumps(cfg))
    if not (folder / 'control_hygemm.json').exists():
        (root / 'status.json').write_text(json.dumps({'phase': name, 'updated': time.time()}))
        with (root / f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(source), '--formal', str(folder), '--out', str(folder),
                            '--kind', 'hygemm', '--batches', '1,2,4,8,16,32'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Measured', name, flush=True)
results = {name: json.loads((root / name / 'control_hygemm.json').read_text()) for name in policies}
winners = {}
for batch in [1, 2, 4, 8, 16, 32]:
    points = {name: next(p for p in d['cells'] if p['batch'] == batch) for name, d in results.items()}
    assert all(p['stable'] for p in points.values())
    best = min(points, key=lambda n: points[n]['median_ms_per_step'])
    gain = points['baseline_compute']['median_ms_per_step'] / points[best]['median_ms_per_step']
    winners[batch] = best if gain >= 1.01 else 'baseline_compute'
    print('Winner', batch, winners[batch], 'gain', gain, flush=True)
final = copy.deepcopy(original)
lookup = {name: {(p['out'], p['k'], p['high'], p['tokens']): p for p in plans['hygemm']} for name, plans in policies.items()}
for i, row in enumerate(final['hygemm']):
    if row['tokens'] <= 32:
        final['hygemm'][i] = copy.deepcopy(lookup[winners[row['tokens']]][(row['out'], row['k'], row['high'], row['tokens'])])
archive = args.out / 'optimized/cold_attempt'
if not archive.exists():
    archive.mkdir()
    for name in ['plans.json', 'shortlists.json', 'decisions.json', 'tuning_raw.jsonl', 'tuning_complete.json', 'result_hygemm.json']:
        source_file = args.out / 'optimized' / name
        if source_file.exists():
            source_file.rename(archive / name)
    if (args.out / 'benchmark_optimized.log').exists():
        (args.out / 'benchmark_optimized.log').rename(args.out / 'benchmark_optimized_cold.log')
    quality = args.out / 'decode_quality_optimized.json'
    if quality.exists():
        quality.rename(args.out / 'decode_quality_optimized_cold.json')
(args.out / 'optimized/plans.json').write_text(json.dumps(final, indent=2))
(args.out / 'optimized/tuning_complete.json').write_text(json.dumps({'passed': True, 'candidates': list(range(28)),
    'tuned_buckets': [1, 2, 4, 8, 16, 32], 'method': 'shape-level proposals with whole-model policy gate',
    'winners': winners, 'adoption_margin': 1.01, 'large_and_mixllm_plans_unchanged': True}, indent=2))
(root / 'selection.json').write_text(json.dumps({'winners': winners, 'adoption_margin': 1.01,
    'protocol': 'Four whole-model policies; fixed prompt128; original capture sizes; 1 warmup + 5 runs, CV<=5%; large/prefill plans unchanged'}, indent=2))
(root / 'status.json').write_text(json.dumps({'phase': 'complete', 'updated': time.time()}))
