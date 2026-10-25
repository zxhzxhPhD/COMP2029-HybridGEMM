"""Sequential control measurements and six bounded Nsight Systems traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--formal', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--probes', action='store_true', help='Also run the small non-serving Graph diagnostics')
args = ap.parse_args()
script = Path(__file__).with_name('profile_decode.py')
status = args.out / 'status.json'
jobs = [(kind, None) for kind in ['mixllm', 'hygemm']]
jobs += [(kind, batch) for batch in [1, 8, 64] for kind in ['mixllm', 'hygemm']]
for kind, batch in jobs:
    label = f'{kind}_b{batch}' if batch else f'control_{kind}'
    output = args.out / (f'profile_{label}.json' if batch else f'{label}.json')
    trace = args.out / f'{label}.nsys-rep'
    if not output.exists() or (batch and not trace.exists()):
        command = [sys.executable, str(script), '--formal', str(args.formal), '--out', str(args.out), '--kind', kind]
        if batch:
            command += ['--batches', str(batch), '--profile']
            command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                       '--cuda-graph-trace=node:nvtx-precapture', '--capture-range=cudaProfilerApi',
                       '--capture-range-end=stop', '--force-overwrite=true', '-o', str(args.out / label), *command]
        status.write_text(json.dumps({'phase': label, 'command': command, 'updated': time.time()}, indent=2))
        with (args.out / f'{label}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed', label, flush=True)
    if batch and not (args.out / f'{label}.sqlite').exists():
        with (args.out / f'{label}_export.log').open('w') as log:
            subprocess.run(['nsys', 'export', '--type=sqlite', '--force-overwrite=true',
                            '-o', str(args.out / f'{label}.sqlite'), str(trace)], stdout=log, stderr=subprocess.STDOUT, check=True)
for kind in ['mixllm', 'hygemm']:
    if not (args.out / f'meter_{kind}.json').exists():
        command = [sys.executable, str(script), '--formal', str(args.formal), '--out', str(args.out), '--kind', kind, '--meter']
        status.write_text(json.dumps({'phase': 'meter_' + kind, 'command': command, 'updated': time.time()}, indent=2))
        with (args.out / f'meter_{kind}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
if args.probes:
    for name, extra in [('graph_probe', []), ('topology_probe', ['--topology'])]:
        probe = script.with_name('graph_node_probe.py')
        if not (args.out / f'{name}_control.json').exists():
            with (args.out / f'{name}_control.log').open('w') as log:
                subprocess.run([sys.executable, str(probe), '--out', str(args.out / f'{name}_control.json'), *extra], stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (args.out / f'{name}.nsys-rep').exists():
            command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                       '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                       '--force-overwrite=true', '-o', str(args.out / name), sys.executable, str(probe),
                       '--out', str(args.out / f'{name}_trace.json'), '--trace', *extra]
            with (args.out / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (args.out / f'{name}.sqlite').exists():
            with (args.out / f'{name}_export.log').open('w') as log:
                subprocess.run(['nsys', 'export', '--type=sqlite', '--force-overwrite=true', '-o',
                                str(args.out / f'{name}.sqlite'), str(args.out / f'{name}.nsys-rep')], stdout=log, stderr=subprocess.STDOUT, check=True)
status.write_text(json.dumps({'phase': 'complete', 'updated': time.time()}, indent=2))
