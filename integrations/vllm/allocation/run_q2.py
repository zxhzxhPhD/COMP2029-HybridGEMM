"""One GPU job at a time; all primary methods, then loss-proxy seed checks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--aliases-only',action='store_true');args=ap.parse_args()
src=Path(__file__).resolve().parent
logs=args.out/'logs';logs.mkdir(exist_ok=True)
protocol=read(args.out/'protocol.json')


def run(model,view):
    target=args.out/model/'native'/f'{view}.json'
    spec=args.out/model/'views'/view
    identity={'script_sha256':sha(src/'evaluate.py'),'offline_sha256':sha(src.parent/'hygemm_vllm/offline.py'),
              'weights_loader_sha256':sha(src.parent/'hygemm_vllm/offline_weights.py'),
              'config_sha256':sha(spec/'config.json'),'selection_sha256':sha(spec/'selection.json'),
              'kernel_sha256':sha(os.environ['HYGEMM_LIBRARY'])}
    stamp=logs/f'{model}_{view}.identity.json'
    if target.exists() and read(target).get('completed'):
        assert read(stamp)==identity,('Stale result',target)
        print('Reuse',model,view,flush=True);return
    write(stamp,identity)
    command=[sys.executable,str(src/'evaluate.py'),'--root',str(args.out/model),'--view',view]
    write(args.out/'status.json',{'phase':f'{model}:{view}','command':command})
    with (logs/f'{model}_{view}.log').open('a') as f:
        status=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
    if status:
        write(args.out/'status.json',{'phase':'failed','model':model,'view':view,'returncode':status});raise RuntimeError((model,view))
    print('Finished',model,view,flush=True)


for model in MODEL_NAMES:run(model,'q1_alias')
write(args.out/'aliases_complete.json',{'passed':True})
if args.aliases_only:sys.exit(0)
for model in ['3B','7B','0.5B','1.5B']:
    for method in protocol['q2_primary']:run(model,'q2_'+method)
for model in MODEL_NAMES:
    for seed in [2030,2031]:
        for family in protocol['q2_robustness_families']:run(model,f'q2_{family}_seed{seed}')
write(args.out/'q2_complete.json',{'passed':True,'native_runs':56})
write(args.out/'status.json',{'phase':'q2_complete'})
