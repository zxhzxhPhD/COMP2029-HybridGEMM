"""Serial GPU pipeline for budget, diagnostics, fixed-count selection and checks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import read,write,sha

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--formal',type=Path,required=True)
ap.add_argument('--through',choices=['budget','diagnosis','selection','robustness','all'],default='all')
args=ap.parse_args();src=Path(__file__).resolve().parent
models=['0.5B','1.5B','3B','7B']
logdir=args.out/'logs';logdir.mkdir(exist_ok=True)
protocol=read(args.out/'protocol.json')
assert sha(os.environ['HYGEMM_LIBRARY'])==protocol['native_library_sha256']


def run(label,script,arguments,target):
    stamp=logdir/f'{label}.identity.json'
    identity={'script_sha256':sha(src/script),'common_sha256':sha(src/'common.py'),
              'arguments':list(map(str,arguments)),'protocol_sha256':sha(args.out/'protocol.json'),
              'library_sha256':sha(os.environ['HYGEMM_LIBRARY'])}
    if script=='native_eval.py':
        identity['worker_sha256']=sha(src.parent/'hygemm_vllm/quality_worker.py')
    if '--root' in arguments:
        folder=Path(arguments[arguments.index('--root')+1])
        identity['tokens_sha256']=sha(folder/'tokens.pt')
        identity['plans_sha256']=sha(folder/'plans.json')
        method=arguments[arguments.index('--method')+1] if '--method' in arguments else 'original'
        identity['model_config_sha256']=sha(folder/'views'/method/'config.json')
    # Never treat partial diagnostic files as successful completion markers.
    if target.exists() and read(target).get('completed',read(target).get('passed',False)):
        assert stamp.exists() and read(stamp)==identity,('Stale completed result; archive it before rerun',target)
        print('Reuse',label,flush=True);return
    write(stamp,identity)
    command=[sys.executable,str(src/script),*map(str,arguments)]
    write(args.out/'status.json',{'phase':label,'command':command,'started':time.time(),'script_sha256':sha(src/script)})
    with (logdir/f'{label}.log').open('a') as log:
        rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:
        write(args.out/'status.json',{'phase':'failed','job':label,'code':rc});raise RuntimeError(label)
    assert target.exists(),target
    print('Finished',label,flush=True)


for name in models:
    folder=args.out/name
    run('budget_'+name,'native_eval.py',['--root',folder,'--audit-only'],folder/'budget/original.json')
write(args.out/'budget_complete.json',{'passed':True,'models':models})
if args.through=='budget':sys.exit(0)
for name in models:
    folder=args.out/name
    run('diagnosis_'+name,'diagnose.py',['--root',folder,'--formal',args.formal/name],folder/'diagnosis.json')
    run('diagnosis_fp32_'+name,'diagnose.py',['--root',folder,'--formal',args.formal/name,'--reference-math','fp32','--output','diagnosis_fp32.json'],folder/'diagnosis_fp32.json')
write(args.out/'diagnosis_complete.json',{'passed':True,'models':models})
if args.through=='diagnosis':sys.exit(0)
write(args.out/'reference_protocol.json',{'methods':['fp16','mixllm'],'split':'same full WT2 validation',
    'purpose':'Unchanged-stack quality references, not channel-score candidates; no test scoring.',
    'mixllm_library_sha256':'1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da'})
for name in ['3B','7B','0.5B','1.5B']:
    folder=args.out/name
    for kind in ['fp16','mixllm']:
        run('reference_'+name+'_'+kind,'reference_eval.py',['--root',folder,'--kind',kind],folder/'native'/f'reference_{kind}.json')
    run('scores_'+name+'_2029','score.py',['--root',folder,'--formal',args.formal/name],folder/'scores/2029/manifest.json')
    subprocess.run([sys.executable,str(src/'make_selections.py'),'--root',str(folder)],check=True)
    for method in protocol['selection_methods']:
        run('native_'+name+'_'+method,'native_eval.py',['--root',folder,'--method',method],folder/'native'/f'{method}.json')
families=['weight_gain','output_gain','loss_abs_gain','loss_signed_gain']
rank={m:sum(read(args.out/n/'native'/f'{m}.json')['validation']['mean_nll']-
                 read(args.out/n/'native/original.json')['validation']['mean_nll'] for n in ['3B','7B'])/2 for m in families}
top=sorted(rank,key=rank.get)[:2]
write(args.out/'shortlist.json',{'mean_validation_nll_delta_3B_7B':rank,'families':top,'rule':protocol['robustness_rule']})
if args.through=='selection':sys.exit(0)
write(args.out/'robustness_extension.json',{'models':models,'seeds':[2029,2030,2031],
    'reason':'Observed model dependence in primary-seed comparisons; expand the same two shortlisted families to all four models before evaluating additional small-model seeds. No change to ranking or seed selection rules.'})
for name in models:
    folder=args.out/name
    for seed in [2030,2031]:
        run(f'scores_{name}_{seed}','score.py',['--root',folder,'--formal',args.formal/name,'--seed',seed],folder/f'scores/{seed}/manifest.json')
        subprocess.run([sys.executable,str(src/'make_selections.py'),'--root',str(folder),'--seed',str(seed),'--methods',','.join(top)],check=True)
        for method in top:
            key=f'{method}_seed{seed}'
            run('native_'+name+'_'+key,'native_eval.py',['--root',folder,'--method',key],folder/'native'/f'{key}.json')
if args.through=='robustness':sys.exit(0)
for name in ['3B','7B']:
    folder=args.out/name
    run('rank_probe_'+name,'probe_rank.py',['--root',folder,'--formal',args.formal/name],folder/'rank_probe.json')
# Decode only the baseline and both predeclared shortlisted families, preserving all capture sizes.
for method in ['original',*top]:
    folder=args.out/'7B'
    target=folder/'native'/f'{method}.json'
    if read(target).get('decode'):continue
    command=[sys.executable,str(src/'native_eval.py'),'--root',str(folder),'--method',method,'--decode']
    write(args.out/'status.json',{'phase':'decode_'+method,'command':command})
    with (logdir/f'decode_7B_{method}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
write(args.out/'status.json',{'phase':'complete','updated':time.time()})
