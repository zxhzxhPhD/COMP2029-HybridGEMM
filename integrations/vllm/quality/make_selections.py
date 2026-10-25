"""Build fixed-unfused-count model views, preserving every original plan."""
import argparse
import copy
from pathlib import Path
import torch
from common import read,write,sha,fused_cases,analytic_budget

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--seed',type=int,default=2029)
ap.add_argument('--methods',default='weight_gain,output_gain,loss_abs_gain,loss_signed_gain')
args=ap.parse_args()
old=read(args.root/'original_selection.json')
meta=read(args.root/'original_calibration_manifest.json')
cfg=read(args.root/'views/original/config.json')
scores=torch.load(args.root/'scores'/str(args.seed)/'scores.pt',map_location='cpu',weights_only=True)['scores']
selections={}
if args.seed==2029:
    for seed in [2029,2030,2031]:
        gen=torch.Generator().manual_seed(seed)
        selections[f'random_{seed}']={n:sorted(torch.randperm(meta['shapes'][n][0],generator=gen)[:len(ids)].tolist()) for n,ids in old.items()}
for method in args.methods.split(','):
    key=method if args.seed==2029 else f'{method}_seed{args.seed}'
    selections[key]={n:sorted(scores[method][n].argsort(descending=True,stable=True)[:len(ids)].tolist()) for n,ids in old.items()}
for key,selection in selections.items():
    assert set(selection)==set(old)
    assert all(len(selection[n])==len(ids) for n,ids in old.items())
    cases=fused_cases(selection,meta['shapes'],cfg['num_hidden_layers'])
    budget=analytic_budget(cases)
    assert budget==read(args.root/'budget_expected.json')
    folder=args.root/'views'/key;folder.mkdir(parents=True,exist_ok=True)
    for file in (args.root/'views/original').iterdir():
        if file.name not in ['config.json','selection.json'] and file.is_file() and not (folder/file.name).exists():
            (folder/file.name).symlink_to(file.resolve())
    conf=copy.deepcopy(cfg);conf['quantization_config']['selection']=selection
    write(folder/'config.json',conf);write(folder/'selection.json',selection)
    overlap=sum(len(set(ids)&set(selection[n])) for n,ids in old.items())
    write(folder/'selection_audit.json',{'method':key,'seed':args.seed,'fixed_every_original_projection_count':True,
        'budget':budget,'original_overlap':overlap,'high_channels':sum(map(len,old.values())),
        'selection_sha256':sha(folder/'selection.json'),'plans_sha256':sha(args.root/'plans.json'),
        'layers':cases})
    print('Selection',args.root.name,key,'overlap',overlap,'budget',budget['persistent_quantized_bytes'],flush=True)
