"""Predeclared mask/quantizer cross comparison; no mask changes within a cell."""
import argparse
from pathlib import Path
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
protocol=read(args.root.parent/'protocol.json');q1=Path(protocol['q1'])/args.root.name
mask_sources={'q1_integer':q1/'original_selection.json','q2_integer':args.root/'views/q2_integer/selection.json'}
for family in ['loss_abs_gain','loss_signed_gain']:
    mask_sources['q1_'+family]=q1/'views'/family/'selection.json'
    mask_sources['q2_'+family]=args.root/'views'/('q2_'+family)/'selection.json'
primary=['q2_'+x for x in protocol['q2_primary']]
winner=min(primary,key=lambda v:read(args.root/'native'/f'{v}.json')['validation']['mean_nll'])
if winner not in mask_sources:
    mask_sources[winner]=args.root/'views'/winner/'selection.json'
    counterpart=winner.replace('q2_','',1)
    path=q1/'views'/counterpart/'selection.json'
    if path.exists():mask_sources['q1_'+counterpart]=path
records={}
for mode in ['native_rtn','mse_clip','act_mse_clip']:
    bank=args.root/'banks'/mode/'manifest.json';assert read(bank)['completed']
    for mask,path in mask_sources.items():
        if mode=='native_rtn' and mask!='q1_integer':continue
        name=f'q3_{mode}__{mask}'
        records[name]=make_view(args.root,name,read(path),q1,'Q3_fixed_masks',mask,bank=bank,
            extra={'mask_source':str(path),'mask_source_sha256':sha(path),'quantizer':mode,'q2_winner':winner})
write(args.root/'q3_views.json',records)
write(args.root/'q3_design.json',{'fixed_mask_keys':['q1_integer','q2_integer','q1_loss_abs_gain','q2_loss_abs_gain','q1_loss_signed_gain','q2_loss_signed_gain'],
    'extra_masks':[x for x in mask_sources if x not in ['q1_integer','q2_integer','q1_loss_abs_gain','q2_loss_abs_gain','q1_loss_signed_gain','q2_loss_signed_gain']],
    'q2_winner':winner,'views':list(records)})
print('Prepared Q3',args.root.name,len(records),'views')
