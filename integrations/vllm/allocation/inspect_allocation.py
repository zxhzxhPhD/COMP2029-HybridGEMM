"""Explain quota movement and proxy objective changes without causal overclaims."""
import argparse
from pathlib import Path
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
protocol=read(args.out/'protocol.json');q1=Path(protocol['q1']);records={}
torch.set_num_threads(4)
for model in MODEL_NAMES:
    root=args.out/model;meta=read(root/'original_calibration_manifest.json');records[model]={}
    values=torch.load(q1/model/'scores/2029/scores.pt',map_location='cpu',weights_only=True)['scores']
    for family in FAMILIES:
        old=read(q1/model/'views'/family/'selection.json')
        new=read(root/'views'/('q2_'+family)/'selection.json');groups={};layers=[]
        intersection=union=oldsum=newsum=negative=0
        for name,(o,k) in meta['shapes'].items():
            a,b=set(old[name]),set(new[name]);intersection+=len(a&b)*k;union+=len(a|b)*k
            score=values[family][name].double()
            before=float(score[old[name]].sum());after=float(score[new[name]].sum())
            oldsum+=before;newsum+=after;negative+=int((score[new[name]]<0).sum())
            group=name.split('.',3)[-1]
            row=groups.setdefault(group,{'weights':0,'old_high_weights':0,'new_high_weights':0,'old_high_channels':0,'new_high_channels':0})
            row['weights']+=o*k;row['old_high_weights']+=len(a)*k;row['new_high_weights']+=len(b)*k
            row['old_high_channels']+=len(a);row['new_high_channels']+=len(b)
            layers.append({'name':name,'out':o,'k':k,'old_high':len(a),'new_high':len(b),
                'old_selected_proxy':before,'new_selected_proxy':after,
                'intersect_channels':len(a&b),'negative_selected':int((score[new[name]]<0).sum())})
        all_scores=np.concatenate([values[family][n].numpy().astype(np.float64) for n in meta['shapes']])
        costs=np.concatenate([np.full(o,k,dtype=np.int64) for o,k in meta['shapes'].values()])
        order=np.argsort(-all_scores/costs,kind='stable');spent=np.cumsum(costs[order]);target=meta['high_weights']
        count=int(np.searchsorted(spent,target,side='right'));fractional=float(all_scores[order[:count]].sum())
        remaining=target-(int(spent[count-1]) if count else 0)
        if count<len(order):fractional+=remaining*all_scores[order[count]]/costs[order[count]]
        records[model][family]={'weighted_mask_jaccard':intersection/union,
            'retained_high_weight_fraction':intersection/meta['high_weights'],
            'old_total_selected_proxy':oldsum,'new_total_selected_proxy':newsum,
            'proxy_improvement':newsum-oldsum,'selected_negative_channels':negative,
            'fractional_matched_budget_proxy_upper_bound':fractional,
            'relative_proxy_gap_to_fractional_upper_bound':max(0.,fractional-newsum)/max(abs(fractional),1e-30),
            'groups':groups,'layers':layers,
            'interpretation':'A better additive proxy objective is not evidence of lower whole-model loss; correlations and multi-layer interactions are unmodeled.'}
write(args.out/'allocation_diagnostics.json',records)
for model in ['3B','7B']:
    for family in ['loss_abs_gain','loss_signed_gain']:
        row=records[model][family]
        print(model,family,'retained_weight_fraction',row['retained_high_weight_fraction'],
              'proxy_old/new',row['old_total_selected_proxy'],row['new_total_selected_proxy'],
              'negative',row['selected_negative_channels'],flush=True)
