"""Q2 allocations reuse Q1 scores and retain each model's original payload."""
import argparse
from pathlib import Path
import shutil
import torch
import numpy as np
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--q1',type=Path,required=True);ap.add_argument('--formal',type=Path,required=True);args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
write(args.out/'protocol.json',{'stage':'Q2/Q3','models':MODEL_NAMES,'q1':str(args.q1),'formal':str(args.formal),
    'payload_policy':'Same per-model weight-code budget as Q1, never above it; residual <0.001 bits/weight; strictly below5.1. Scales/IDs/padding reported separately.',
    'allocation':'global score divided by K; no per-layer high-count constraint; force near-full matching budget including negative scores if necessary',
    'q2_primary':['integer','random_2029','random_2030','random_2031',*FAMILIES,'uniform_abs'],
    'q2_robustness_families':['loss_abs_gain','loss_signed_gain'],'seeds':[2029,2030,2031],
    'q3_quantizers':['mse_clip','act_mse_clip'],'q3_exponent_offsets':[0,-1,-2],
    'q3_mask_matrix':['Q1/Q2 integer','Q1/Q2 loss_abs_gain','Q1/Q2 loss_signed_gain'],
    'q3_joint_rule':'Select quantizer by mean validation NLL across the six fixed masks; recompute both loss-proxy families with that quantizer and reallocate at unchanged budget.',
    'q3_extra_family_rule':'If another Q2 view wins a model, also include its Q1/Q2 counterpart masks in Q3; if a floating-score family, include it in joint reselection.',
    'quality_plans':'Per fused layer inherit Q1 kernel IDs/stages/token buckets, despite changed H; verify actual dispatch. No new device kernels.',
    'graph':'FULL','inductor':False,'activation_quantizer':'unchanged native MXFP8','high_weight_quantizer':'unchanged native MXFP8',
    'kernel_sha256':'efea6b22b907a536b95bf343c817fada2fedd9977e6245251c4693e82159ec90',
    'mixllm_sha256':'1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da',
    'test_scoring':False,'validation':'same full WT2 validation 128x2048,262016 targets',
    'scope_note':'Latest user payload requirement supersedes previous exact physical-byte cap; all physical overhead changes disclosed.'})
for model in MODEL_NAMES:
    q1=args.q1/model;folder=args.out/model;folder.mkdir(exist_ok=True)
    meta=read(q1/'original_calibration_manifest.json');old=read(q1/'original_selection.json')
    shutil.copy2(q1/'plans.json',folder/'reference_plans.json')
    cfg=read(q1/'views/original/config.json')
    write(folder/'reference_cases.json',{'layers':fused_cases(old,meta['shapes'],cfg['num_hidden_layers'])})
    for name in ['tokens.pt','original_calibration_manifest.json','original_selection.json']:
        if not (folder/name).exists():(folder/name).symlink_to((q1/name).resolve())
    make_view(folder,'q1_alias',old,q1,'control','integer')
    integer=torch.load(args.formal/model/'calibration/scores.pt',map_location='cpu',weights_only=True)['scores']
    # Verify the original integer score provenance before using it in a new allocation.
    flat=torch.cat([integer[n] for n in meta['shapes']]);chosen=flat.argsort(descending=True,stable=True)[:meta['high_channels']]
    original_flat=torch.zeros(len(flat),dtype=torch.bool);original_flat[chosen]=True;offset=0
    for n,(o,k) in meta['shapes'].items():
        assert torch.where(original_flat[offset:offset+o])[0].tolist()==old[n];offset+=o
    masks={}
    for seed in [2029,2030,2031]:
        scores=torch.load(q1/'scores'/str(seed)/'scores.pt',map_location='cpu',weights_only=True)['scores']
        families=FAMILIES if seed==2029 else ['loss_abs_gain','loss_signed_gain']
        for family in families:
            name=f'q2_{family}' if seed==2029 else f'q2_{family}_seed{seed}'
            selection,info=allocate(scores[family],meta['shapes'],meta['high_weights'])
            masks[name]=make_view(folder,name,selection,q1,'Q2',family,seed,extra=info)
    selection,info=allocate(integer,meta['shapes'],meta['high_weights'])
    masks['q2_integer']=make_view(folder,'q2_integer',selection,q1,'Q2','integer',extra=info)
    for seed in [2029,2030,2031]:
        rng=np.random.default_rng(seed);scores={n:rng.random(o) for n,(o,k) in meta['shapes'].items()}
        selection,info=allocate(scores,meta['shapes'],meta['high_weights'],density=False)
        name=f'q2_random_{seed}';masks[name]=make_view(folder,name,selection,q1,'Q2','random',seed,extra=info)
    # Equal per-layer high fraction, with absolute-loss ranking within each layer.
    scores=torch.load(q1/'scores/2029/scores.pt',map_location='cpu',weights_only=True)['scores']['loss_abs_gain']
    fraction=meta['high_weights']/meta['weights'];counts={n:int(o*fraction) for n,(o,k) in meta['shapes'].items()}
    remain=meta['high_weights']-sum(counts[n]*k for n,(o,k) in meta['shapes'].items())
    order=sorted(meta['shapes'],key=lambda n:meta['shapes'][n][0]*fraction-counts[n],reverse=True)
    while any(counts[n]<meta['shapes'][n][0] and meta['shapes'][n][1]<=remain for n in order):
        for n in order:
            o,k=meta['shapes'][n]
            if counts[n]<o and k<=remain:counts[n]+=1;remain-=k
    selection={n:sorted(scores[n].argsort(descending=True,stable=True)[:counts[n]].tolist()) for n in counts}
    masks['q2_uniform_abs']=make_view(folder,'q2_uniform_abs',selection,q1,'Q2','uniform_abs')
    write(folder/'q2_views.json',masks)
    print('Prepared',model,len(masks),'Q2 views',flush=True)
