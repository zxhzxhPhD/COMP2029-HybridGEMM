"""Shared Q2/Q3 experiment IO and exact cost accounting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quality'))
from common import read,write,sha,tensor_sha,configure,dq,decode_mx,aggregate,fused_cases,analytic_budget
import numpy as np
import torch

FAMILIES=['weight_gain','output_gain','loss_abs_gain','loss_signed_gain']
MODEL_NAMES=['0.5B','1.5B','3B','7B']


def allocate(scores,shapes,target_high_weights,density=True):
    names=list(shapes)
    values=np.concatenate([np.asarray(scores[n],dtype=np.float64) for n in names])
    costs=np.concatenate([np.full(shapes[n][0],shapes[n][1],dtype=np.int64) for n in names])
    assert np.isfinite(values).all() and (costs>0).all()
    priority=values/costs if density else values
    order=np.argsort(-priority,kind='stable')
    cumulative=np.cumsum(costs[order],dtype=np.int64)
    count=int(np.searchsorted(cumulative,target_high_weights,side='right'))
    selected=np.zeros(len(costs),dtype=bool);selected[order[:count]]=True
    spent=int(cumulative[count-1]) if count else 0
    remaining=target_high_weights-spent
    minimum_cost=int(costs.min())
    for index in order[count:]:
        cost=int(costs[index])
        if cost<=remaining:
            selected[index]=True;remaining-=cost
        if remaining<minimum_cost:break
    assert int(costs[selected].sum())<=target_high_weights
    result={};offset=0
    for n in names:
        size=shapes[n][0];result[n]=np.flatnonzero(selected[offset:offset+size]).tolist();offset+=size
    return result,{'target_high_weight_elements':target_high_weights,'actual_high_weight_elements':int(costs[selected].sum()),
        'unspent_high_weight_elements':remaining,'selected_channels':int(selected.sum()),
        'negative_score_selected_channels':int((values[selected]<0).sum()),
        'negative_score_selected_weights':int(costs[selected & (values<0)].sum()),
        'method':'stable benefit/cost greedy with tail fill' if density else 'global random priority with cost-feasible tail fill',
        'globally_optimal_claim':False}


def make_view(folder,name,selection,q1,stage,family,seed=2029,bank=None,extra=None):
    meta=read(q1/'original_calibration_manifest.json')
    cfg=read(q1/'views/original/config.json')
    cases=fused_cases(selection,meta['shapes'],cfg['num_hidden_layers'])
    budget=analytic_budget(cases)
    avg=budget['payload_bits']/meta['weights'];target=read(q1/'budget_expected.json')['payload_bits']/meta['weights']
    assert avg<5.1 and avg<=target+1e-12 and target-avg<.001,(name,avg,target)
    view=folder/'views'/name;view.mkdir(parents=True,exist_ok=True)
    for file in (q1/'views/original').iterdir():
        if file.is_file() and file.name not in ['config.json','selection.json'] and not (view/file.name).exists():
            (view/file.name).symlink_to(file.resolve())
    cfg['quantization_config']={'quant_method':'hygemm_offline','checkpoint_format':'fp16_rtn','selection':selection,
        'reference_plans':str(folder/'reference_plans.json'),'reference_cases':str(folder/'reference_cases.json')}
    if bank:cfg['quantization_config']['q4_bank']=str(bank)
    write(view/'config.json',cfg);write(view/'selection.json',selection)
    old=read(q1/'budget_expected.json')
    record={'name':name,'stage':stage,'family':family,'seed':seed,'bank':str(bank) if bank else None,
        'budget':budget,'payload_bits_per_weight':avg,'target_payload_bits_per_weight':target,
        'physical_byte_delta_vs_q1':budget['persistent_quantized_bytes']-old['persistent_quantized_bytes'],
        'weights':meta['weights'],'selection_sha256':sha(view/'selection.json'),
        'pure4_fused_layers':sum(r['high']==0 for r in cases),'pure8_fused_layers':sum(r['high']==r['out'] for r in cases),
        'mixed_fused_layers':sum(0<r['high']<r['out'] for r in cases),'layers':cases,'extra':extra or {}}
    write(view/'view.json',record)
    return record
