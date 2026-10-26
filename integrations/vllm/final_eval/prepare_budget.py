"""Preregister same-quantizer budget curves and physical storage before scoring."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
from final_common import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'allocation'))
from alloc_utils import allocate,analytic_budget,fused_cases


def quotas(shapes,counts,target,scores):
    selected={n:list(scores[n].argsort(descending=True,stable=True)[:counts[n]].tolist()) for n in shapes}
    spent=sum(len(selected[n])*k for n,(o,k) in shapes.items())
    candidates=sorted(((float(scores[n][i])/k,n,int(i),k) for n,(o,k) in shapes.items()
        for i in scores[n].argsort(descending=True,stable=True)[counts[n]:].tolist()),reverse=True)
    for _,n,i,k in candidates:
        if spent+k<=target:selected[n].append(i);spent+=k
        if target-spent<min(v[1] for v in shapes.values()):break
    assert spent<=target
    return {n:sorted(ids) for n,ids in selected.items()}


def create_method(stage,model,name,selection,record):
    methods=read(stage/'methods.json');main=methods[model]['Hy-F-Q'];folder=stage/model
    cfg=read(Path(main['model_path'])/'config.json');meta=read(Q1/model/'original_calibration_manifest.json')
    layers=fused_cases(selection,meta['shapes'],cfg['num_hidden_layers']);budget=analytic_budget(layers)
    parent=read(cfg['quantization_config']['decode_plans']);layer_map={r['layer']:r for r in layers}
    plans=deepcopy(parent);unique={}
    for row in plans['layer_plans']:
        row['high']=layer_map[row['layer']]['high']
        unique.setdefault((row['out'],row['k'],row['high'],row['tokens']),{k:v for k,v in row.items() if k!='layer'})
    plans['hygemm']=list(unique.values());plans['binding']={'library_sha256':sha(NEW),
        'layout':'MX K32 E8M0 native physical scales v1','selection_canonical_sha256':canonical_sha(selection),
        'bank_manifest_sha256':sha(cfg['quantization_config']['q4_bank']),'stage3_method':name}
    planfile=folder/'plans'/(name+'.json');write(planfile,plans)
    cfg['quantization_config'].update(selection=selection,decode_plans=str(planfile))
    dest=folder/'models'/name;view(main['model_path'],dest,cfg)
    write(folder/'selections'/(name+'.json'),selection)
    shapefile=folder/'shapes'/(name+'.json');write(shapefile,{'layers':layers,'budget':budget,'weights':meta['weights']})
    methods[model][name]={**deepcopy(main),'model_path':str(dest),'budget':budget,'config_sha256':sha(dest/'config.json'),
        'shape_manifest':str(shapefile),'status':'prepared; bounded actual-shape tuning and freeze required'}
    write(stage/'methods.json',methods)
    return {**record,'method':name,'selection_sha256':sha(folder/'selections'/(name+'.json')),'budget':budget,
        'payload_bits_per_weight':budget['payload_bits']/meta['weights'],'shape_manifest':str(shapefile)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
    curves={}
    for model in ['3B','7B']:
        meta=read(Q1/model/'original_calibration_manifest.json');shapes=meta['shapes']
        original=read(Q1/model/'original_selection.json');cap=meta['high_weights'];total=meta['weights']
        integer=torch.load(FORMAL/model/'calibration/scores.pt',map_location='cpu',weights_only=True)['scores']
        floating=torch.load(Q1/model/'scores/2029/scores.pt',map_location='cpu',weights_only=True)['scores']['loss_signed_gain']
        rows=[]
        for alpha in [0,.25,.5,.75,1]:
            target=int(alpha*cap)
            if alpha==0:
                mask={n:[] for n in shapes}
                record=create_method(stage,model,'B-zero',mask,{'alpha':0,'rule':'shared_zero'})
                rows.append(record);continue
            for rule in ['integer_cost','frozen_float_fixed_quota','uniform_float']:
                if rule=='frozen_float_fixed_quota' and alpha==1:
                    existing=read(stage/'methods.json')[model]['Hy-F-Q'];selection=read(Path(existing['model_path'])/'config.json')['quantization_config']['selection']
                    assert selection=={n:sorted(floating[n].argsort(descending=True,stable=True)[:len(original[n])].tolist()) for n in shapes}
                    rows.append({'alpha':alpha,'rule':rule,'method':'Hy-F-Q','budget':existing['budget'],
                        'payload_bits_per_weight':existing['budget']['payload_bits']/total,'shared_primary_result':True});continue
                if rule=='integer_cost':mask,_=allocate(integer,shapes,target)
                else:
                    counts={n:int(len(original[n])*alpha) if rule=='frozen_float_fixed_quota' else int(o*target/total) for n,(o,k) in shapes.items()}
                    mask=quotas(shapes,counts,target,floating)
                name='B-'+{'integer_cost':'integer','frozen_float_fixed_quota':'float','uniform_float':'uniform'}[rule]+f'-a{round(alpha*100):03d}'
                record=create_method(stage,model,name,mask,{'alpha':alpha,'rule':rule})
                assert record['budget']['payload_bits']<=4*total+4*target
                rows.append(record)
        rng=np.random.default_rng(2029);random_scores={n:rng.random(o) for n,(o,k) in shapes.items()}
        mask,_=allocate(random_scores,shapes,cap,density=False)
        rows.append(create_method(stage,model,'B-random-a100',mask,{'alpha':1,'rule':'random_cost_feasible','seed':2029}))
        mask={n:list(range(o)) for n,(o,k) in shapes.items()}
        rows.append(create_method(stage,model,'W8A8',mask,{'alpha':'all','rule':'over_budget_endpoint'}))
        curves[model]=rows
    write(stage/'budget_design.json',{'quantizer':'same frozen act_mse_clip bank for all budget rows',
        'source_score':'Frozen stage-one Q1 signed scores; no reranking after final quality scores',
        'uniform':'equal upgraded weight fraction per layer; signed ranking within layer; cost-feasible residual fill',
        'curves':curves})


if __name__=='__main__':main()
