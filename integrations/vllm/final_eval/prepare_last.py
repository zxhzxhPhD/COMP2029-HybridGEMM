"""Create startup diagnostics and the independent-family engineering control."""
import argparse
from copy import deepcopy
from pathlib import Path
from final_common import *
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
methods=read(stage/'methods.json')
for model in ['3B','7B']:
    # Complete selector x quantizer controls without changing the primary winner.
    parent=methods[model]['Hy-I'];cfg=read(Path(parent['model_path'])/'config.json')
    clipped=read(Path(methods[model]['Hy-F-Q']['model_path'])/'config.json')['quantization_config']['q4_bank']
    cfg['quantization_config']['q4_bank']=clipped
    plan=read(cfg['quantization_config']['decode_plans']);plan['binding']['bank_manifest_sha256']=sha(clipped)
    plan['binding']['stage3_method']='Hy-I-Q'
    path=stage/model/'plans/Hy-I-Q.json';write(path,plan);cfg['quantization_config']['decode_plans']=str(path)
    dest=stage/model/'models/Hy-I-Q';view(parent['model_path'],dest,cfg)
    methods[model]['Hy-I-Q']={**deepcopy(parent),'model_path':str(dest),'config_sha256':sha(dest/'config.json'),
        'evaluation_scope':'selector x quantizer ablation, same I mask and same accepted I compute plan; primary method is unchanged'}
for method in ['FP16','Mix-I','Hy-F-Q']:
    name='Startup-'+method;spec=deepcopy(methods['7B'][method]);spec['environment']['HYGEMM_RECORD_LOAD_TIMES']='1'
    spec['evaluation_scope']='startup instrumentation only; not a decode baseline';methods['7B'][name]=spec
model='Mistral-7B';name='K0-F-Q';parent=methods[model]['Hy-F-Q'];cfg=read(Path(parent['model_path'])/'config.json')
plan=read(stage/model/'reference_plans.json');plan={'hygemm':plan['hygemm'],'binding':{
    'library_sha256':sha(OLD),'layout':'MX K32 E8M0 native physical scales v1',
    'selection_canonical_sha256':canonical_sha(cfg['quantization_config']['selection']),
    'bank_manifest_sha256':sha(cfg['quantization_config']['q4_bank'])}}
path=stage/model/'plans'/(name+'.json');write(path,plan);cfg['quantization_config']['decode_plans']=str(path)
dest=stage/model/'models'/name;view(parent['model_path'],dest,cfg)
methods[model][name]={**deepcopy(parent),'model_path':str(dest),'environment':{**environment(True),'HYGEMM_FUSE_LLAMA':'0'},
    'config_sha256':sha(dest/'config.json'),'implementation':'K0','evaluation_scope':'engineering control'}
write(stage/'methods.json',methods)
