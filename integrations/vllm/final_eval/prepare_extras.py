"""Choose ancillary baselines and mechanism ablations before final results."""
import argparse
from copy import deepcopy
from pathlib import Path
from final_common import *

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
methods=read(stage/'methods.json')
for model in ['0.5B','7B']:
    source=Path(methods[model]['FP16']['model_path']);cfg=read(source/'config.json');name='FP8-native'
    dest=stage/model/'models'/name;view(source,dest,cfg)
    methods[model][name]={'model_path':str(dest),'environment':environment(),'budget':None,'config_sha256':sha(dest/'config.json'),
        'source_checkpoint':methods[model]['FP16']['source_checkpoint'],'llm_options':{'quantization':'fp8'},
        'audit_mixed':False,'status':'optional baseline smoke pending; actual backend must be recorded',
        'budget_scope':'8-bit endpoint outside primary payload cap; format is ordinary vLLM FP8, not MXFP8'}
model='7B';parent=methods[model]['Hy-F-Q'];cfg=read(Path(parent['model_path'])/'config.json');base=read(cfg['quantization_config']['decode_plans'])
for name,env_updates in [('A-no-pack',{'HYGEMM_PACK_MODE':'0'}),('A-no-producer',{'HYGEMM_FUSE_SILU':'0'}),
                         ('A-serial',{}),('A-no-light-high',{})]:
    spec=deepcopy(parent);config=deepcopy(cfg);plan=deepcopy(base)
    for collection in ['hygemm','layer_plans']:
        for row in plan.get(collection,[]):
            if name=='A-serial':row['policy']='serial'
            if name=='A-no-light-high' and row['high_id']>=8:row['high_id']=0
    plan['binding'].pop('runtime_options',None)
    file=stage/model/'plans'/(name+'.json');write(file,plan);config['quantization_config']['decode_plans']=str(file)
    dest=stage/model/'models'/name;view(parent['model_path'],dest,config)
    spec.update(model_path=str(dest),config_sha256=sha(dest/'config.json'),status='mechanism ablation; not a primary candidate')
    spec['environment'].update(env_updates);methods[model][name]=spec
write(stage/'methods.json',methods)
write(stage/'additional_baselines.json',{'selection_before_final_scores':True,
    'selected':['same-quantizer W4A8 and MX-W8A8 endpoints for 3B/7B','vLLM ordinary FP8 for 0.5B/7B, conditional on native smoke'],
    'not_primary':['NVFP4/MXFP4 dense end-to-end checkpoints','INT4 weight-only AWQ/GPTQ','Mix-paper offline GPTQ'],
    'rationale':'Endpoints isolate mixed precision; ordinary FP8 adds a mature backend in the unchanged inference environment. Other formats require separate matched checkpoint/quantizer work and are audited but not part of the preregistered main matrix. Mix-I remains RTN and must not be called paper-optimal MixLLM.',
    'all_measured_selected_baselines_reported_regardless_of_rank':True})
print('Prepared optional baselines and four mechanism ablations')
