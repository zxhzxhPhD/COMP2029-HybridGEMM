"""Freeze only a twice-confirmed, numerically accepted model configuration."""
import argparse
import json
import math
import os
from pathlib import Path
from prepare import read,write,sha,runtime_sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
ap.add_argument('--baseline');ap.add_argument('--candidate',default='combination_full');ap.add_argument('--repeat',default='combination_repeat')
ap.add_argument('--quality',default='quality_combination');ap.add_argument('--mix-label',default='mix_original');args=ap.parse_args()
stage=args.stage;folder=stage/args.model;out=folder/'final';out.mkdir(exist_ok=True)
baseline=args.baseline or (read(folder/'selection_decision.json')['baseline'] if (folder/'selection_decision.json').exists() else 'baseline_full')
def cells(label):return {(r['batch'],r['context']):r for r in read(folder/label/'result.json')['cells']}
a,b,c=cells(baseline),cells(args.candidate),cells(args.repeat)
assert set(a)==set(b)==set(c)=={(t,p) for t in [1,2,4,8,16,32,64,128] for p in [128,1024]}
groups=[]
for value in [b,c]:
    ratios=[a[k]['median_ms_per_step']/value[k]['median_ms_per_step'] for k in sorted(a)]
    groups.append({'geomean_speedup':math.prod(ratios)**(1/len(ratios)),'max_latency_ratio':1/min(ratios)})
assert groups[0]['geomean_speedup']>=1.01 and groups[1]['geomean_speedup']>1.0,groups
assert all(g['max_latency_ratio']<=1.02 for g in groups),groups
quality=read(folder/args.quality/'quality.json');assert quality['comparison']['passed']
ppl=read(folder/'validation_ppl.json');assert ppl['passed']
winner=read(folder/args.candidate/'result.json');config=read(folder/args.candidate/'model/config.json')
plans=read(config['quantization_config']['decode_plans']);binding=plans['binding'];identity=winner['identity']
assert binding['library_sha256']==identity['kernel_sha256']==quality['kernel_sha256']
options={'HYGEMM_PACK_MODE':identity['pack_mode'],'HYGEMM_FUSE_SILU':identity.get('fused_silu','0'),
    'HYGEMM_SILU_MATH':identity.get('producer_math','0'),'HYGEMM_ROW_ORDER':identity.get('row_order','none'),
    'HYGEMM_FUSED_GATEUP':identity.get('fused_gateup_max_tokens','0')}
binding.update(runtime_options=options,producer_sha256=identity.get('producer_sha256'),fused_sha256=identity.get('fused_library_sha256'),
               runtime_source_sha256=runtime_sha())
planfile=out/'plans.json';write(planfile,plans)
config['quantization_config']['decode_plans']=str(planfile)
model=out/'model';model.mkdir(exist_ok=True)
for file in (folder/args.candidate/'model').iterdir():
    if file.is_file() and file.name!='config.json' and not (model/file.name).exists():(model/file.name).symlink_to(file.resolve())
write(model/'config.json',config)
environment={**options,'HYGEMM_LIBRARY':str(stage/'build/libhygemm_serving.so'),
    'HYGEMM_PRODUCER_LIBRARY':str(stage/'build/libhygemm_producer_v2.so'),
    'MIXLLM_LIBRARY':'/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so'}
if int(options['HYGEMM_FUSED_GATEUP']):environment['HYGEMM_FUSED_LIBRARY']=str(stage/'fused_serving_build/libfused_probe.so')
meta=read(stage/'inputs.json')[args.model]
assert winner['budget']['persistent_quantized_storage_bytes']==meta['budget']['persistent_quantized_bytes']
record={'status':'frozen','model':args.model,'model_path':str(model),'environment':environment,'baseline':baseline,
    'candidate':args.candidate,'repeat':args.repeat,'quality':args.quality,'groups':groups,'ppl':ppl['ppl'],
    'budget':meta['budget'],'selection_sha256':meta['selection_sha256'],'plans_sha256':sha(planfile),
    'config_sha256':sha(model/'config.json'),'binding':binding,'mix_baseline':args.mix_label,
    'mix_model_path':str(folder/args.mix_label/'model'),'source_commit':'pending stage-two commit',
    'protocol':{'graph':'FULL','inductor':False,'dtype':'float16','kv_dtype':'float16','kv_cache_bytes':10*1024**3,
                'capture_sizes':[1,2,4,8,16,32,64,128],'max_num_seqs':128,'max_num_batched_tokens':2048,
                'max_model_len':2049,'attention':'FLASH_ATTN','prefix_cache':False}}
write(out/'manifest.json',record)
