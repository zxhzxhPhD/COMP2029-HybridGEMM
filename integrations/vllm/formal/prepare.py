"""Prepare shared-map model views and all actual fused projection shapes."""
import argparse
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--calibration',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
sel=json.loads((args.calibration/'selection.json').read_text())
meta=json.loads((args.calibration/'manifest.json').read_text())
cfg=json.loads((args.model/'config.json').read_text())
cases={}
layers=[]
for layer in range(cfg['num_hidden_layers']):
    stem=f'model.layers.{layer}.'
    for path,parts in [('self_attn.qkv_proj',['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj']),
                       ('self_attn.o_proj',['self_attn.o_proj']),
                       ('mlp.gate_up_proj',['mlp.gate_proj','mlp.up_proj']),('mlp.down_proj',['mlp.down_proj'])]:
        offset=0;ids=[];k=None
        for part in parts:
            name=stem+part;o,k=meta['shapes'][name]
            ids.extend(offset+i for i in sel[name]);offset+=o
        row={'layer':stem+path,'out':offset,'k':k,'high':len(ids),'high_ids':sorted(ids)}
        layers.append(row);cases.setdefault((offset,k,len(ids)),row)
(args.out/'cases.json').write_text(json.dumps({'layers':layers,'cases':list(cases.values())},indent=2))
for kind in ['fp16','mixllm','hygemm']:
    dest=args.out/'models'/kind;dest.mkdir(parents=True,exist_ok=True)
    for file in args.model.iterdir():
        if file.is_file() and file.name!='config.json' and not (dest/file.name).exists():
            (dest/file.name).symlink_to(file.resolve())
    new=dict(cfg)
    if kind!='fp16':
        new['quantization_config']={'quant_method':kind,'checkpoint_format':'fp16_rtn','selection':sel,
                                   'tuning_plan':str((args.out/'plans.json').resolve())}
    (dest/'config.json').write_text(json.dumps(new))
print('Prepared',len(layers),'layers,',len(cases),'exact partition shapes')
