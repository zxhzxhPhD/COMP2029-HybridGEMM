"""Reuse a pinned FP16/BF16 checkpoint and one saved channel map for both paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--selection', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
selection = json.loads(args.selection.read_text())
config = json.loads((args.model/'config.json').read_text())
with safe_open(str(args.model/'model.safetensors'),framework='pt',device='cpu') as f:
    shapes={name:f.get_slice(name+'.weight').get_shape() for name in selection}
total_channels=sum(shape[0] for shape in shapes.values())
total_weights=sum(shape[0]*shape[1] for shape in shapes.values())
high_weights=sum(len(selection[name])*shape[1] for name,shape in shapes.items())
for kind in ['mixllm', 'hygemm']:
    path = args.out/kind
    path.mkdir(parents=True, exist_ok=True)
    for source in args.model.iterdir():
        if source.is_file() and source.name != 'config.json':
            target = path/source.name
            if not target.exists():
                target.symlink_to(source.resolve())
    cfg = {**config, 'quantization_config': {'quant_method':kind, 'checkpoint_format':'fp16_rtn',
        'selection':selection, 'low_id':0, 'high_id':0, 'concurrent':True}}
    (path/'config.json').write_text(json.dumps(cfg))
args.out.mkdir(parents=True, exist_ok=True)
(args.out/'manifest.json').write_text(json.dumps({'model':str(args.model.resolve()),
    'config_sha256':hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),
    'model_sha256':hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
    'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest(),
    'selection_layers':len(selection), 'selected_channels':sum(map(len,selection.values())),
    'total_quantized_channels':total_channels,'total_quantized_weights':total_weights,
    'high_precision_weights':high_weights,'high_precision_weight_fraction':high_weights/total_weights,
    'quantization':'load-time RTN; common saved channel selection; no GPTQ/clipping/recalibration'}, indent=2))
print('Prepared both adapters from', args.model)
