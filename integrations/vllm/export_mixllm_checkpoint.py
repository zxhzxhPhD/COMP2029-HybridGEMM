"""Export the original five-tensor MixLLM checkpoint schema for loader validation.

This is RTN with an existing selection, not a new calibration or GPTQ run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file, save_file
from hygemm_vllm.mixllm import quantize_weight

ap=argparse.ArgumentParser()
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--selection',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
selection=json.loads(args.selection.read_text())
state=load_file(str(args.model/'model.safetensors'))
for name,ids in selection.items():
    w=state.pop(name+'.weight').to(device='cuda',dtype=torch.float16)
    mask=torch.zeros(w.shape[0],dtype=torch.bool,device=w.device)
    mask[ids]=True
    hi,lo=torch.where(mask)[0].int(),torch.where(~mask)[0].int()
    q8,s8,z8=quantize_weight(w[hi.long()],8)
    q4,s4,z4=quantize_weight(w[lo.long()],4)
    combined={
        'qweight':torch.cat((q8,q4.to(torch.int8)),dim=0).contiguous(),
        'scale':torch.cat((s8,s4),dim=1).contiguous(),
        'zero':torch.cat((z8,z4),dim=1).contiguous(),
        'indices':torch.cat((hi,lo)).contiguous(),
        'is_int8_channel':torch.cat((torch.ones_like(hi,dtype=torch.bool),torch.zeros_like(lo,dtype=torch.bool)))
    }
    state.update({name+'.'+k:v.cpu() for k,v in combined.items()})
save_file(state,str(args.out/'model.safetensors'))
for p in args.model.iterdir():
    if p.is_file() and p.name not in ['config.json','model.safetensors']:
        shutil.copy2(p,args.out/p.name)
cfg=json.loads((args.model/'config.json').read_text())
cfg['quantization_config']={'quant_method':'mixllm','ratio':.1,'checkpoint_format':'mixllm_v1'}
(args.out/'config.json').write_text(json.dumps(cfg,indent=2))
(args.out/'export_manifest.json').write_text(json.dumps({
    'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest(),
    'source_model_sha256':hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
    'layers':len(selection),'format':'original qweight/scale/zero/indices/is_int8_channel; RTN'},indent=2))
print('Exported',len(selection),'MixLLM projection tensors')
