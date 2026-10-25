"""Compare native, FP32 dense, and FP16 dense on identical real activations."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from common import configure,read,write,dq
from hygemm_vllm.native import DualLinear,pack
from test_components import decode_mx

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--formal',type=Path,required=True);args=ap.parse_args();configure()
model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
selection=read(args.root/'original_selection.json')
records=[];hooks=[]
for name,m in model.named_modules():
    if not isinstance(m,torch.nn.Linear) or not name.startswith('model.layers.'):continue
    def check(module,inputs,n=name):
        x=inputs[0].reshape(-1,module.in_features).contiguous()
        r=DualLinear(module.weight,torch.tensor(selection[n],device='cuda'),0,0)
        native=r(x);dx=dq(x)
        w=torch.empty(module.out_features,module.in_features,device='cuda',dtype=torch.float32)
        w[r.lo.long()]=decode_mx(r.w4,r.sf4,r.k,True)
        w[r.hi.long()]=decode_mx(r.w8,r.sf8,r.k,False)
        dense32=F.linear(dx,w).half();dense16=F.linear(dx.half(),w.half())
        row={'layer':n,'native_fp32_max_abs':(native-dense32).abs().max().item(),
             'native_fp32_rmse':(native.float()-dense32.float()).square().mean().sqrt().item(),
             'native_fp16_max_abs':(native-dense16).abs().max().item(),
             'fp32_fp16_max_abs':(dense32-dense16).abs().max().item(),
             'native_fp32_mismatches':int((~torch.isclose(native,dense32,atol=.002,rtol=.002)).sum()),
             'x_half_error':(dx-dx.half()).abs().max().item(),'w_half_error':(w-w.half()).abs().max().item()}
        # Independently quantize original row order and verify partitioning does not change the values.
        w4=dq(module.weight,True);w8=dq(module.weight,False);w4[r.hi.long()]=w8[r.hi.long()]
        row['weight_partition_max_abs']=(w-w4).abs().max().item()
        records.append(row);r.close();print(row,flush=True)
    hooks.append(m.register_forward_pre_hook(check))
with torch.inference_mode():
    ids=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['diagnostic'][0]
    model(ids[None].cuda(),use_cache=False)
write(args.root/'reference_check.json',{'completed':True,'records':records})
