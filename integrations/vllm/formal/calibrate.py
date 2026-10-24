"""Common full MixLLM channel scores, streamed by output rows to fit 7B/32GB."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from hygemm_vllm.mixllm import quantize_weight

ap=argparse.ArgumentParser()
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--train',type=Path,required=True)
ap.add_argument('--test',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);torch.manual_seed(2029)
torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
train=torch.tensor(tok(args.train.read_text(),add_special_tokens=False)['input_ids'])
test=torch.tensor(tok(args.test.read_text(),add_special_tokens=False)['input_ids'])
starts=torch.randint(0,train.numel()-512,(32,),generator=torch.Generator().manual_seed(2029))
cal=torch.stack([train[i:i+512] for i in starts])
torch.save({'calibration':cal,'test':test},args.out/'tokens.pt')
start=time.perf_counter()
model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,
                                        attn_implementation='sdpa',local_files_only=True).eval().cuda()
model.requires_grad_(False)
modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
scores={n:torch.zeros(m.out_features,device='cuda',dtype=torch.float32) for n,m in modules.items()}
first={n:torch.zeros_like(v) for n,v in scores.items()}
inputs={};handles=[]
for name,module in modules.items():
    def pre(m,a,n=name):inputs[n]=a[0].detach()
    def bwd(m,grad,n=name):
        x=inputs.pop(n).reshape(-1,m.in_features).float()
        dy=grad[0].detach().reshape(-1,m.out_features).float()
        # No all-model FP32 delta or parameter-gradient allocation.
        for begin in range(0,m.out_features,256):
            w=m.weight[begin:begin+256]
            q,s,z=quantize_weight(w,4)
            dq=((q.float().reshape(w.shape[0],-1,128)-z.t()[...,None])*s.t()[...,None]).reshape_as(w)
            g=dy[:,begin:begin+w.shape[0]].t()@x
            t=(g*(dq-w.float())).sum(1)
            scores[n][begin:begin+w.shape[0]]+=(t+.5*t.square()).abs()
            first[n][begin:begin+w.shape[0]]+=t.abs()
    handles.extend([module.register_forward_pre_hook(pre),module.register_full_backward_pre_hook(bwd)])
losses=[];torch.cuda.reset_peak_memory_stats()
for i,ids in enumerate(cal):
    ids=ids[None].cuda()
    embed=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    loss=model(inputs_embeds=embed,labels=ids,use_cache=False).loss
    torch.autograd.grad(loss,embed)
    losses.append(loss.item())
    assert not inputs
    del loss,embed
    print(f'calibration {i+1}/32 loss={losses[-1]:.5f}',flush=True)
for h in handles:h.remove()
scores={n:v.cpu()/32 for n,v in scores.items()}
assert all(torch.isfinite(v).all() for v in scores.values())
names=list(scores);flat=torch.cat(list(scores.values()))
mask=torch.zeros(flat.numel(),dtype=torch.bool)
mask[flat.argsort(descending=True,stable=True)[:round(flat.numel()*.1)]]=True
selection={};offset=0
for n,v in scores.items():
    selection[n]=torch.where(mask[offset:offset+v.numel()])[0].tolist();offset+=v.numel()
(args.out/'selection.json').write_text(json.dumps(selection))
torch.save({'scores':scores,'first':{n:v.cpu()/32 for n,v in first.items()}},args.out/'scores.pt')
shapes={n:list(m.weight.shape) for n,m in modules.items()}
manifest={'model':str(args.model),'method':'E[abs(t+0.5*t^2)], t=sum_K(gradient*RTN_INT4_delta)',
          'seed':2029,'samples':32,'calibration_length':512,'train_sha256':hashlib.sha256(args.train.read_bytes()).hexdigest(),
          'test_sha256':hashlib.sha256(args.test.read_bytes()).hexdigest(),'test_tokens':test.numel(),
          'selection_sha256':hashlib.sha256((args.out/'selection.json').read_bytes()).hexdigest(),
          'budget':'10% output channels, global, stable sort','shapes':shapes,'calibration_losses':losses,
          'elapsed_s':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'channels':flat.numel(),'high_channels':int(mask.sum()),
          'weights':sum(o*k for o,k in shapes.values()),
          'high_weights':sum(len(selection[n])*k for n,(o,k) in shapes.items())}
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Calibration complete',manifest['elapsed_s'],flush=True)
