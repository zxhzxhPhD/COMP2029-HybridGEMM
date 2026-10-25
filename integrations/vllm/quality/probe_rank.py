"""Bounded single-projection loss checks of shortlisted channel rankings."""
import argparse
import math
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from hygemm_vllm.native import DualLinear
from common import read,write,configure,aggregate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--formal',type=Path,required=True);args=ap.parse_args();configure()
    original=read(args.root/'original_selection.json')
    shortlist=read(args.root.parent/'shortlist.json')['families']
    scores=torch.load(args.root/'scores/2029/scores.pt',map_location='cpu',weights_only=True)['scores']
    windows=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['diagnostic'][:8]
    model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
            attn_implementation='sdpa',local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n in original}
    eligible=[n for n,m in modules.items() if n.endswith(('.q_proj','.down_proj')) and 0<len(original[n])<m.out_features]
    chosen=[eligible[i] for i in sorted({0,len(eligible)//2,len(eligible)-1})]
    records=[]
    with torch.inference_mode():
        for name in chosen:
            m=modules[name];count=len(original[name]);saved=m.forward
            gen=torch.Generator().manual_seed(2029)
            masks={'original':original[name], 'random':torch.randperm(m.out_features,generator=gen)[:count].tolist()}
            for family in shortlist:
                order=scores[family][name].argsort(descending=True,stable=True)
                masks[family+'_top']=order[:count].tolist()
                masks[family+'_bottom']=order[-count:].tolist()
            for label,ids in masks.items():
                runtime=DualLinear(m.weight,torch.tensor(sorted(ids),device='cuda'),0,0)
                def forward(self,x):
                    y=runtime(x.reshape(-1,self.in_features).contiguous())
                    if self.bias is not None:y=y+self.bias
                    return y.reshape(*x.shape[:-1],self.out_features)
                m.forward=types.MethodType(forward,m)
                blocks=[]
                for tokens in windows:
                    logits=model(tokens[None].cuda(),use_cache=False).logits[0,:-1].float()
                    loss=F.cross_entropy(logits,tokens[1:].cuda(),reduction='none')
                    assert torch.isfinite(loss).all()
                    blocks.append({'nll':loss.double().sum().item(),'tokens':loss.numel()})
                    del logits,loss
                records.append({'layer':name,'method':label,'high_count':count,'outputs':m.out_features,
                                'predicted_score_sum':{f:float(scores[f][name][ids].sum()) for f in shortlist},
                                **aggregate(blocks)})
                runtime.close();del runtime
                print('Rank probe',args.root.name,name,label,records[-1]['mean_nll'],flush=True)
            m.forward=saved
    write(args.root/'rank_probe.json',{'completed':True,'families':shortlist,'projection_selection':chosen,
        'windows':8,'length':512,'scope':'One projection quantized with frozen native kernel; other projections FP16. Diagnostic only, exceeds deployed bit budget.',
        'records':records})


if __name__=='__main__':main()
