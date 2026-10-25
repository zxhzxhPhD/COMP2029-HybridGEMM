"""Same-HF-backend controlled weight/activation quantization error diagnosis."""
import argparse
import math
from pathlib import Path
import time
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from hygemm_vllm.native import DualLinear
from common import read,write,dq,aggregate,configure,decode_mx


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--formal',type=Path,required=True)
    ap.add_argument('--reference-math',choices=['fp16','fp32'],default='fp16')
    ap.add_argument('--output',default='diagnosis.json')
    args=ap.parse_args();configure()
    data=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['diagnostic']
    selected=read(args.root/'original_selection.json')
    model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
            attn_implementation='sdpa',local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    weights={};native={};records={};layer_stats={};rounding={};native_checks={}
    started=time.perf_counter()
    with torch.inference_mode():
        for name,m in modules.items():
            original=m.weight.detach().cpu().clone()
            versions=[];errs=[]
            for fp4 in [True,False]:
                chunks=[];err=0.
                for start in range(0,m.out_features,1024):
                    value=dq(m.weight[start:start+1024],fp4)
                    err=max(err,(value-value.half().float()).abs().max().item())
                    chunks.append(value.half().cpu())
                versions.append(torch.cat(chunks));errs.append(err)
            weights[name]=(original,*versions)
            rounding[name]={'weight4_dequant_to_half_max_abs':errs[0],'weight8_dequant_to_half_max_abs':errs[1]}
        for mode in ['fp16','w16_a8','mixed_a16','mixed_a8','w4_a8','w8_a8','native_mixed_a8']:
            for name,m in modules.items():
                original,w4,w8=weights[name]
                ids=torch.tensor(selected[name],dtype=torch.long)
                w=original if mode in ['fp16','w16_a8','native_mixed_a8'] else w4.clone()
                if mode in ['mixed_a16','mixed_a8']:w[ids]=w8[ids]
                if mode=='w8_a8':w=w8
                m.weight.copy_(w.to(m.weight.device))
                if mode=='native_mixed_a8':native[name]=DualLinear(m.weight,ids.cuda(),0,0)
                def forward(self,x,n=name,variant=mode):
                    flat=x.reshape(-1,self.in_features).contiguous()
                    if variant=='native_mixed_a8':
                        y=native[n](flat)
                        if n not in native_checks:
                            r=native[n];decoded_x=dq(flat)
                            decoded_w=torch.empty(self.out_features,self.in_features,device=flat.device,dtype=torch.float32)
                            decoded_w[r.lo.long()]=decode_mx(r.w4,r.sf4,r.k,True)
                            decoded_w[r.hi.long()]=decode_mx(r.w8,r.sf8,r.k,False)
                            reference=(decoded_x@decoded_w.t()).half()
                            torch.testing.assert_close(y,reference,atol=.002,rtol=.002)
                            native_checks[n]={'passed':True,'max_abs':(y-reference).abs().max().item(),
                                'rmse':(y.float()-reference.float()).square().mean().sqrt().item()}
                    else:
                        if variant.endswith('a8'):
                            exact=dq(flat)
                            if n not in records.setdefault('activation_rounding',{}):
                                records['activation_rounding'][n]=(exact-exact.half().float()).abs().max().item()
                            flat=exact.half()
                        y=(F.linear(flat.float(),self.weight.float(),None).half() if args.reference_math=='fp32'
                           else F.linear(flat,self.weight,None))
                    if self.bias is not None:y=y+self.bias
                    return y.reshape(*x.shape[:-1],self.out_features)
                m.forward=types.MethodType(forward,m)
            blocks=[];lps=[]
            for i,ids in enumerate(data):
                logits=model(ids[None].cuda(),use_cache=False).logits[0,:-1].float()
                losses=F.cross_entropy(logits,ids[1:].cuda(),reduction='none')
                assert torch.isfinite(losses).all()
                blocks.append({'nll':losses.double().sum().item(),'tokens':losses.numel()})
                if i<2:lps.extend((-losses).cpu().tolist())
                del logits,losses
            records[mode]={**aggregate(blocks),'first_two_target_logprobs':lps}
            if mode=='native_mixed_a8':
                delta=max(abs(x-y) for x,y in zip(lps,records['mixed_a8']['first_two_target_logprobs']))
                mean_delta=abs(records[mode]['mean_nll']-records['mixed_a8']['mean_nll'])
                records['native_software_alignment']={'target_logprob_max_abs':delta,'mean_nll_abs':mean_delta,
                    'same_backend':'HF SDPA; only native vs dequantized dense linear differs',
                    'native_layer_checks':native_checks,'software_reference_is_approximate':True,
                    'interpretation':'Use native full-model scores for selection. Software counterfactual NLL effects near this discrepancy are unresolved; native per-layer reference tolerance is independently enforced.'}
                write(args.root/args.output,{'completed':False,'modes':records,'weight_half_rounding':rounding})
                for r in native.values():r.close()
                native.clear()
            print('Diagnosis',args.root.name,mode,records[mode]['ppl'],flush=True)
            write(args.root/args.output,{'completed':False,'modes':records,'weight_half_rounding':rounding})
    write(args.root/args.output,{'completed':True,'backend':f'HF SDPA, {args.reference_math} dense GEMM, FP16 output; native pack/dequant reference',
        'validation_shape':list(data.shape),'modes':records,'weight_half_rounding':rounding,
        'interpretation':'Controlled perturbations, not an additive causal decomposition; endpoints may exceed budget; not speed measurements.',
        'elapsed_s':time.perf_counter()-started})


if __name__=='__main__':main()
