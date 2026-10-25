"""Floating-point promotion benefit scores with original per-layer counts."""
import argparse
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM
from common import read,write,dq,configure,tensor_sha

METHODS=['weight_gain','output_gain','output_weight_gain','loss_abs_gain','loss_signed_gain']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--formal',type=Path,required=True);ap.add_argument('--seed',type=int,default=2029)
    ap.add_argument('--samples',type=int,default=32);args=ap.parse_args();configure()
    cal=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['calibration'][args.seed][:args.samples]
    model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
            attn_implementation='sdpa',local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    scores={method:{n:torch.zeros(m.out_features,device='cuda') for n,m in modules.items()} for method in METHODS}
    diagnostics={n:{k:torch.zeros(m.out_features,device='cuda') for k in ['e4','e8','y2']} for n,m in modules.items()}
    inputs={};handles=[];sample_index=0;activation={}
    for name,module in modules.items():
        def pre(m,a,n=name):inputs[n]=a[0].detach()
        def bwd(m,grad,n=name):
            raw=inputs.pop(n).reshape(-1,m.in_features).contiguous()
            x=raw.float();dx=dq(raw);dy=grad[0].detach().reshape(-1,m.out_features).float()
            if n not in activation:
                activation[n]={'relative_mse':((dx-x).square().sum()/x.square().sum().clamp_min(1e-30)).item(),
                    'nonzero_to_zero_fraction':((x!=0)&(dx==0)).float().mean().item(),
                    'dequant_to_half_max_abs':(dx-dx.half().float()).abs().max().item()}
            for begin in range(0,m.out_features,1024):
                sl=slice(begin,begin+1024)
                raw_w=m.weight[sl];w=raw_w.float();w4=dq(raw_w,True);w8=dq(raw_w,False)
                target=x@w.t();p4=dx@w4.t();p8=dx@w8.t()
                e4=(p4-target).square().mean(0);e8=(p8-target).square().mean(0)
                scores['output_gain'][n][sl]+=e4-e8
                weight_target=dx@w.t()
                scores['output_weight_gain'][n][sl]+=(p4-weight_target).square().mean(0)-(p8-weight_target).square().mean(0)
                diagnostics[n]['e4'][sl]+=e4;diagnostics[n]['e8'][sl]+=e8
                diagnostics[n]['y2'][sl]+=target.square().mean(0)
                g=dy[:,sl].t()@x
                t4=(g*(w4-w)).sum(1);t8=(g*(w8-w)).sum(1)
                a4=t4+.5*t4.square();a8=t8+.5*t8.square()
                scores['loss_abs_gain'][n][sl]+=a4.abs()-a8.abs()
                scores['loss_signed_gain'][n][sl]+=a4-a8
                if sample_index==0:
                    scores['weight_gain'][n][sl]=(w4-w).square().sum(1)-(w8-w).square().sum(1)
        handles.extend([module.register_forward_pre_hook(pre),module.register_full_backward_pre_hook(bwd)])
    losses=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for sample_index,ids in enumerate(cal):
        ids=ids[None].cuda()
        embedded=model.get_input_embeddings()(ids).detach().requires_grad_(True)
        loss=model(inputs_embeds=embedded,labels=ids,use_cache=False).loss
        torch.autograd.grad(loss,embedded)
        losses.append(loss.item());assert not inputs
        del loss,embedded
        print('Score',args.root.name,args.seed,sample_index+1,'/',len(cal),flush=True)
    for h in handles:h.remove()
    scores={method:{n:x.cpu()/(1 if method=='weight_gain' else len(cal)) for n,x in values.items()} for method,values in scores.items()}
    diagnostics={n:{k:x.cpu()/len(cal) for k,x in values.items()} for n,values in diagnostics.items()}
    assert all(torch.isfinite(x).all() for values in scores.values() for x in values.values())
    folder=args.root/'scores'/str(args.seed);folder.mkdir(parents=True,exist_ok=True)
    torch.save({'scores':scores,'output_errors':diagnostics},folder/'scores.pt')
    write(folder/'manifest.json',{'completed':True,'seed':args.seed,'samples':len(cal),'calibration_sha256':tensor_sha(cal),
        'methods':METHODS,'activation_first_calibration_sample':activation,'calibration_losses':losses,
        'score_definitions':{'weight_gain':'sum((Q4-W)^2-(Q8-W)^2)',
            'output_gain':'E[(Xhat Q4-XW)^2-(Xhat Q8-XW)^2]',
            'output_weight_gain':'E[(Xhat Q4-Xhat W)^2-(Xhat Q8-Xhat W)^2], diagnostic only',
            'loss_abs_gain':'E[abs(t4+0.5*t4^2)-abs(t8+0.5*t8^2)], tb=sum(dL/dW*(Qb-W))',
            'loss_signed_gain':'E[(t4+0.5*t4^2)-(t8+0.5*t8^2)], heuristic signed proxy, not exact Hessian'},
        'elapsed_s':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
