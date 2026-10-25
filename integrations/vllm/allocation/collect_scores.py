"""Recompute all promotion scores for an offline bank or a larger calibration set."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from alloc_utils import *
from fp4_quant import decode_logical


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--formal',type=Path,required=True);ap.add_argument('--bank',type=Path)
    ap.add_argument('--samples',type=int,default=32);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();configure();torch.use_deterministic_algorithms(True)
    original=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['calibration'][2029]
    if args.samples==32:cal=original
    else:
        tok=AutoTokenizer.from_pretrained(args.formal/'models/fp16',local_files_only=True)
        train=torch.tensor(tok((args.formal.parent/'data/wikitext2_train.txt').read_text(),add_special_tokens=False)['input_ids'])
        starts=torch.randint(0,train.numel()-512,(args.samples,),generator=torch.Generator().manual_seed(2029))
        cal=torch.stack([train[i:i+512] for i in starts]);assert torch.equal(cal[:32],original)
    model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda();model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    bank={}
    if args.bank:
        manifest=read(args.bank);assert manifest['completed']
        for n,spec in manifest['layers'].items():
            path=args.bank.parent/spec['file'];assert sha(path)==spec['sha256']
            bank[n]=torch.load(path,map_location='cpu',weights_only=True)
    scores={f:{n:torch.zeros(m.out_features,device='cuda') for n,m in modules.items()} for f in FAMILIES}
    paired={f:{n:torch.zeros(m.out_features,device='cuda') for n,m in modules.items()} for f in FAMILIES} if args.bank else None
    inputs={};hooks=[];sample_index=0
    for name,m in modules.items():
        def pre(module,values,n=name):inputs[n]=values[0].detach()
        def backward(module,grad,n=name):
            raw=inputs.pop(n).reshape(-1,module.in_features).contiguous();x=raw.float();dx=dq(raw)
            dy=grad[0].detach().reshape(-1,module.out_features).float()
            for start in range(0,module.out_features,1024):
                sl=slice(start,start+1024);raw_w=module.weight[sl];w=raw_w.float();w8=dq(raw_w,False)
                if bank:
                    q=bank[n]['q4'][sl].to(w.device);sf=bank[n]['scales'][sl].to(w.device)
                    w4=decode_logical(q,sf,module.in_features)
                else:w4=dq(raw_w,True)
                target=x@w.t();e4=(dx@w4.t()-target).square().mean(0);e8=(dx@w8.t()-target).square().mean(0)
                scores['output_gain'][n][sl]+=e4-e8
                g=dy[:,sl].t()@x;t4=(g*(w4-w)).sum(1);t8=(g*(w8-w)).sum(1)
                a4=t4+.5*t4.square();a8=t8+.5*t8.square()
                scores['loss_abs_gain'][n][sl]+=a4.abs()-a8.abs();scores['loss_signed_gain'][n][sl]+=a4-a8
                if sample_index==0:scores['weight_gain'][n][sl]=(w4-w).square().sum(1)-(w8-w).square().sum(1)
                if paired is not None:
                    # Both quantizers see exactly the same teacher X and dL/dY.
                    native4=dq(raw_w,True)
                    paired['output_gain'][n][sl]+=(dx@native4.t()-target).square().mean(0)-e8
                    native_t=(g*(native4-w)).sum(1);native_a=native_t+.5*native_t.square()
                    paired['loss_abs_gain'][n][sl]+=native_a.abs()-a8.abs()
                    paired['loss_signed_gain'][n][sl]+=native_a-a8
                    if sample_index==0:paired['weight_gain'][n][sl]=(native4-w).square().sum(1)-(w8-w).square().sum(1)
        hooks.extend([m.register_forward_pre_hook(pre),m.register_full_backward_pre_hook(backward)])
    start=time.perf_counter();losses=[]
    for sample_index,ids in enumerate(cal):
        ids=ids[None].cuda();embedding=model.get_input_embeddings()(ids).detach().requires_grad_(True)
        loss=model(inputs_embeds=embedding,labels=ids,use_cache=False).loss
        torch.autograd.grad(loss,embedding);assert not inputs
        losses.append(loss.item());del loss,embedding
        print('Scores',args.root.name,args.samples,sample_index+1,flush=True)
    for h in hooks:h.remove()
    values={f:{n:x.cpu()/(1 if f=='weight_gain' else len(cal)) for n,x in rows.items()} for f,rows in scores.items()}
    assert all(torch.isfinite(x).all() for rows in values.values() for x in rows.values())
    references={f:{n:x.cpu()/(1 if f=='weight_gain' else len(cal)) for n,x in rows.items()} for f,rows in paired.items()} if paired is not None else values
    args.out.mkdir(parents=True,exist_ok=True);torch.save({'scores':values,'paired_rtn_scores':references},args.out/'scores.pt')
    torch.save(cal,args.out/'calibration_tokens.pt')
    write(args.out/'manifest.json',{'completed':True,'bank':str(args.bank) if args.bank else None,
        'source_sha256':sha(__file__),
        'teacher_deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'paired_rtn_scores':True,
        'calibration_file_sha256':sha(args.out/'calibration_tokens.pt'),
        'train_text_sha256':sha(args.formal.parent/'data/wikitext2_train.txt'),
        'bank_sha256':sha(args.bank) if args.bank else None,'samples':len(cal),'length':512,'seed':2029,
        'calibration_sha256':tensor_sha(cal),'first32_match_q1':True,'losses':losses,'elapsed_s':time.perf_counter()-start})


if __name__=='__main__':main()
