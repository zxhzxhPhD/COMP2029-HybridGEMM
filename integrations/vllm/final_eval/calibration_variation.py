"""Paired calibration factors: frozen signed ranking plus recalibrated clipping."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
import time
from final_common import *
os.environ.update(environment())
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'allocation'))
from alloc_utils import configure,dq,tensor_sha
from fp4_quant import quantize
from prepare_budget import create_method


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--seed',type=int,default=2029)
    ap.add_argument('--samples',type=int,default=32);ap.add_argument('--length',type=int,default=512)
    ap.add_argument('--corpus',choices=['train','c4_train'],default='train');args=ap.parse_args()
    configure();torch.use_deterministic_algorithms(True)
    stage=args.stage;out=stage/args.model/'robustness'/args.label;out.mkdir(parents=True,exist_ok=True)
    data=torch.load(stage/'data'/args.model/'tokens.pt',map_location='cpu',weights_only=True)[args.corpus]
    gen=torch.Generator().manual_seed(args.seed)
    # Draw the original 512-window starts, repairing only starts too close to
    # the corpus end. All sample-count/length variants share these 128 starts.
    starts=torch.randint(0,len(data)-512,(128,),generator=gen);repaired=[]
    for i,start in enumerate(starts.tolist()):
        if start>len(data)-2048:
            starts[i]=torch.randint(0,len(data)-2048,(1,),generator=gen)[0];repaired.append(i)
    cal=torch.stack([data[i:i+args.length] for i in starts[:args.samples]])
    identity={'model':args.model,'seed':args.seed,'samples':args.samples,'length':args.length,'corpus':args.corpus,
        'calibration_token_sha256':tensor_sha(cal),'source_sha256':sha(__file__),'repaired_start_indices':repaired,
        'score':'fixed-quota signed RTN promotion score','bank':'act_mse_clip recomputed from the same calibration setting',
        'deterministic_algorithms':True,'test_scoring':False}
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    if (out/'complete.json').exists():assert read(out/'complete.json')['completed'];return
    write(out/'identity.json',identity);torch.save(cal,out/'calibration_tokens.pt')
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(FORMAL/args.model/'models/fp16',dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda();model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    torch.cuda.reset_peak_memory_stats()
    if (out/'scores.pt').exists() and (out/'moments.pt').exists():
        scores=torch.load(out/'scores.pt',map_location='cpu',weights_only=True)
        moments=torch.load(out/'moments.pt',map_location='cpu',weights_only=True)
    else:
        sums={n:torch.zeros(m.in_features,device='cuda',dtype=torch.float64) for n,m in modules.items()}
        counts={n:0 for n in modules};scores={n:torch.zeros(m.out_features,device='cuda') for n,m in modules.items()}
        inputs={};hooks=[]
        for name,m in modules.items():
            def pre(module,values,n=name):inputs[n]=values[0].detach()
            def backward(module,grad,n=name):
                raw=inputs.pop(n).reshape(-1,module.in_features).contiguous();x=raw.float();dx=dq(raw)
                sums[n]+=dx.square().sum(0,dtype=torch.float64);counts[n]+=len(dx)
                dy=grad[0].detach().reshape(-1,module.out_features).float()
                for start in range(0,module.out_features,1024):
                    sl=slice(start,start+1024);raw_w=module.weight[sl];w=raw_w.float()
                    w4=dq(raw_w,True);w8=dq(raw_w,False);g=dy[:,sl].t()@x
                    t4=(g*(w4-w)).sum(1);t8=(g*(w8-w)).sum(1)
                    scores[n][sl]+=(t4+.5*t4.square())-(t8+.5*t8.square())
            hooks.extend([m.register_forward_pre_hook(pre),m.register_full_backward_pre_hook(backward)])
        losses=[]
        for i,ids in enumerate(cal):
            ids=ids[None].cuda();embedding=model.get_input_embeddings()(ids).detach().requires_grad_(True)
            loss=model(inputs_embeds=embedding,labels=ids,use_cache=False).loss
            torch.autograd.grad(loss,embedding);assert not inputs
            losses.append(loss.item());del loss,embedding
            print('Calibration',args.model,args.label,i+1,'/',len(cal),flush=True)
        for hook in hooks:hook.remove()
        scores={n:x.cpu()/len(cal) for n,x in scores.items()}
        moments={n:(sums[n]/counts[n]).float().cpu() for n in modules}
        assert all(torch.isfinite(v).all() for v in scores.values())
        torch.save(scores,out/'scores.pt');torch.save(moments,out/'moments.pt')
        write(out/'teacher.json',{'losses':losses,'score_elapsed_s':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    bank=out/'bank';bank.mkdir(exist_ok=True);bankpath=bank/'manifest.json'
    manifest=read(bankpath) if bankpath.exists() else {'mode':'act_mse_clip','format':'E2M1 packed nibbles + logical K32 E8M0 scales',
        'layers':{},'completed':False,'offsets':[0,-1,-2],'activation_unchanged':True,'fp8_weight_unchanged':True,
        'input_moments_sha256':sha(out/'moments.pt')}
    with torch.inference_mode():
        for name,m in modules.items():
            if name in manifest['layers']:
                assert sha(bank/manifest['layers'][name]['file'])==manifest['layers'][name]['sha256'];continue
            qs=[];ss=[];stats=[]
            for start in range(0,m.out_features,1024):
                q,s,st=quantize(m.weight[start:start+1024],moments[name],offsets=(0,-1,-2))
                qs.append(q.cpu());ss.append(s.cpu());stats.append(st)
            path=bank/(name+'.pt');torch.save({'q4':torch.cat(qs),'scales':torch.cat(ss)},path)
            manifest['layers'][name]={'file':path.name,'sha256':sha(path),'rows':m.out_features,'k':m.in_features,
                'weighted_error_before':sum(v['weighted_error_before'] for v in stats),
                'weighted_error_after':sum(v['weighted_error_after'] for v in stats)}
            write(bankpath,manifest)
    manifest['completed']=True;write(bankpath,manifest)
    original=read(Q1/args.model/'original_selection.json')
    selection={n:sorted(scores[n].argsort(descending=True,stable=True)[:len(original[n])].tolist()) for n in original}
    name='R-'+args.label
    record=create_method(stage,args.model,name,selection,{'calibration':identity})
    methods=read(stage/'methods.json');spec=methods[args.model][name];configfile=Path(spec['model_path'])/'config.json'
    cfg=read(configfile);cfg['quantization_config']['q4_bank']=str(bankpath);write(configfile,cfg)
    planfile=Path(cfg['quantization_config']['decode_plans']);plans=read(planfile);plans['binding']['bank_manifest_sha256']=sha(bankpath);write(planfile,plans)
    spec['config_sha256']=sha(configfile);spec['status']='calibration variation prepared; same counts and inherited frozen tiling';write(stage/'methods.json',methods)
    main_cfg=read(Path(methods[args.model]['Hy-F-Q']['model_path'])/'config.json')['quantization_config']['selection']
    intersection=sum(len(set(selection[n])&set(main_cfg[n])) for n in selection)
    union=sum(len(set(selection[n])|set(main_cfg[n])) for n in selection)
    write(out/'complete.json',{'completed':True,**record,'elapsed_s':time.perf_counter()-started,
        'mask_jaccard_vs_primary':intersection/union,'same_per_layer_high_counts':all(len(selection[n])==len(main_cfg[n]) for n in selection),
        'bank_sha256':sha(bankpath),'source_sha256':sha(__file__)})


if __name__=='__main__':main()
