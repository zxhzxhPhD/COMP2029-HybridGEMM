"""Three reusable full-row FP4 banks; high FP8 weights/activation stay native."""
import argparse
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM
from alloc_utils import *
from fp4_quant import quantize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--formal',type=Path,required=True);args=ap.parse_args();configure()
    modes=['native_rtn','mse_clip','act_mse_clip']
    model=AutoModelForCausalLM.from_pretrained(args.formal/'models/fp16',dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda();model.requires_grad_(False)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    cal=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['calibration'][2029]
    moment_path=args.root/'input_moments.pt'
    started=time.perf_counter()
    if moment_path.exists():moments=torch.load(moment_path,map_location='cpu',weights_only=True)
    else:
        sums={n:torch.zeros(m.in_features,device='cuda',dtype=torch.float64) for n,m in modules.items()};counts={n:0 for n in modules};hooks=[]
        for name,m in modules.items():
            def pre(module,inputs,n=name):
                x=inputs[0].reshape(-1,module.in_features).contiguous();dx=dq(x)
                sums[n]+=dx.square().sum(0,dtype=torch.float64);counts[n]+=len(x)
            hooks.append(m.register_forward_pre_hook(pre))
        with torch.inference_mode():
            for i,ids in enumerate(cal):
                model.model(ids[None].cuda(),use_cache=False)
                print('Moments',args.root.name,i+1,'/',len(cal),flush=True)
        for h in hooks:h.remove()
        moments={n:(sums[n]/counts[n]).float().cpu() for n in modules}
        assert all(torch.isfinite(x).all() and (x>=0).all() for x in moments.values())
        torch.save(moments,moment_path)
        write(args.root/'input_moments.json',{'samples':len(cal),'length':cal.shape[1],'seed':2029,
            'calibration_sha256':tensor_sha(cal),'definition':'E[Q_A8(X)^2] per input feature from frozen FP16 teacher; diagonal approximation only'})
    manifests={}
    for mode in modes:
        folder=args.root/'banks'/mode;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'manifest.json'
        manifests[mode]=read(path) if path.exists() else {'mode':mode,'format':'E2M1 packed nibbles + logical K32 E8M0 scales',
            'layers':{},'completed':False,'offsets':[0] if mode=='native_rtn' else [0,-1,-2],
            'activation_unchanged':True,'fp8_weight_unchanged':True,'input_moments_sha256':sha(moment_path)}
    with torch.inference_mode():
        for name,m in modules.items():
            for mode in modes:
                manifest=manifests[mode];folder=args.root/'banks'/mode
                if name in manifest['layers']:
                    spec=manifest['layers'][name]
                    assert sha(folder/spec['file'])==spec['sha256'];continue
                qs=[];ss=[];stats=[]
                for start in range(0,m.out_features,1024):
                    q,s,st=quantize(m.weight[start:start+1024],moments[name] if mode=='act_mse_clip' else None,
                                   offsets=(0,) if mode=='native_rtn' else (0,-1,-2))
                    qs.append(q.cpu());ss.append(s.cpu());stats.append(st)
                path=folder/(name+'.pt')
                torch.save({'q4':torch.cat(qs),'scales':torch.cat(ss)},path)
                total={key:sum(x[key] for x in stats) for key in ['groups','weighted_error_before','weighted_error_after','weight_error_before','weight_error_after']}
                total['shift_groups']={str(i):sum(x['shift_groups'].get(str(i),0) for x in stats) for i in [0,1,2]}
                manifest['layers'][name]={'file':path.name,'sha256':sha(path),'rows':m.out_features,'k':m.in_features,'stats':total}
                write(folder/'manifest.json',manifest)
            print('Bank',args.root.name,name,flush=True)
    for mode,manifest in manifests.items():
        assert set(manifest['layers'])==set(modules)
        manifest['completed']=True;manifest['elapsed_s']=time.perf_counter()-started
        write(args.root/'banks'/mode/'manifest.json',manifest)
    write(args.root/'banks_complete.json',{'passed':True,'modes':modes,'layers':len(modules),
        'kernel_sha256':sha(__import__('os').environ['HYGEMM_LIBRARY'])})


if __name__=='__main__':main()
