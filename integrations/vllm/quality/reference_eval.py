"""Unchanged FP16/MixLLM validation references on the same frozen serving stack."""
import argparse
import math
import os
from pathlib import Path
import torch
from common import read,write,sha,tensor_sha,aggregate,configure

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--kind',choices=['fp16','mixllm'],required=True);args=ap.parse_args();configure()
    from vllm import LLM,SamplingParams
    cfg=read(args.root/'views/original/config.json')
    if args.kind=='fp16':cfg.pop('quantization_config')
    else:
        cfg['quantization_config']['quant_method']='mixllm'
        assert sha(os.environ['MIXLLM_LIBRARY'])=='1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da'
    view=args.root/'references'/args.kind;view.mkdir(parents=True,exist_ok=True)
    for file in (args.root/'views/original').iterdir():
        if file.name not in ['config.json','selection.json'] and file.is_file() and not (view/file.name).exists():
            (view/file.name).symlink_to(file.resolve())
    write(view/'config.json',cfg)
    llm=LLM(model=str(view),dtype='float16',tensor_parallel_size=1,max_model_len=2049,
        max_num_seqs=128,max_num_batched_tokens=2048,enable_prefix_caching=False,
        gpu_memory_utilization=.93,kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.worker.AuditWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    tokens=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['validation']
    blocks=tokens[:tokens.numel()//2048*2048].reshape(-1,2048)
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    def score(part,group):
        rows=[]
        for start in range(0,len(part),group):
            outputs=llm.generate([{'prompt_token_ids':x.tolist()} for x in part[start:start+group]],params,use_tqdm=False)
            for ids,result in zip(part[start:start+group],outputs):
                lp=result.prompt_logprobs
                assert len(lp)==len(ids) and lp[0] is None
                losses=[-lp[i][int(ids[i])].logprob for i in range(1,len(ids))]
                assert all(math.isfinite(x) for x in losses)
                rows.append({'nll':math.fsum(losses),'tokens':len(losses)})
        return rows
    values=score(blocks,8);control=score(blocks[:4],1)
    delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(values,control))
    assert delta<=.001
    write(args.root/'native'/f'reference_{args.kind}.json',{'completed':True,'method':args.kind,
        'validation':aggregate(values),'validation_token_sha256':tensor_sha(tokens),
        'batch1_control':control,'max_batching_delta_nll':delta,'graph':'FULL','inductor':False,
        'audit':llm.collective_rpc('hygemm_model_audit')[0],
        'mixllm_library_sha256':sha(os.environ['MIXLLM_LIBRARY']) if args.kind=='mixllm' else None})
    print('Reference',args.root.name,args.kind,aggregate(values)['ppl'],flush=True)


if __name__=='__main__':main()
