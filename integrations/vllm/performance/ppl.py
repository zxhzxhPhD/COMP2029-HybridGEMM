"""Full development-validation prefill PPL; separate from cached-decode gates."""
import argparse
import math
import os
from pathlib import Path
import torch
from prepare import read,write,sha
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--model-path',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    llm=LLM(model=str(args.model_path),dtype='float16',tensor_parallel_size=1,max_model_len=2049,max_num_seqs=128,
        max_num_batched_tokens=2048,enable_prefix_caching=False,gpu_memory_utilization=.93,
        kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.quality_worker.QualityWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    tokens=torch.load(args.stage/args.model/'calibration/tokens.pt',map_location='cpu',weights_only=True)['validation']
    blocks=tokens[:len(tokens)//2048*2048].reshape(-1,2048)
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    records=[]
    for offset in range(0,len(blocks),8):
        outputs=llm.generate([{'prompt_token_ids':x.tolist()} for x in blocks[offset:offset+8]],params,use_tqdm=False)
        for x,result in zip(blocks[offset:offset+8],outputs):
            loss=[-result.prompt_logprobs[i][int(x[i])].logprob for i in range(1,2048)]
            assert all(math.isfinite(v) for v in loss)
            records.append({'nll':math.fsum(loss),'tokens':2047})
        print('Validation',len(records),len(blocks),flush=True)
    nll=math.fsum(r['nll'] for r in records);count=sum(r['tokens'] for r in records);value=math.exp(nll/count)
    stage1=read(args.stage/'stage1_handoff.json')['models'][args.model]['validation_ppl']
    report={'completed':True,'ppl':value,'mean_nll':nll/count,'tokens':count,'blocks':records,
        'stage1_ppl':stage1,'delta_mean_nll':math.log(value/stage1),'passed':abs(math.log(value/stage1))<=1e-5,
        'split':'validation','final_test_untouched':True,'model_config_sha256':sha(args.model_path/'config.json'),
        'budget':llm.collective_rpc('quality_budget')[0],'audit':llm.collective_rpc('hygemm_model_audit')[0]}
    write(args.out,report);assert report['passed'],report['delta_mean_nll']


if __name__=='__main__':main()
