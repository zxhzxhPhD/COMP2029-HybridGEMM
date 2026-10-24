"""Small fixed-input vLLM prefill/decode smoke; not a performance benchmark."""
import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--kind',choices=['fp16','mixllm','hygemm'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--graph',action='store_true')
    ap.add_argument('--no-compile',action='store_true',help='Capture full graphs while retaining eager arithmetic kernels')
    args=ap.parse_args()
    if args.no_compile and not args.graph:
        ap.error('--no-compile is a graph-mode control')
    import torch
    from vllm import LLM,SamplingParams
    start=time.perf_counter()
    cc={'cudagraph_capture_sizes':[1,2,4],'max_cudagraph_capture_size':4} if args.graph else None
    if args.no_compile:cc.update(mode=0,cudagraph_mode='FULL')
    llm=LLM(model=args.model,dtype='float16',tensor_parallel_size=1,
            max_model_len=256,max_num_seqs=4,max_num_batched_tokens=256,
            gpu_memory_utilization=.35,seed=20261024,
            worker_extension_cls='hygemm_vllm.worker.AuditWorkerExtension',
            enforce_eager=not args.graph,enable_prefix_caching=False,
            compilation_config=cc)
    prompts=['The capital of France is','Write a short sentence about a cat.']
    params=SamplingParams(temperature=0,max_tokens=16,ignore_eos=True,logprobs=1,prompt_logprobs=1)
    outputs=llm.generate(prompts,params,use_tqdm=False)
    again=llm.generate(prompts,params,use_tqdm=False)
    assert [x.outputs[0].token_ids for x in outputs]==[x.outputs[0].token_ids for x in again]
    rows=[]
    for result in outputs:
        o=result.outputs[0]
        assert len(o.token_ids)==16 and o.text
        vals=[p.logprob for record in o.logprobs for p in record.values()]
        assert all(math.isfinite(x) for x in vals)
        pvals=[p.logprob for record in result.prompt_logprobs if record for p in record.values()]
        assert all(math.isfinite(x) for x in pvals)
        rows.append({'prompt':result.prompt,'token_ids':o.token_ids,'text':o.text,
                     'cumulative_logprob':o.cumulative_logprob,'prompt_token_ids':result.prompt_token_ids,
                     'prompt_logprobs':[{str(k):v.logprob for k,v in r.items()} if r else None for r in result.prompt_logprobs]})
    audits=llm.collective_rpc('hygemm_model_audit')
    if args.kind!='fp16':
        assert len(audits)==1 and len(audits[0])==96, f'Expected 24x4 fused projection modules, got {list(map(len,audits))}'
        assert all(x['kind']==args.kind for x in audits[0])
        # Explicitly verify small decode calls and larger prefill/profile calls.
        assert all(any(int(t)<=4 for t in x['calls_by_tokens']) for x in audits[0])
        assert all(any(int(t)>4 for t in x['calls_by_tokens']) for x in audits[0])
        assert any(x['high'] and x['low'] for x in audits[0])
    report={'kind':args.kind,'graph':args.graph,'compile_disabled':args.no_compile,'model':args.model,
            'vllm':metadata.version('vllm'),'torch':torch.__version__,'cuda':torch.version.cuda,
            'elapsed_setup_and_smoke_s':time.perf_counter()-start,'deterministic_repeat':True,
            'outputs':rows,'audit':audits,'passed':True}
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'kind':args.kind,'graph':args.graph,'passed':True,'outputs':[r['text'] for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
