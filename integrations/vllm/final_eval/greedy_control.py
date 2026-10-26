"""Short free generation with a common enqueue barrier and recorded top logits."""
import argparse
import math
import os
from pathlib import Path
from final_common import *


def generate(llm,prompts):
    from vllm import SamplingParams
    core=llm.llm_engine.engine_core;core.call_utility('pause_scheduler','keep',False)
    original=llm._run_engine
    def resume(*args,**kwargs):
        core.call_utility('resume_scheduler');llm._run_engine=original
        return original(*args,**kwargs)
    llm._run_engine=resume
    try:
        rows=llm.generate([{'prompt_token_ids':p} for p in prompts],
            SamplingParams(temperature=0,max_tokens=32,min_tokens=32,ignore_eos=True,detokenize=False,logprobs=5),use_tqdm=False)
    finally:llm._run_engine=original
    assert [r.prompt_token_ids for r in rows]==prompts
    return {'tokens':[r.outputs[0].token_ids for r in rows],
        'logprobs':[[{str(k):v.logprob for k,v in step.items()} for step in r.outputs[0].logprobs] for r in rows],
        'enqueue_barrier':True,'topk':5}


def compare(a,b):
    assert a['prompts']==b['prompts'];left=a['runs'][0];right=b['runs'][0];errors=[];divergences=[]
    for request,(x,y) in enumerate(zip(left['tokens'],right['tokens'])):
        for step,(u,v) in enumerate(zip(x,y)):
            p=left['logprobs'][request][step];q=right['logprobs'][request][step]
            for token in [u] if u==v else [u,v]:
                assert str(token) in p and str(token) in q,('Divergent token outside the other top five',request,step)
                errors.append(abs(p[str(token)]-q[str(token)]))
            if u!=v:
                divergences.append({'request':request,'first_divergence':step,'old_token':u,'new_token':v,
                    'old_top1_gap':p[str(u)]-p[str(v)],'new_top1_gap':q[str(v)]-q[str(u)]});break
    repeat=all(r['tokens']==data['runs'][0]['tokens'] for data in [a,b] for r in data['runs'])
    value={'greedy_tokens_equal':left['tokens']==right['tokens'],'repeat_tokens_equal':repeat,
        'common_history_logprob_samples':len(errors),'mean_abs_logprob':math.fsum(errors)/len(errors),
        'max_abs_logprob':max(errors),'first_divergences':divergences}
    value['passed']=repeat and value['mean_abs_logprob']<=.002 and value['max_abs_logprob']<=.05
    return value


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--method');ap.add_argument('--compare',action='store_true');args=ap.parse_args();s=args.stage
    if args.compare:
        a=read(s/'development'/args.model/'K0-I/greedy_control/result.json')
        b=read(s/'development'/args.model/'Hy-I/greedy_control/result.json');r=compare(a,b)
        write(s/'development'/args.model/'controlled_greedy_comparison.json',r);print(r);assert r['passed']
    else:
        spec=read(s/'methods.json')[args.model][args.method];os.environ.update(spec['environment'])
        from vllm import LLM
        import torch
        torch.set_num_threads(4)
        prompts=read(s/'development'/args.model/'FP16/development_decode/quality.json')['greedy_prompts']
        llm=LLM(**model_kwargs(spec['model_path']));runs=[generate(llm,prompts) for _ in range(2)]
        write(s/'development'/args.model/args.method/'greedy_control/result.json',{'completed':True,'prompts':prompts,'runs':runs,
            'source_sha256':sha(__file__),'model_config_sha256':sha(Path(spec['model_path'])/'config.json'),
            'environment':spec['environment'],'development_validation_only':True})
