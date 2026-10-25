"""Actual unchanged native path: persistent bytes, validation NLL and decode."""
import argparse
import math
import os
from pathlib import Path
import statistics
import torch
from common import read,write,sha,tensor_sha,aggregate,configure

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--method',default='original')
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--decode',action='store_true')
    args=ap.parse_args();configure()
    from vllm import LLM,SamplingParams
    view=args.root/'views'/args.method
    config=read(view/'config.json')
    llm=LLM(model=str(view),dtype='float16',tensor_parallel_size=1,max_model_len=2049,
        max_num_seqs=128,max_num_batched_tokens=2048,enable_prefix_caching=False,
        gpu_memory_utilization=.93,kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.quality_worker.QualityWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    budget=llm.collective_rpc('quality_budget')[0]
    expected=read(args.root/'budget_expected.json')
    assert budget['persistent_quantized_storage_bytes']==expected['persistent_quantized_bytes']
    assert budget['library_sha256']==read(args.root.parent/'protocol.json')['native_library_sha256']
    assert len(budget['layers'])==config['num_hidden_layers']*4
    payload=sum(4*(r['out']+r['high'])*r['k'] for r in budget['layers'])
    assert payload==expected['payload_bits']
    budget.update(expected);budget['passed']=True
    write(args.root/'budget'/f'{args.method}.json',budget)
    if args.audit_only:return
    validation=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['validation']
    blocks=validation[:validation.numel()//2048*2048].reshape(-1,2048)
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    def score(data,group):
        result=[]
        for start in range(0,len(data),group):
            outs=llm.generate([{'prompt_token_ids':x.tolist()} for x in data[start:start+group]],params,use_tqdm=False)
            for x,output in zip(data[start:start+group],outs):
                lp=output.prompt_logprobs
                assert len(lp)==len(x) and lp[0] is None
                losses=[-lp[i][int(x[i])].logprob for i in range(1,len(x))]
                assert all(math.isfinite(v) for v in losses)
                result.append({'nll':math.fsum(losses),'tokens':len(losses)})
        return result
    values=score(blocks,8);control=score(blocks[:4],1)
    delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(values,control))
    assert delta<=.001,delta
    report={'method':args.method,'budget_passed':True,'validation':aggregate(values),
        'validation_token_sha256':tensor_sha(validation),'discarded_tail_tokens':validation.numel()%2048,
        'batch1_control':control,'max_batching_delta_nll':delta,'library_sha256':budget['library_sha256'],
        'selection_sha256':sha(view/'selection.json') if (view/'selection.json').exists() else sha(args.root/'original_selection.json'),
        'plans_sha256':sha(args.root/'plans.json'),'graph':'FULL','inductor':False,'decode':[]}
    if args.decode:
        gen=torch.Generator().manual_seed(2157)
        offsets=torch.randint(0,validation.numel()-128,(128,),generator=gen)
        dp=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
        for batch in [1,8,64]:
            prompts=[{'prompt_token_ids':validation[i:i+128].tolist()} for i in offsets[:batch]]
            def once():
                llm.collective_rpc('formal_begin',args=(batch,))
                out=llm.generate(prompts,dp,use_tqdm=False)
                row=llm.collective_rpc('formal_end')[0]
                assert row['steps']==row['graph_steps']>=32
                assert all(len(x.outputs[0].token_ids)==129 for x in out)
                return row
            attempts=[]
            for _ in range(3):
                once();samples=[once() for _ in range(5)]
                times=[x['decode_ms_per_step'] for x in samples]
                cv=statistics.pstdev(times)/statistics.mean(times)
                attempts.append({'samples':samples,'cv':cv,'median_ms_per_step':statistics.median(times)})
                if cv<=.05:break
            report['decode'].append({'batch':batch,'context':128,'attempts':attempts,'stable':cv<=.05,
                                     'median_ms_per_step':statistics.median(times),'cv':cv})
    report['audit']=llm.collective_rpc('hygemm_model_audit')[0]
    report['completed']=True
    write(args.root/'native'/f'{args.method}.json',report)
    print('Complete native',args.root.name,args.method,report['validation']['ppl'],flush=True)


if __name__=='__main__':main()
