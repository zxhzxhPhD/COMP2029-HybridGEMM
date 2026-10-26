"""Separate frozen protocols for TTFT/prefill, longer context and graph padding."""
import argparse
import os
from pathlib import Path
import statistics
import time
from final_common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--method',required=True)
    ap.add_argument('--operation',choices=['prefill','long','nonbucket'],required=True);ap.add_argument('--development',action='store_true')
    args=ap.parse_args();stage=args.stage;model='7B';spec=read(stage/'methods.json')[model][args.method];os.environ.update(spec['environment'])
    if not args.development:
        from freeze import verify_frozen
        verify_frozen(stage,model,args.method)
    import torch
    from vllm import LLM,SamplingParams
    from native_run import speed
    torch.set_num_threads(4)
    out=stage/('development' if args.development else 'results')/model/args.method/('workload_'+args.operation);out.mkdir(parents=True,exist_ok=True)
    identity={'method_config_sha256':sha(Path(spec['model_path'])/'config.json'),'source_sha256':sha(__file__),
        'operation':args.operation,'development':args.development,'environment':spec['environment']}
    if (out/'result.json').exists():assert read(out/'result.json')['identity']==identity;return
    kwargs=model_kwargs(spec['model_path'],max_len=2049 if args.operation=='nonbucket' else 4225)
    if args.operation=='prefill':kwargs['disable_log_stats']=False
    llm=LLM(**kwargs)
    tokens=torch.load(stage/'data'/model/'tokens.pt',map_location='cpu',weights_only=True)['validation' if args.development else 'speed']
    rows=[]
    if args.operation=='prefill':
        for context in ([128] if args.development else [128,1024,4096]):
            starts=torch.randint(0,len(tokens)-context,(8,),generator=torch.Generator().manual_seed(3107+context))
            for batch in ([1] if args.development else [1,8]):
                prompts=[{'prompt_token_ids':tokens[i:i+context].tolist()} for i in starts[:batch]]
                params=SamplingParams(temperature=0,max_tokens=1,min_tokens=1,ignore_eos=True,detokenize=False)
                def once():
                    begin=time.perf_counter();outputs=llm.generate(prompts,params,use_tqdm=False);wall=time.perf_counter()-begin
                    values=[]
                    for result in outputs:
                        m=result.metrics;assert m is not None and m.first_token_latency>0
                        assert m.first_token_ts>=m.scheduled_ts>=m.queued_ts
                        values.append({'ttft_ms':m.first_token_latency*1000,'core_prefill_elapsed_ms':(m.first_token_ts-m.scheduled_ts)*1000,
                            'queue_ms':(m.scheduled_ts-m.queued_ts)*1000})
                    return {'request_group_wall_ms':wall*1000,'requests':values,'median_ttft_ms':statistics.median(v['ttft_ms'] for v in values)}
                attempts=[]
                for _ in range(3):
                    once();samples=[once() for _ in range(5)];times=[r['median_ttft_ms'] for r in samples]
                    cv=statistics.pstdev(times)/statistics.mean(times);attempts.append({'samples':samples,'cv':cv})
                    if cv<=.05:break
                rows.append({'batch':batch,'context':context,'attempts':attempts,'stable':cv<=.05,'cv':cv,
                    'median_ttft_ms':statistics.median(times),'median_request_group_wall_ms':statistics.median(r['request_group_wall_ms'] for r in samples)})
    else:
        cfg=read(Path(spec['model_path'])/'config.json')
        kv=4*cfg['num_hidden_layers']*cfg['num_key_value_heads']*(cfg['hidden_size']//cfg['num_attention_heads'])
        batches=[3,12,24,48] if args.operation=='nonbucket' else [1,8,32]
        contexts=[128,1024] if args.operation=='nonbucket' else [2048,4096]
        if args.development:batches=batches[:1];contexts=contexts[:1]
        rows=speed(llm,tokens,batches,contexts,kv_bytes_per_token=kv)
    write(out/'result.json',{'completed':True,'identity':identity,'rows':rows,'llm_options':kwargs,
        'runtime_audit':llm.collective_rpc('final_runtime_audit')[0],
        'scope':'Prefill timing uses engine monotonic timestamps and frontend TTFT, not isolated GPU kernel time; request wall includes the whole submitted group.'})


if __name__=='__main__':main()
