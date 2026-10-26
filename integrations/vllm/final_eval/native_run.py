"""Native vLLM dry run and frozen quality/speed evaluation; no software weights."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import time
from final_common import *


def score_blocks(llm,blocks,group=8):
    from vllm import SamplingParams
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    rows=[]
    for start in range(0,len(blocks),group):
        items=blocks[start:start+group]
        outputs=llm.generate([{'prompt_token_ids':list(map(int,x))} for x in items],params,use_tqdm=False)
        for x,output in zip(items,outputs):
            assert output.prompt_token_ids==list(map(int,x)),'PPL request/output ordering mismatch'
            lp=output.prompt_logprobs;assert lp[0] is None and len(lp)==len(x)
            losses=[-lp[i][int(x[i])].logprob for i in range(1,len(x))]
            assert all(math.isfinite(v) for v in losses)
            rows.append({'nll':math.fsum(losses),'tokens':len(losses)})
        if start%64==0:print('PPL blocks',len(rows),'/',len(blocks),flush=True)
    nll=math.fsum(r['nll'] for r in rows);tokens=sum(r['tokens'] for r in rows)
    return {'blocks':rows,'nll':nll,'tokens':tokens,'mean_nll':nll/tokens,'ppl':math.exp(nll/tokens)}


def speed(llm,tokens,batches,contexts,repeat=5,kv_bytes_per_token=0):
    import torch
    from vllm import SamplingParams
    rows=[];params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
    for ci,context in enumerate(contexts):
        starts=torch.randint(0,len(tokens)-context,(128,),generator=torch.Generator().manual_seed(2029+context))
        for batch in (batches if ci%2==0 else batches[::-1]):
            required_kv=batch*(context+129)*kv_bytes_per_token
            if required_kv>10*1024**3:
                rows.append({'batch':batch,'context':context,'feasible':False,'required_kv_bytes_lower_bound':required_kv,
                    'reason':'Fixed 10 GiB KV budget cannot hold the full requested batch; same exclusion for every method.'})
                continue
            prompts=[{'prompt_token_ids':tokens[i:i+context].tolist()} for i in starts[:batch]]
            def once():
                llm.collective_rpc('formal_begin',args=(batch,));begin=time.perf_counter()
                output=llm.generate(prompts,params,use_tqdm=False)
                wall=time.perf_counter()-begin;row=llm.collective_rpc('formal_end')[0]
                assert row['steps']==row['graph_steps']>=32
                row['request_wall_s']=wall;row['output_sha256']=canonical_sha([r.outputs[0].token_ids for r in output])
                return row
            attempts=[]
            for _ in range(3):
                once();samples=[once() for _ in range(repeat)];values=[r['decode_ms_per_step'] for r in samples]
                cv=statistics.pstdev(values)/statistics.mean(values)
                attempts.append({'samples':samples,'cv':cv,'median_ms_per_step':statistics.median(values)})
                if cv<=.05:break
            rows.append({'batch':batch,'context':context,'feasible':True,'attempts':attempts,'cv':cv,'stable':cv<=.05,
                'median_ms_per_step':statistics.median(values),'tokens_per_second':1000*batch/statistics.median(values)})
            print('Speed',batch,context,statistics.median(values),cv,flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--mode',choices=['dry','tasks_dry','development','final'],default='dry')
    ap.add_argument('--label');ap.add_argument('--operations',default='speed,ppl')
    ap.add_argument('--batches',default='1,2,4,8,16,32,64,128');ap.add_argument('--contexts',default='128,1024')
    ap.add_argument('--max-len',type=int,default=2049);args=ap.parse_args()
    stage=args.stage;spec=read(stage/'methods.json')[args.model][args.method]
    task_only=args.mode=='tasks_dry' or args.operations=='tasks'
    if 'tasks' in args.operations.split(','):assert args.operations=='tasks','Task quality uses its own single-sequence engine'
    os.environ.update(spec['environment'])
    if args.mode=='final':
        from freeze import verify_frozen
        verify_frozen(stage,args.model,args.method)
    import torch
    from vllm import LLM
    torch.set_num_threads(4)
    out=stage/('development' if args.mode!='final' else 'results')/args.model/args.method/(args.label or args.mode)
    out.mkdir(parents=True,exist_ok=True)
    identity={'config_sha256':sha(Path(spec['model_path'])/'config.json'),'environment':spec['environment'],
        'source_sha256':sha(__file__),'mode':args.mode,'operations':args.operations,
        'batches':args.batches,'contexts':args.contexts,'max_len':args.max_len,'llm_options':spec.get('llm_options',{}),
        'task_single_sequence_engine':task_only}
    if (out/'result.json').exists():
        prior=read(out/'result.json');assert prior['identity']==identity
        if prior.get('completed'):return
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity,'Changed partial evaluation identity'
    write(out/'identity.json',identity)
    kwargs={**model_kwargs(spec['model_path'],max_len=args.max_len),**spec.get('llm_options',{})}
    if task_only:
        kwargs['max_num_seqs']=1
        kwargs['compilation_config']={**kwargs['compilation_config'],'cudagraph_capture_sizes':[1],'max_cudagraph_capture_size':1}
    started=time.perf_counter();llm=LLM(**kwargs)
    load_s=time.perf_counter()-started
    if (stage/'data'/args.model/'tokens.pt').exists():
        data=torch.load(stage/'data'/args.model/'tokens.pt',map_location='cpu',weights_only=True)
    else:
        assert args.mode!='final'
        fallback=Q1/args.model/'tokens.pt'
        if not fallback.exists():fallback=stage/args.model/'tokens.pt'
        data={'validation':torch.load(fallback,map_location='cpu',weights_only=True)['validation']}
    result={'identity':identity,'load_and_capture_s':load_s,'completed':False,'operations':{}}
    config=read(Path(spec['model_path'])/'config.json')
    kv_bytes_per_token=4*config['num_hidden_layers']*config.get('num_key_value_heads',config['num_attention_heads'])*config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    if args.mode=='dry':
        blocks=data['validation'][:4*2048].reshape(4,2048)
        grouped=score_blocks(llm,blocks);single=score_blocks(llm,blocks[:2],1)
        delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(grouped['blocks'],single['blocks']))
        assert delta<=.001,(args.model,args.method,delta)
        result['operations']['dry_ppl']={'score':grouped,'batch1_control':single,'max_batching_delta_nll':delta}
        result['operations']['dry_speed']=speed(llm,data['validation'],[1,8],[128],repeat=2)
    elif args.mode=='tasks_dry':
        from tasks_native import score_tasks
        result['operations']['tasks_development']=score_tasks(llm,stage,args.model,out,development=True)
    else:
        operations=args.operations.split(',')
        if 'speed' in operations:
            result['operations']['speed']=speed(llm,data['speed'] if args.mode=='final' else data['validation'],
                list(map(int,args.batches.split(','))),list(map(int,args.contexts.split(','))),kv_bytes_per_token=kv_bytes_per_token)
            write(out/'partial.json',result)
        if 'ppl' in operations:
            splits=['test','c4'] if args.mode=='final' else ['validation']
            for split in splits:
                tokens=data[split];blocks=tokens[:len(tokens)//2048*2048].reshape(-1,2048)
                if split=='c4':blocks=blocks[:256]
                value=score_blocks(llm,blocks);control=score_blocks(llm,blocks[:4],1)
                delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(value['blocks'],control['blocks']))
                value.update(batch1_control=control,max_batching_delta_nll=delta,discarded_tail_tokens=len(tokens)%2048,
                    source_tokens=len(tokens),scored_input_tokens=len(blocks)*2048,
                    unscored_source_tokens=len(tokens)-len(blocks)*2048,excluded_initial_targets=len(blocks))
                assert delta<=.001,(split,delta)
                result['operations'][split]=value;write(out/'partial.json',result)
        if 'tasks' in operations:
            assert args.mode=='final'
            from tasks_native import score_tasks
            result['operations']['tasks']=score_tasks(llm,stage,args.model,out)
    audit=llm.collective_rpc('hygemm_model_audit')[0]
    if config.get('quantization_config') and spec.get('audit_mixed',True):assert len(audit)==config['num_hidden_layers']*4
    result['audit']=audit
    result['runtime_audit']=llm.collective_rpc('final_runtime_audit')[0]
    if spec.get('budget'):
        budget=llm.collective_rpc('quality_budget')[0]
        assert budget['persistent_quantized_storage_bytes']==spec['budget']['persistent_quantized_bytes']
        result['budget']=budget
    result['completed']=True;write(out/'result.json',result)
    print('Completed',args.model,args.method,args.mode,flush=True)


if __name__=='__main__':main()
