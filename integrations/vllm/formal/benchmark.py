"""One model/path process: full-graph decode matrix and complete WT2 PPL."""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
import torch

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')

BATCHES=[1,2,4,8,16,32,64,128]
CONTEXTS=[128,1024]


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--kind',choices=['fp16','mixllm','hygemm'],required=True)
    ap.add_argument('--batches',default=','.join(map(str,BATCHES)))
    ap.add_argument('--contexts',default=','.join(map(str,CONTEXTS)))
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--skip-ppl',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();batches=list(map(int,args.batches.split(',')));contexts=list(map(int,args.contexts.split(',')))
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    model=args.out/'models'/args.kind
    cfg=json.loads((model/'config.json').read_text())
    test=torch.load(args.out/'calibration/tokens.pt',map_location='cpu',weights_only=True)['test']
    target=args.out/f'result_{args.kind}.json'
    previous=json.loads(target.read_text()) if args.resume and target.exists() else None
    if previous:
        assert previous['protocol']['batches']==batches and previous['protocol']['contexts']==contexts and previous['protocol']['repeats']==args.repeats
        if previous.get('completed'):print('Already completed',args.kind);return
    start=time.perf_counter()
    llm=LLM(model=str(model),dtype='float16',tensor_parallel_size=1,max_model_len=2049,
        max_num_seqs=max(batches),max_num_batched_tokens=2048,enable_prefix_caching=False,
        gpu_memory_utilization=.93,kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},
        worker_extension_cls='hygemm_vllm.worker.AuditWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':batches,'max_cudagraph_capture_size':max(batches)})
    report={'kind':args.kind,'model':str(model),'protocol':{'batches':batches,'contexts':contexts,'new_tokens':129,
        'warmups':1,'repeats':args.repeats,'max_attempts':3,'cv_gate':.05,'prefix_cache':False,'inductor':False,
        'graph':'FULL','attention':'FlashAttention 2 on SM120','dtype':'FP16','kv_dtype':'FP16',
        'decode_metric':'CUDA event span from execute_model start to sampling end across pure fixed-batch decode steps, including inter-step idle gaps',
        'excluded':'prefill and mixed prefill/decode steps; loading/capture; endpoint postprocessing',
        'ppl':'complete nonoverlapping 2048-token WT2 test blocks; 2047 targets/block; no BOS; group8',
        'kv_cache_bytes':10*1024**3},
        'setup_s':time.perf_counter()-start,'decode':[]}
    plan_hash=hashlib.sha256((args.out/'plans.json').read_bytes()).hexdigest() if args.kind!='fp16' else None
    report['plans_sha256']=plan_hash
    if previous:
        assert previous['plans_sha256']==plan_hash
        report=previous
    params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
    def run_decode(b,context,trial):
        gen=torch.Generator().manual_seed(2029+context)
        offsets=torch.randint(0,test.numel()-context,(max(batches),),generator=gen)
        prompts=[{'prompt_token_ids':test[i:i+context].tolist()} for i in offsets[:b]]
        llm.collective_rpc('formal_begin',args=(b,))
        t=time.perf_counter();outs=llm.generate(prompts,params,use_tqdm=False);wall=time.perf_counter()-t
        values=llm.collective_rpc('formal_end')[0]
        assert all(len(r.outputs[0].token_ids)==129 for r in outs)
        assert values['steps']>=32,values
        values.update(wall_request_ms=wall*1000,trial=trial,
            output_sha256=hashlib.sha256(json.dumps([r.outputs[0].token_ids for r in outs]).encode()).hexdigest())
        return values
    for context in contexts:
        for b in (batches if context==contexts[0] else list(reversed(batches))):
            if any(r['batch']==b and r['context']==context for r in report['decode']):continue
            row={'batch':b,'context':context,'attempts':[]}
            for attempt in range(3):
                run_decode(b,context,'warmup')
                samples=[run_decode(b,context,i) for i in range(args.repeats)]
                times=[x['decode_ms_per_step'] for x in samples]
                cv=statistics.pstdev(times)/statistics.mean(times)
                rec={'samples':samples,'median_ms_per_step':statistics.median(times),'cv':cv,'stable':cv<=.05}
                row['attempts'].append(rec)
                if rec['stable']:break
            row.update({k:rec[k] for k in ['median_ms_per_step','cv','stable']})
            row['tokens_per_second']=b*1000/row['median_ms_per_step']
            report['decode'].append(row);write(target,report)
            print('decode',args.kind,b,context,'ms',row['median_ms_per_step'],'CV',cv,'stable',row['stable'],flush=True)
    if not args.skip_ppl:
        blocks=test[:test.numel()//2048*2048].reshape(-1,2048)
        pplparams=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
        def score(part,group):
            values=[]
            for start in range(0,len(part),group):
                outs=llm.generate([{'prompt_token_ids':x.tolist()} for x in part[start:start+group]],pplparams,use_tqdm=False)
                for x,result in zip(part[start:start+group],outs):
                    lp=result.prompt_logprobs
                    assert len(lp)==2048 and lp[0] is None
                    losses=[-lp[i][int(x[i])].logprob for i in range(1,2048)]
                    assert all(math.isfinite(v) for v in losses)
                    values.append({'nll':math.fsum(losses),'tokens':len(losses)})
                print('ppl',args.kind,len(values),'/',len(part),flush=True)
            return values
        t=time.perf_counter();full=score(blocks,min(8,max(batches)));control=score(blocks[:8],1)
        nll=math.fsum(x['nll'] for x in full);count=sum(x['tokens'] for x in full)
        delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(full,control))
        assert delta<=.001,('PPL batching mismatch',delta)
        report['ppl']={'value':math.exp(nll/count),'mean_nll':nll/count,'nll':nll,'scored_tokens':count,'blocks':full,
            'batch1_control':control,'max_batching_delta_nll_per_token':delta,'elapsed_s':time.perf_counter()-t,
            'test_token_sha256':hashlib.sha256(test.numpy().tobytes()).hexdigest(),'discarded_tail_tokens':test.numel()%2048}
        write(target,report)
    audit=llm.collective_rpc('hygemm_model_audit')[0]
    if args.kind!='fp16':
        assert len(audit)==cfg['num_hidden_layers']*4
        assert all(layer['plans_used'] for layer in audit)
    report['audit']=audit;report['completed']=True;write(target,report)
    print('COMPLETE',args.kind,flush=True)


if __name__=='__main__':main()
