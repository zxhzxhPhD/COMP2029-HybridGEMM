"""Teacher-forced target logprobs from actual cached FULL Graph decode."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from prepare import read,write,sha

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--kind',choices=['hygemm','mixllm'],default='hygemm')
    ap.add_argument('--plans',type=Path);ap.add_argument('--compare',type=Path)
    ap.add_argument('--batches',default='1,2,4,8,16,32,64,128');args=ap.parse_args()
    folder=args.stage/args.model;out=folder/args.label;out.mkdir(parents=True,exist_ok=True)
    original=folder/'models'/args.kind;model=out/'model';model.mkdir(exist_ok=True)
    for file in original.iterdir():
        if file.is_file() and file.name!='config.json' and not (model/file.name).exists():(model/file.name).symlink_to(file.resolve())
    cfg=read(original/'config.json')
    if args.plans:cfg['quantization_config']['decode_plans' if args.kind=='hygemm' else 'tuning_plan']=str(args.plans.resolve())
    write(model/'config.json',cfg)
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    llm=LLM(model=str(model),dtype='float16',tensor_parallel_size=1,max_model_len=2049,max_num_seqs=128,
        max_num_batched_tokens=2048,enable_prefix_caching=False,gpu_memory_utilization=.93,
        kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.performance_worker.PerformanceWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    tokens=torch.load(folder/'calibration/tokens.pt',map_location='cpu',weights_only=True)['validation']
    records=[]
    for context in [128,1024]:
        gen=torch.Generator().manual_seed(2029+context);offsets=torch.randint(0,tokens.numel()-context-129,(128,),generator=gen)
        for b in map(int,args.batches.split(',')):
            prompts=[tokens[i:i+context].tolist() for i in offsets[:b]]
            targets=[tokens[i+context:i+context+129].tolist() for i in offsets[:b]]
            llm.collective_rpc('teacher_begin',args=(prompts,targets,b))
            params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False,logprobs=0)
            # Submit the whole quality batch before scheduling starts. Otherwise
            # the asynchronous engine can begin after one versus two arrivals,
            # changing prefill chunking and numerical KV history independently
            # of the candidate kernels. This barrier is not used for speed tests.
            core=llm.llm_engine.engine_core
            core.call_utility('pause_scheduler','keep',False)
            original_run=llm._run_engine
            def resume_then_run(*args_,**kwargs_):
                core.call_utility('resume_scheduler')
                llm._run_engine=original_run
                return original_run(*args_,**kwargs_)
            llm._run_engine=resume_then_run
            try:outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],params,use_tqdm=False)
            finally:llm._run_engine=original_run
            audit=llm.collective_rpc('teacher_end')[0];assert audit['steps']==audit['graph_steps']>=32,audit
            values=[]
            for target,result in zip(targets,outputs):
                actual=result.outputs[0];assert actual.token_ids==target
                values.append([item[token].logprob for item,token in zip(actual.logprobs,target)])
            full_indices=[[r['target_index'] for r in audit['positions'] if r['prompt_index']==i and r['full_batch_decode']]
                          for i in range(b)]
            assert all(len(indices)>=32 and min(indices)>=1 for indices in full_indices)
            records.append({'context':context,'batch':b,'logprobs':values,'targets':targets,'full_target_indices':full_indices,
                'prefill_schedule_sha256':hashlib.sha256(json.dumps(audit['prefill_schedule']).encode()).hexdigest(),'audit':audit})
            write(out/'partial.json',records);print('Teacher',b,context,flush=True)
    report={'completed':True,'kind':args.kind,'records':records,'kernel_sha256':sha(os.environ['HYGEMM_LIBRARY' if args.kind=='hygemm' else 'MIXLLM_LIBRARY']),
            'pack_mode':os.environ.get('HYGEMM_PACK_MODE','0'),'config_sha256':sha(model/'config.json'),
            'raw_target_logprobs':True,'teacher_forced':True,'graph':'FULL','inductor':False,
            'budget':llm.collective_rpc('quality_budget')[0] if args.kind=='hygemm' else None,
            'paired_prefill':True,'audit':llm.collective_rpc('hygemm_model_audit')[0]}
    if args.compare:
        old=read(args.compare);deltas=[];signed=[]
        for a,b in zip(records,old['records']):
            assert (a['batch'],a['context'],a['targets'])==(b['batch'],b['context'],b['targets'])
            assert a['prefill_schedule_sha256']==b['prefill_schedule_sha256'],'Unpaired prefill scheduling'
            # Exclude the initial prefill prediction, evaluate the 32 cached decode targets.
            for i,(x,y) in enumerate(zip(a['logprobs'],b['logprobs'])):
                indices=sorted(set(a['full_target_indices'][i]) & set(b['full_target_indices'][i]))[:32]
                assert len(indices)>=16,('Insufficient common FULL decode targets',len(indices))
                for index in indices:
                    u,v=x[index],y[index];deltas.append(abs(u-v));signed.append(v-u)
        summary={'targets':len(deltas),'max_abs_logprob':max(deltas),'mean_abs_logprob':math.fsum(deltas)/len(deltas),
            'mean_nll_delta':math.fsum(signed)/len(signed)}
        summary['passed']=summary['max_abs_logprob']<=.05 and summary['mean_abs_logprob']<=.002
        report['comparison']=summary
    write(out/'quality.json',report)
    if args.compare:assert report['comparison']['passed'],report['comparison']


if __name__=='__main__':main()
