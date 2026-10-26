"""Paired cached-decode NLL and small full-vocabulary distribution samples."""
import argparse
import os
from pathlib import Path
from final_common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--development',action='store_true')
    ap.add_argument('--batches',default='1,8,32,64,128');ap.add_argument('--contexts',default='128,1024')
    ap.add_argument('--label',default='decode_quality');args=ap.parse_args()
    stage=args.stage;spec=read(stage/'methods.json')[args.model][args.method];os.environ.update(spec['environment'])
    if not args.development:
        from freeze import verify_frozen
        verify_frozen(stage,args.model,args.method)
    import torch
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    out=stage/('development' if args.development else 'results')/args.model/args.method/args.label;out.mkdir(parents=True,exist_ok=True)
    identity={'config_sha256':sha(Path(spec['model_path'])/'config.json'),'source_sha256':sha(__file__),
        'tokens_file_sha256':sha(stage/'data'/args.model/'tokens.pt'),'environment':spec['environment'],
        'development':args.development,'batches':args.batches,'contexts':args.contexts}
    if (out/'quality.json').exists():assert read(out/'quality.json')['identity']==identity;return
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    write(out/'identity.json',identity)
    llm=LLM(**{**model_kwargs(spec['model_path']),**spec.get('llm_options',{})})
    tokens=torch.load(stage/'data'/args.model/'tokens.pt',map_location='cpu',weights_only=True)['validation' if args.development else 'c4']
    records=[]
    for context in map(int,args.contexts.split(',')):
        starts=torch.randint(0,len(tokens)-context-129,(128,),generator=torch.Generator().manual_seed(2029+context))
        for batch in map(int,args.batches.split(',')):
            prompts=[tokens[i:i+context].tolist() for i in starts[:batch]]
            targets=[tokens[i+context:i+context+129].tolist() for i in starts[:batch]]
            distribution=out/'distributions'/f'b{batch}_c{context}.pt'
            llm.collective_rpc('teacher_begin',args=(prompts,targets,batch,str(distribution)))
            core=llm.llm_engine.engine_core;core.call_utility('pause_scheduler','keep',False)
            original=llm._run_engine
            def resume(*args_,**kwargs_):
                core.call_utility('resume_scheduler');llm._run_engine=original
                return original(*args_,**kwargs_)
            llm._run_engine=resume
            params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False,logprobs=0)
            try:outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],params,use_tqdm=False)
            finally:llm._run_engine=original
            audit=llm.collective_rpc('teacher_end')[0];assert audit['steps']==audit['graph_steps']>=32
            values=[]
            for target,result in zip(targets,outputs):
                output=result.outputs[0];assert output.token_ids==target
                values.append([item[token].logprob for item,token in zip(output.logprobs,target)])
            indices=[[r['target_index'] for r in audit['positions'] if r['prompt_index']==i and r['full_batch_decode']] for i in range(batch)]
            assert all(len(v)>=32 for v in indices)
            records.append({'batch':batch,'context':context,'targets':targets,'logprobs':values,'full_target_indices':indices,
                'prefill_schedule_sha256':canonical_sha(audit['prefill_schedule']),'distribution_file':str(distribution),
                'distribution_sha256':sha(distribution),'audit':audit})
            write(out/'partial.json',records);print('Decode quality',args.model,args.method,batch,context,flush=True)
    # Unforced generation is a separate engineering check, not cross-method NLL.
    prompts=[tokens[i:i+128].tolist() for i in starts[:4]]
    from greedy_control import generate
    generated=generate(llm,prompts)
    write(out/'quality.json',{'completed':True,'identity':identity,'records':records,
        'greedy_prompts':prompts,'greedy_tokens':generated['tokens'],'greedy_control':generated,
        'runtime_audit':llm.collective_rpc('final_runtime_audit')[0],'native':True,'teacher_forced':True})


if __name__=='__main__':main()
