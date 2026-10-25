"""Stage-two FULL Graph model gates on frozen development-validation prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
from prepare import read,write,sha,runtime_sha

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--kind',choices=['hygemm','mixllm','fp16','hy_pre_i'],default='hygemm')
    ap.add_argument('--plans',type=Path);ap.add_argument('--batches',default='1,2,4,8,16,32,64,128')
    ap.add_argument('--contexts',default='128,1024');ap.add_argument('--repeat',type=int,default=5)
    args=ap.parse_args();torch.set_num_threads(4)
    folder=args.stage/args.model;out=folder/args.label;out.mkdir(parents=True,exist_ok=True)
    original=folder/'models'/args.kind;model=out/'model';model.mkdir(exist_ok=True)
    for file in original.iterdir():
        if file.is_file() and file.name!='config.json' and not (model/file.name).exists():(model/file.name).symlink_to(file.resolve())
    cfg=read(original/'config.json')
    if args.plans:
        assert args.kind in ['hygemm','mixllm']
        cfg['quantization_config']['decode_plans' if args.kind=='hygemm' else 'tuning_plan']=str(args.plans.resolve())
    write(model/'config.json',cfg)
    identity={'config_sha256':sha(model/'config.json'),'plans_sha256':sha(args.plans) if args.plans else None,
        'kernel_sha256':sha(os.environ['HYGEMM_LIBRARY']) if args.kind in ['hygemm','hy_pre_i'] else None,
        'mix_sha256':sha(os.environ['MIXLLM_LIBRARY']) if args.kind=='mixllm' else None,
        'pack_mode':os.environ.get('HYGEMM_PACK_MODE','0'),'source_sha256':sha(__file__),
        'runtime_source_sha256':runtime_sha(),
        'fused_silu':os.environ.get('HYGEMM_FUSE_SILU','0'),
        'row_order':os.environ.get('HYGEMM_ROW_ORDER','none'),
        'fused_gateup_max_tokens':os.environ.get('HYGEMM_FUSED_GATEUP','0'),
        'fused_library_sha256':sha(os.environ['HYGEMM_FUSED_LIBRARY']) if int(os.environ.get('HYGEMM_FUSED_GATEUP','0')) else None,
        'producer_math':os.environ.get('HYGEMM_SILU_MATH','0'),
        'producer_sha256':sha(os.environ['HYGEMM_PRODUCER_LIBRARY']) if os.environ.get('HYGEMM_FUSE_SILU')=='1' else None,
        'tokens_sha256':sha(folder/'calibration/tokens.pt'),'batches':args.batches,'contexts':args.contexts,'repeat':args.repeat}
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity,'Stale model result'
    write(out/'identity.json',identity)
    if (out/'result.json').exists():
        previous=read(out/'result.json')
        if previous.get('completed'):return
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(model),dtype='float16',tensor_parallel_size=1,max_model_len=2049,max_num_seqs=128,
        max_num_batched_tokens=2048,enable_prefix_caching=False,gpu_memory_utilization=.93,
        kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.quality_worker.QualityWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    tokens=torch.load(folder/'calibration/tokens.pt',map_location='cpu',weights_only=True)['validation']
    rows=[]
    for context in map(int,args.contexts.split(',')):
        gen=torch.Generator().manual_seed(2029+context)
        offsets=torch.randint(0,tokens.numel()-context,(128,),generator=gen)
        params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
        batches=list(map(int,args.batches.split(',')))
        if context==1024:batches.reverse()
        for batch in batches:
            prompts=[{'prompt_token_ids':tokens[i:i+context].tolist()} for i in offsets[:batch]]
            def once():
                llm.collective_rpc('formal_begin',args=(batch,));output=llm.generate(prompts,params,use_tqdm=False)
                rec=llm.collective_rpc('formal_end')[0];assert rec['steps']==rec['graph_steps']>=32
                rec['output_sha256']=hashlib.sha256(json.dumps([r.outputs[0].token_ids for r in output]).encode()).hexdigest()
                return rec
            attempts=[]
            for _ in range(3):
                once();samples=[once() for _ in range(args.repeat)];times=[r['decode_ms_per_step'] for r in samples]
                cv=statistics.pstdev(times)/statistics.mean(times)
                attempts.append({'samples':samples,'cv':cv,'median_ms_per_step':statistics.median(times)})
                if cv<=.05:break
            assert cv<=.05
            rows.append({'batch':batch,'context':context,'attempts':attempts,'cv':cv,'stable':True,
                'median_ms_per_step':statistics.median(times),'output_sha256':samples[0]['output_sha256']})
            write(out/'partial.json',rows);print(args.label,batch,context,statistics.median(times),cv,flush=True)
    audit=llm.collective_rpc('hygemm_model_audit')[0]
    if args.kind!='fp16':assert len(audit)==cfg['num_hidden_layers']*4 and all(r['plans_used'] for r in audit)
    budget=llm.collective_rpc('quality_budget')[0] if args.kind=='hygemm' else None
    if budget:assert budget['persistent_quantized_storage_bytes']==read(args.stage/'inputs.json')[args.model]['budget']['persistent_quantized_bytes']
    write(out/'result.json',{'completed':True,'identity':identity,'cells':rows,'audit':audit,'budget':budget,
        'graph':'FULL','inductor':False,'validation_prompts':True})


if __name__=='__main__':main()
