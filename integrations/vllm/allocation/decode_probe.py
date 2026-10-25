"""Bounded quality-stage decode probe; exact plans, FULL Graph, no Inductor."""
import argparse
import hashlib
import os
from pathlib import Path
import statistics
import torch
from alloc_utils import *

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('VLLM_PLUGINS','hygemm_quantization')
os.environ.setdefault('VLLM_NO_USAGE_STATS','1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--view',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plans',type=Path);ap.add_argument('--compare',type=Path)
    args=ap.parse_args();configure();args.out.mkdir(parents=True,exist_ok=True)
    from vllm import LLM,SamplingParams
    original=args.root/'views'/args.view;model=args.out/'model';model.mkdir(exist_ok=True)
    for p in original.iterdir():
        if p.is_file() and p.name!='config.json' and not (model/p.name).exists():(model/p.name).symlink_to(p.resolve())
    cfg=read(original/'config.json')
    if args.plans:cfg['quantization_config']['decode_plans']=str(args.plans.resolve())
    write(model/'config.json',cfg)
    llm=LLM(model=str(model),dtype='float16',tensor_parallel_size=1,max_model_len=2049,max_num_seqs=128,
        max_num_batched_tokens=2048,enable_prefix_caching=False,gpu_memory_utilization=.93,
        kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.quality_worker.QualityWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    budget=llm.collective_rpc('quality_budget')[0];spec=read(original/'view.json')
    assert budget['persistent_quantized_storage_bytes']==spec['budget']['persistent_quantized_bytes']
    tokens=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['validation']
    rows=[]
    for context in [128,1024]:
        gen=torch.Generator().manual_seed(2029+context)
        offsets=torch.randint(0,tokens.numel()-context,(128,),generator=gen)
        params=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
        for batch in [1,8,64,128]:
            prompts=[{'prompt_token_ids':tokens[i:i+context].tolist()} for i in offsets[:batch]]
            def once():
                llm.collective_rpc('formal_begin',args=(batch,));output=llm.generate(prompts,params,use_tqdm=False)
                row=llm.collective_rpc('formal_end')[0]
                assert row['steps']==row['graph_steps']>=32
                ids=[r.outputs[0].token_ids for r in output];assert all(len(x)==129 for x in ids)
                row['output_sha256']=hashlib.sha256(str(ids).encode()).hexdigest()
                return row
            attempts=[]
            for _ in range(3):
                once();samples=[once() for _ in range(5)];times=[s['decode_ms_per_step'] for s in samples]
                cv=statistics.pstdev(times)/statistics.mean(times)
                attempts.append({'samples':samples,'cv':cv,'median_ms_per_step':statistics.median(times)})
                if cv<=.05:break
            assert cv<=.05
            rows.append({'batch':batch,'context':context,'attempts':attempts,'cv':cv,'stable':True,
                         'median_ms_per_step':statistics.median(times),'output_sha256':samples[0]['output_sha256'],
                         'repeated_outputs_identical':len({s['output_sha256'] for s in samples})==1})
            write(args.out/'partial.json',{'cells':rows})
    audit=llm.collective_rpc('hygemm_model_audit')[0]
    reference={r['layer']:r for r in read(args.root/'reference_cases.json')['layers']}
    base=read(args.root/'reference_plans.json')['hygemm'];over=read(args.plans)['hygemm'] if args.plans else None
    for layer in audit:
        old=reference[layer['name']]
        for t,used in layer['plans_used'].items():
            actual=next(r for r in spec['layers'] if r['layer']==layer['name'])
            table=over if over is not None and int(t)<=128 else base
            high=actual['high'] if table is over else old['high']
            candidates=sorted([r for r in table if (r['out'],r['k'],r['high'])==(old['out'],old['k'],high)],key=lambda r:r['tokens'])
            plan=next(r for r in candidates if r['tokens']>=int(t))
            assert used==[plan['tokens'],plan['low_id'],plan['high_id'],plan['policy']=='concurrent']
    report={'completed':True,'view':args.view,'cells':rows,'audit':audit,'graph':'FULL','inductor':False,
            'budget':budget,'selection_sha256':sha(original/'selection.json'),'plans_sha256':sha(args.plans) if args.plans else None,
            'model_config_sha256':sha(model/'config.json'),'source_view_config_sha256':sha(original/'config.json'),
            'reference_plans_sha256':sha(args.root/'reference_plans.json'),
            'bank_manifest_sha256':sha(spec['bank']) if spec['bank'] else None,
            'kernel_sha256':sha(os.environ['HYGEMM_LIBRARY']),'validation_prompts':True}
    if args.compare:
        old=read(args.compare);assert old['selection_sha256']==report['selection_sha256']
        assert old['bank_manifest_sha256']==report['bank_manifest_sha256']
        report['same_selection_decode_outputs_match']=all(a['output_sha256']==b['output_sha256'] for a,b in zip(rows,old['cells']))
    write(args.out/'decode.json',report)


if __name__=='__main__':main()
