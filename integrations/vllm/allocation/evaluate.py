"""Native WT2 validation with measured payload and inherited per-layer plans."""
import argparse
import math
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
    ap.add_argument('--view',required=True);ap.add_argument('--decode',action='store_true')
    args=ap.parse_args();configure()
    from vllm import LLM,SamplingParams
    view=args.root/'views'/args.view;spec=read(view/'view.json');protocol=read(args.root.parent/'protocol.json')
    assert sha(os.environ['HYGEMM_LIBRARY'])==protocol['kernel_sha256']
    llm=LLM(model=str(view),dtype='float16',tensor_parallel_size=1,max_model_len=2049,max_num_seqs=128,
        max_num_batched_tokens=2048,enable_prefix_caching=False,gpu_memory_utilization=.93,
        kv_cache_memory_bytes=10*1024**3,seed=2029,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='hygemm_vllm.quality_worker.QualityWorkerExtension',
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128],
                            'max_cudagraph_capture_size':128})
    budget=llm.collective_rpc('quality_budget')[0]
    weights=sum(r['tensors']['w4']['storage_bytes']+r['tensors']['w8']['storage_bytes'] for r in budget['layers'])
    assert weights==spec['budget']['weight_bytes']
    assert budget['persistent_quantized_storage_bytes']==spec['budget']['persistent_quantized_bytes']
    actual={r['name']:(r['out'],r['k'],r['high']) for r in budget['layers']}
    assert actual=={r['layer']:(r['out'],r['k'],r['high']) for r in spec['layers']}
    payload_bpw=weights*8/spec['weights']
    assert payload_bpw<5.1 and abs(payload_bpw-spec['payload_bits_per_weight'])<1e-12
    assert 0<=spec['target_payload_bits_per_weight']-payload_bpw<.001
    budget.update(passed=True,actual_payload_bits_per_weight=payload_bpw)
    write(args.root/'budgets'/f'{args.view}.json',budget)
    tokens=torch.load(args.root/'tokens.pt',map_location='cpu',weights_only=True)['validation']
    blocks=tokens[:tokens.numel()//2048*2048].reshape(-1,2048)
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    def score(data,group):
        rows=[]
        for start in range(0,len(data),group):
            outputs=llm.generate([{'prompt_token_ids':x.tolist()} for x in data[start:start+group]],params,use_tqdm=False)
            for x,result in zip(data[start:start+group],outputs):
                lp=result.prompt_logprobs;assert len(lp)==len(x) and lp[0] is None
                losses=[-lp[i][int(x[i])].logprob for i in range(1,len(x))]
                assert all(math.isfinite(v) for v in losses)
                rows.append({'nll':math.fsum(losses),'tokens':len(losses)})
        return rows
    values=score(blocks,8);control=score(blocks[:4],1)
    delta=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(values,control));assert delta<=.001
    report={'view':args.view,'spec':spec,'completed':False,'validation':aggregate(values),
        'validation_token_sha256':tensor_sha(tokens),'discarded_tail_tokens':tokens.numel()%2048,
        'max_batching_delta_nll':delta,'batch1_control':control,'decode':[],
        'kernel_sha256':budget['library_sha256'],'config_sha256':sha(view/'config.json'),
        'selection_sha256':sha(view/'selection.json'),'reference_plans_sha256':sha(args.root/'reference_plans.json'),
        'graph':'FULL','inductor':False,'actual_payload_bits_per_weight':payload_bpw}
    if args.view=='q1_alias':
        q1=Path(protocol['q1'])/args.root.name
        old=read(q1/'native/original.json')['validation']
        difference=max(abs(a['nll']-b['nll'])/2047 for a,b in zip(values,old['blocks']))
        assert difference<=1e-4,('Loader control changed NLL',difference)
        report['q1_alias_max_block_nll_delta']=difference
    if args.decode:
        gen=torch.Generator().manual_seed(2157)
        offsets=torch.randint(0,tokens.numel()-128,(128,),generator=gen)
        params_d=SamplingParams(temperature=0,max_tokens=129,min_tokens=129,ignore_eos=True,detokenize=False)
        for batch in [1,8,64]:
            prompts=[{'prompt_token_ids':tokens[i:i+128].tolist()} for i in offsets[:batch]]
            def once():
                llm.collective_rpc('formal_begin',args=(batch,));outputs=llm.generate(prompts,params_d,use_tqdm=False)
                row=llm.collective_rpc('formal_end')[0]
                assert row['steps']==row['graph_steps']>=32
                assert all(len(x.outputs[0].token_ids)==129 for x in outputs)
                return row
            attempts=[]
            for _ in range(3):
                once();samples=[once() for _ in range(5)];times=[x['decode_ms_per_step'] for x in samples]
                cv=statistics.pstdev(times)/statistics.mean(times)
                attempts.append({'samples':samples,'cv':cv,'median_ms_per_step':statistics.median(times)})
                if cv<=.05:break
            report['decode'].append({'batch':batch,'context':128,'attempts':attempts,'cv':cv,'stable':cv<=.05,
                                    'median_ms_per_step':statistics.median(times)})
    report['audit']=llm.collective_rpc('hygemm_model_audit')[0]
    reference={r['layer']:r for r in read(args.root/'reference_cases.json')['layers']}
    plans=read(args.root/'reference_plans.json')['hygemm']
    for layer in report['audit']:
        row=reference[layer['name']]
        candidates=sorted([r for r in plans if (r['out'],r['k'],r['high'])==(row['out'],row['k'],row['high'])],key=lambda r:r['tokens'])
        for t,used in layer['plans_used'].items():
            plan=next(r for r in candidates if r['tokens']>=int(t))
            assert used==[plan['tokens'],plan['low_id'],plan['high_id'],plan['policy']=='concurrent']
    report['completed']=True;write(args.root/'native'/f'{args.view}.json',report)
    print('Complete',args.root.name,args.view,'PPL',report['validation']['ppl'],'bits',payload_bpw,flush=True)


if __name__=='__main__':main()
