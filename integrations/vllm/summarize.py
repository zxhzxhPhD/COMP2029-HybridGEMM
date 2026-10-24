"""Summarize smoke evidence without presenting setup time as decode speed."""
import argparse
import json
from pathlib import Path


def prompt_delta(a,b):
    diffs=[]
    for x,y in zip(a['outputs'],b['outputs']):
        assert x['prompt_token_ids']==y['prompt_token_ids']
        for i,token in enumerate(x['prompt_token_ids']):
            left,right=x['prompt_logprobs'][i],y['prompt_logprobs'][i]
            if left and right:diffs.append(abs(left[str(token)]-right[str(token)]))
    return {'positions':len(diffs),'max_abs_logprob_delta':max(diffs,default=0),
            'mean_abs_logprob_delta':sum(diffs)/max(len(diffs),1),
            'same_generated_tokens':[x['token_ids']==y['token_ids'] for x,y in zip(a['outputs'],b['outputs'])]}


ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
smokes={p.stem:json.loads(p.read_text()) for p in args.out.glob('smoke_*.json')}
assert all(x['passed'] for x in smokes.values())
result={'passed_runs':list(smokes),'graph_vs_eager':{},'graph_no_compile_vs_eager':{},'checkpoint_vs_online':None,
        'limits':'Smoke only: no speed, perplexity, task accuracy or equal-quality claim.'}
for kind in ['fp16','mixllm','hygemm']:
    if 'smoke_'+kind+'_graph' in smokes:
        result['graph_vs_eager'][kind]=prompt_delta(smokes['smoke_'+kind],smokes['smoke_'+kind+'_graph'])
    if 'smoke_'+kind+'_graph_no_compile' in smokes:
        check=prompt_delta(smokes['smoke_'+kind],smokes['smoke_'+kind+'_graph_no_compile'])
        assert all(check['same_generated_tokens']) and check['max_abs_logprob_delta']<1e-4, check
        result['graph_no_compile_vs_eager'][kind]=check
if 'smoke_mixllm_online' in smokes:
    check=prompt_delta(smokes['smoke_mixllm'],smokes['smoke_mixllm_online'])
    assert all(check['same_generated_tokens']) and check['max_abs_logprob_delta']<1e-4, check
    result['checkpoint_vs_online']=check
result['layers']={name:{'quantized_modules':len(x['audit'][0]),
    'mixed_modules':sum(bool(r['high'] and r['low']) for r in x['audit'][0]),
    'high_channels':sum(r['high'] for r in x['audit'][0]),
    'low_channels':sum(r['low'] for r in x['audit'][0])} for name,x in smokes.items()}
(args.out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
