"""Paired targets and distributions; no comparison of divergent continuations."""
import argparse
import math
from pathlib import Path
import torch
from final_common import read,write,sha


def compare(reference,candidate):
    a=read(reference);b=read(candidate);base={(r['batch'],r['context']):r for r in a['records']}
    deltas=[];signed=[];nll=[];ref_nll=[];distributions=[]
    for row in b['records']:
        old=base[(row['batch'],row['context'])]
        assert row['targets']==old['targets'] and row['prefill_schedule_sha256']==old['prefill_schedule_sha256']
        for i,(x,y) in enumerate(zip(row['logprobs'],old['logprobs'])):
            indices=sorted(set(row['full_target_indices'][i])&set(old['full_target_indices'][i]))[:32]
            assert len(indices)==32
            for j in indices:
                deltas.append(abs(x[j]-y[j]));signed.append(y[j]-x[j]);nll.append(-x[j]);ref_nll.append(-y[j])
        left=torch.load(old['distribution_file'],map_location='cpu',weights_only=True)
        right=torch.load(row['distribution_file'],map_location='cpu',weights_only=True)
        assert [r['target_index'] for r in left]==[r['target_index'] for r in right]
        for p,q in zip(left,right):
            lp=p['logprob'].double();lq=q['logprob'].double();assert torch.isfinite(lp).all() and torch.isfinite(lq).all()
            prob=lp.exp();other=lq.exp()
            distributions.append({'batch':row['batch'],'context':row['context'],'target_index':p['target_index'],
                'vocab_size':len(lp),'kl_reference_to_candidate':float((prob*(lp-lq)).sum()),
                'total_variation':float((prob-other).abs().sum()/2),'top1_agree':int(lp.argmax()==lq.argmax()),
                'max_abs_logprob':float((lp-lq).abs().max())})
    return {'reference':str(reference),'candidate':str(candidate),'targets':len(deltas),
        'target_mean_nll':math.fsum(nll)/len(nll),'reference_target_mean_nll':math.fsum(ref_nll)/len(ref_nll),
        'mean_nll_delta':math.fsum(signed)/len(signed),'mean_abs_target_logprob':math.fsum(deltas)/len(deltas),
        'max_abs_target_logprob':max(deltas),'distribution_samples':len(distributions),
        'mean_kl':sum(r['kl_reference_to_candidate'] for r in distributions)/len(distributions),
        'top1_agreement':sum(r['top1_agree'] for r in distributions)/len(distributions),'distributions':distributions,
        'greedy_tokens_equal':a['greedy_prompts']==b['greedy_prompts'] and a['greedy_tokens']==b['greedy_tokens']}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--engineering-gate',action='store_true');args=ap.parse_args()
    value=compare(args.reference,args.candidate)
    if args.engineering_gate:
        value['passed']=value['mean_abs_target_logprob']<=.002 and value['max_abs_target_logprob']<=.05
    write(args.out,value)
    if args.engineering_gate:assert value['passed'],value
    print({k:v for k,v in value.items() if k!='distributions'})
