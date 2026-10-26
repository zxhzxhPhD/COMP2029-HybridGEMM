"""Score lm-eval context/continuation requests through actual native vLLM layers."""
import gzip
import json
import math
from pathlib import Path
from final_common import sha,write


def score_tasks(llm,stage,model,out,development=False):
    from vllm import SamplingParams
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0,detokenize=False)
    llm.collective_rpc('task_scoring_begin')
    reports={}
    for name in ['hellaswag','arc_challenge','piqa']:
        suffix='_development' if development else ''
        source=stage/'data'/model/(name+suffix+'.jsonl.gz')
        with gzip.open(source,'rt') as f:requests=[json.loads(line) for line in f]
        target=out/(name+suffix+'_scores.jsonl');scores={}
        if target.exists():
            for line in target.read_text().splitlines():
                row=json.loads(line);scores[row['request_id']]=row
        def run(batch):
            outputs=llm.generate([{'prompt_token_ids':r['tokens']} for r in batch],params,use_tqdm=False)
            values=[]
            for request,result in zip(batch,outputs):
                tokens=request['tokens'];begin=request['target_begin'];lp=result.prompt_logprobs
                assert result.prompt_token_ids==tokens,'Request/output ordering mismatch'
                assert len(lp)==len(tokens)
                selected=[lp[i][tokens[i]] for i in range(begin,len(tokens))]
                value=math.fsum(p.logprob for p in selected);assert math.isfinite(value)
                values.append({'request_id':request['request_id'],'doc_id':request['doc_id'],'choice':request['choice'],
                    'loglikelihood':value,'target_tokens':len(selected),'is_greedy':all(p.rank==1 for p in selected)})
            return values
        pending=sorted((r for r in requests if r['request_id'] not in scores),key=lambda r:len(r['tokens']),reverse=True)
        with target.open('a',buffering=1) as f:
            for start in range(0,len(pending),32):
                for row in run(pending[start:start+32]):
                    scores[row['request_id']]=row;f.write(json.dumps(row)+'\n')
                if start%2048==0:print('Task',name,len(scores),'/',len(requests),flush=True)
        controls=[run([r])[0] for r in requests[:16]]
        delta=max(abs(r['loglikelihood']-scores[r['request_id']]['loglikelihood'])/r['target_tokens'] for r in controls)
        assert delta<=.001,(name,delta)
        reports[name]={'requests':len(requests),'scores_sha256':sha(target),'tokens_sha256':sha(source),
            'submission_group_size':32,'device_max_num_seqs':1,'batch1_control':controls,'max_batching_delta_mean_logprob':delta,'native':True}
    execution=llm.collective_rpc('task_scoring_end')[0]
    assert execution['single_sequence_device_batches'] and execution['execute_calls']>0
    write(out/'task_execution_audit.json',execution)
    write(out/('tasks_development.json' if development else 'tasks_native.json'),reports)
    return reports
