"""Apply the pinned lm-eval task metrics to saved native loglikelihoods."""
import argparse
from collections import defaultdict
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
from lm_eval.tasks import TaskManager
from prepare_tasks import load_task
from final_common import read,write,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--scores',type=Path,required=True)
    ap.add_argument('--development',action='store_true');args=ap.parse_args()
    manifest=read(args.stage/'data/tasks/manifest.json');manager=TaskManager();summary={}
    assert importlib.metadata.version('lm_eval')==manifest['lm_eval_version']
    import lm_eval
    package=Path(lm_eval.__file__).parent
    for relative,digest in read(args.stage/'data/tasks/source_hashes.json').items():
        assert sha(package/relative)==digest,('Changed task source',relative)
    for name,spec in manifest['tasks'].items():
        task=load_task(manager,name,spec['config_override']);suffix='_development' if args.development else ''
        assert sha(inspect.getsourcefile(task.__class__))==spec['task_implementation_sha256']
        evaluation_split=spec['task_config'].get('test_split') or spec['task_config'].get('validation_split')
        assert evaluation_split in ['test','validation']
        source=args.stage/'data/tasks'/(name+suffix+'.jsonl');scores_file=args.scores/(name+suffix+'_scores.jsonl')
        with scores_file.open() as f:scores={r['request_id']:r for r in map(json.loads,f)}
        groups=defaultdict(list)
        with source.open() as f:
            for row in map(json.loads,f):groups[row['doc_id']].append(row)
        per_metric=defaultdict(list);sample_path=args.scores/(name+suffix+'_samples.jsonl')
        with sample_path.open('w') as f:
            for doc_id,requests in sorted(groups.items()):
                requests.sort(key=lambda r:r['choice']);responses=[]
                for row in requests:
                    value=scores.pop(row['request_id'])
                    responses.append((value['loglikelihood'],value['is_greedy']))
                result={k:float(v) for k,v in task.process_results(requests[0]['doc'],responses).items()}
                for key,value in result.items():per_metric[key].append(value)
                f.write(json.dumps({'doc_id':doc_id,'doc':requests[0]['doc'],'arguments':[[r['context'],r['continuation']] for r in requests],
                    'responses':responses,'metrics':result},ensure_ascii=False)+'\n')
        assert not scores,'Unexpected scored requests'
        metrics={key:float(task.aggregation()[key](values)) for key,values in per_metric.items()}
        rng=np.random.default_rng(3071);intervals={}
        for key,values in per_metric.items():
            arr=np.asarray(values);boot=np.array([arr[rng.integers(0,len(arr),len(arr))].mean() for _ in range(1000)])
            intervals[key]=[float(x) for x in np.quantile(boot,[.025,.975])]
        summary[name]={'documents':len(groups),'metrics':metrics,'bootstrap_95':intervals,
            'sample_sha256':sha(sample_path),'scores_sha256':sha(scores_file),'requests_sha256':sha(source),
            'dataset_revision':spec['revision'],'lm_eval_version':manifest['lm_eval_version'],'native':True,
            'split':'training dry-run' if args.development else evaluation_split}
    write(args.scores/('task_metrics_development.json' if args.development else 'task_metrics.json'),summary)
    print(json.dumps({k:v['metrics'] for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
