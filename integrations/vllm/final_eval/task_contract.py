"""CPU-only checks of pinned task request/metric contracts using training docs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from lm_eval.tasks import TaskManager
from prepare_tasks import load_task
from final_common import *

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
manifest=read(stage/'data/tasks/manifest.json');manager=TaskManager();checks={}
for name,spec in manifest['tasks'].items():
    task=load_task(manager,name,spec['config_override']);groups=defaultdict(list)
    for line in (stage/'data/tasks'/(name+'_development.jsonl')).read_text().splitlines():
        row=json.loads(line);assert row['request_id']==canonical_sha([row['task'],row['doc_id'],row['choice'],row['context'],row['continuation']])
        groups[row['doc_id']].append(row)
    for doc_id,rows in groups.items():
        rows.sort(key=lambda r:r['choice']);doc=rows[0]['doc'];target=int(task.doc_to_target(doc))
        choices=task.doc_to_choice(doc);assert len(choices)==len(rows)
        assert [r['choice'] for r in rows]==list(range(len(rows)))
        correct=[(-.001 if i==target else -1000.,False) for i in range(len(rows))]
        incorrect=[(-.001 if i==(target+1)%len(rows) else -1000.,False) for i in range(len(rows))]
        assert all(float(v)==1 for k,v in task.process_results(doc,correct).items() if k in ['acc','acc_norm'])
        assert all(float(v)==0 for k,v in task.process_results(doc,incorrect).items() if k in ['acc','acc_norm'])
    checks[name]={'training_documents':len(groups),'request_id_contract':True,'official_metric_perfect_and_wrong_choice_checks':True,
        'dataset_revision':spec['revision'],'native_model_scoring':False}
write(stage/'data/tasks/contract_checks.json',{'passed':True,'checks':checks,
    'official_test_suite_note':'The pip environment lacks pytest; the upstream check_integrity entry failed before running tests. This explicit contract check is not labeled as passing the upstream test suite.'})
print('Pinned task contracts passed',checks,flush=True)
