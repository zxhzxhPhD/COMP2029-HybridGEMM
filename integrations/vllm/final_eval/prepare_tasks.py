"""Export pinned lm-eval requests and task sources, without evaluating a model."""
import argparse
from collections import defaultdict
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi
from lm_eval.tasks import TaskManager,get_task_dict
from final_common import read,write,sha,canonical_sha

DATASETS={'hellaswag':'Rowan/hellaswag','arc_challenge':'allenai/ai2_arc','piqa':'baber/piqa'}
COUNTS={'hellaswag':10042,'arc_challenge':1172,'piqa':1838}


def load_task(manager,name,overrides):
    # In lm-eval 0.4.13, an inline dict is a complete task definition. Resolve
    # the registered YAML entry first so includes/functions survive overrides.
    entry=manager._entry(name);assert entry is not None
    return manager._factory.build(entry,overrides=overrides,registry=manager._index)


def export(task,path,limit=None):
    task.set_fewshot_seed(2029);task.build_all_requests(limit=limit,cache_requests=False,apply_chat_template=False)
    counts=defaultdict(int);records=[]
    for instance in task.instances:
        assert instance.request_type=='loglikelihood' and instance.repeats==1
        context,continuation=instance.args
        assert isinstance(context,str) and isinstance(continuation,str) and continuation
        row={'task':task.config.task,'doc_id':instance.doc_id,'choice':instance.idx,
             'context':context,'continuation':continuation,'doc':instance.doc}
        row['request_id']=canonical_sha([row['task'],row['doc_id'],row['choice'],context,continuation])
        records.append(row);counts[instance.doc_id]+=1
    assert len({r['request_id'] for r in records})==len(records)
    assert all(2<=n<=5 for n in counts.values())
    with path.open('w') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return {'requests':len(records),'documents':len(counts),'sha256':sha(path),'path':str(path)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
    out=args.stage/'data/tasks';out.mkdir(parents=True,exist_ok=True);manager=TaskManager();api=HfApi()
    pins=read(out/'revisions.json') if (out/'revisions.json').exists() else {name:api.dataset_info(repo).sha for name,repo in DATASETS.items()}
    write(out/'revisions.json',pins);manifest={}
    for name,repo in DATASETS.items():
        config={'task':name,'num_fewshot':0,'dataset_kwargs':{'revision':pins[name],'cache_dir':str(out/'dataset_cache')}}
        task=load_task(manager,name,config)
        assert task.OUTPUT_TYPE=='multiple_choice'
        main=export(task,out/(name+'.jsonl'));assert main['documents']==COUNTS[name],main
        main['dataset_fingerprint']=getattr(task.eval_docs,'_fingerprint',None)
        def encode_config(value):
            if callable(value):return {'callable':value.__qualname__,'source_sha256':sha(inspect.getsourcefile(value))}
            raise TypeError(type(value).__name__)
        main['task_config']=json.loads(json.dumps(task.dump_config(),default=encode_config))
        main['config_override']=config;main['revision']=pins[name];main['dataset']=repo
        main['task_implementation_sha256']=sha(inspect.getsourcefile(task.__class__))
        # Only training-split examples will be scored before the final freeze.
        task.config.test_split=None;task.config.validation_split=task.config.training_split
        assert task.config.validation_split=='train'
        main['development']=export(task,out/(name+'_development.jsonl'),limit=16)
        manifest[name]=main;write(out/'manifest.json',{'lm_eval_version':importlib.metadata.version('lm_eval'),
            'tasks':manifest,'zero_shot':True,'chat_template':False,'model_scores_computed':False})
        print('Prepared task',name,main['documents'],main['requests'],flush=True)
    import lm_eval
    package=Path(lm_eval.__file__).parent;source=out/'lm_eval_source'
    for rel in ['api/task.py','api/instance.py','api/model.py','tasks/hellaswag','tasks/arc','tasks/piqa']:
        src=package/rel;dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
        else:shutil.copy2(src,dst)
    write(out/'source_hashes.json',{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml']})


if __name__=='__main__':main()
