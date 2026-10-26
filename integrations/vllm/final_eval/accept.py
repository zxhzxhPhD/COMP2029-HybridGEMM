"""Final completeness, provenance, stability and paired-numerics acceptance."""
import argparse
import json
from pathlib import Path
import time
from final_common import *
from freeze import verify_frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--full-hashes',action='store_true');args=ap.parse_args();s=args.stage
    manifest=verify_frozen(s,full=args.full_hashes);jobs=read(s/'final_jobs.json');missing=[];failed=[]
    for job in jobs:
        file=s/'queues/final'/ (job['id']+'.json')
        if not file.exists():missing.append(job['id']);continue
        value=read(file)
        if not value['passed'] or value['job']!=job:failed.append(job['id'])
    unstable=[];speed_cells=0;infeasible=[];ppl=[];task_entries=[];engineering=[];full_steps=0
    workload_cells=0;workload_full_steps=0;max_cv=0.0
    sampling={k:{'cells':0,'groups':0,'single_group_cells':0,'maximum_attempt_cv':0.0} for k in ['ordinary','workload']}
    retried_cells=[]
    def record_attempts(kind,row,identity):
        value=sampling[kind];attempts=row['attempts'];value['cells']+=1;value['groups']+=len(attempts)
        value['single_group_cells']+=len(attempts)==1
        value['maximum_attempt_cv']=max(value['maximum_attempt_cv'],*(a['cv'] for a in attempts))
        if len(attempts)>1:retried_cells.append({'cell':identity,'attempt_cv':[a['cv'] for a in attempts],'stable':row['stable']})
    for file in (s/'results').glob('*/*/*/result.json'):
        r=read(file);assert r['completed'],file
        model,method,label=file.relative_to(s/'results').parts[:3]
        for row in r.get('operations',{}).get('speed',[]):
            if not row.get('feasible',True):infeasible.append([model,method,label,row['batch'],row['context']]);continue
            speed_cells+=1
            max_cv=max(max_cv,row['cv'])
            record_attempts('ordinary',row,[model,method,label,row['batch'],row['context']])
            if not row['stable'] or row['cv']>.05:unstable.append([model,method,label,row['batch'],row['context']])
            for attempt in row['attempts']:
                assert len(attempt['samples'])==5
                for v in attempt['samples']:
                    assert v['steps']==v['graph_steps']>=32
                    assert v['first_pure_decode_graph_descriptor']['num_tokens']>=row['batch']
                    full_steps+=v['graph_steps']
        for split in ['test','c4']:
            if split in r.get('operations',{}):
                value=r['operations'][split];assert value['max_batching_delta_nll']<=.001
                assert all(b['tokens']==2047 for b in value['blocks'])
                if split=='c4':assert len(value['blocks'])==256
                ppl.append([model,method,label,split,len(value['blocks']),value['tokens']])
        if 'tasks' in r.get('operations',{}):
            audit=read(file.parent/'task_execution_audit.json');assert audit['single_sequence_device_batches']
            metrics=read(file.parent/'task_metrics.json')
            for task,count in [('hellaswag',10042),('arc_challenge',1172),('piqa',1838)]:
                assert metrics[task]['documents']==count
                assert r['operations']['tasks'][task]['max_batching_delta_mean_logprob']<=.001
                task_entries.append([model,method,task,count])
        if label.startswith('workload_'):
            operation=label.removeprefix('workload_')
            batches,contexts={'prefill':([1,8],[128,1024,4096]),'long':([1,8,32],[2048,4096]),
                'nonbucket':([3,12,24,48],[128,1024])}[operation]
            assert {(v['batch'],v['context']) for v in r['rows']}=={(b,c) for b in batches for c in contexts},file
        for row in r.get('rows',[]):
            if not row.get('feasible',True):continue
            workload_cells+=1;max_cv=max(max_cv,row['cv'])
            record_attempts('workload',row,[model,method,label,row['batch'],row['context']])
            if not row['stable'] or row['cv']>.05:unstable.append([model,method,label,row['batch'],row['context']])
            for attempt in row['attempts']:
                assert len(attempt['samples'])==5
                if label!='workload_prefill':
                    for v in attempt['samples']:
                        assert v['steps']==v['graph_steps']>=32
                        descriptor=v['first_pure_decode_graph_descriptor']
                        expected_bucket=min(b for b in BATCHES if b>=row['batch'])
                        assert descriptor['num_tokens']==expected_bucket and descriptor['mode']=='FULL'
                        workload_full_steps+=v['graph_steps']
    for file in (s/'comparisons').glob('*/*.json'):
        r=read(file)
        if 'passed' in r:engineering.append({'path':str(file),'passed':r['passed'],'targets':r['targets']})
    expected_main={(m,k) for m in MODELS for k in METHODS}
    actual_main={(f.relative_to(s/'results').parts[0],f.relative_to(s/'results').parts[1]) for f in (s/'results').glob('*/*/main/result.json')}
    assert actual_main<=expected_main
    missing_main=sorted(expected_main-actual_main)
    # The fifth model has exactly one infeasible point per main/repeat method.
    assert all(row[0]=='Mistral-7B' and row[-2:]==[128,1024] for row in infeasible),infeasible
    # Coverage audit of the already frozen v1 matrix, not new numerical gates.
    coverage={'ordinary_decode_cells':{'expected':988,'actual':speed_cells},
        'workload_cells':{'expected':100,'actual':workload_cells},
        'ppl_evaluations':{'expected':146,'actual':len(ppl)},
        'task_evaluations':{'expected':87,'actual':len(task_entries)},
        'engineering_comparisons':{'expected':9,'actual':len(engineering)},
        'capacity_infeasible_main_repeat_cells':{'expected':8,'actual':len(infeasible)}}
    complete_coverage=all(v['expected']==v['actual'] for v in coverage.values())
    result={'passed':not(missing or failed or unstable or missing_main) and all(r['passed'] for r in engineering),
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'freeze_manifest_sha256':sha(s/'freeze/v1/manifest.json'),
        'expected_jobs':len(jobs),'missing_jobs':missing,'failed_jobs':failed,'missing_main':missing_main,
        'speed_cells':speed_cells,'unstable_cells':unstable,'infeasible_cells':infeasible,'verified_timed_full_steps':full_steps,
        'ppl_evaluations':ppl,'task_evaluations':task_entries,'engineering_gates':engineering,
        'coverage':coverage,'workload_cells':workload_cells,'max_cv':max_cv,
        'max_cv_scope':'Accepted group, or the final attempted group for an unstable cell; see sampling_audit for all attempts.',
        'sampling_audit':sampling,'retried_cells':retried_cells,
        'verified_workload_timed_full_steps':workload_full_steps,
        'full_input_hash_recheck':args.full_hashes,'mix_original_binary_unchanged':manifest['mix_device_sha256']==sha(MIX)}
    result['passed']=result['passed'] and complete_coverage
    result['execution_complete']=not(missing or failed or missing_main) and complete_coverage
    result['ordinary_decode_stability_passed']=not any(not r[2].startswith('workload_') for r in unstable)
    result['workload_stability_passed']=not any(r[2].startswith('workload_') for r in unstable)
    result['numerical_checks_passed']=all(r['passed'] for r in engineering)
    write(s/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['ppl_evaluations','task_evaluations','engineering_gates']},indent=2))
    assert result['passed'],'See acceptance.json for exact incomplete/failed cells'


if __name__=='__main__':main()
