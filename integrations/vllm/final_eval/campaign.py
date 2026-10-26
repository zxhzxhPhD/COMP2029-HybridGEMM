"""Declare every final cell before freeze, independent of final measurements."""
import argparse
from pathlib import Path
from final_common import *

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();s=args.stage
    assert not (s/'freeze/current.json').exists()
    methods=read(s/'methods.json');jobs=[]
    def add(identifier,script,arguments,python=PYTHON,**extra):
        jobs.append({'id':identifier,'command':[python,str(HERE/script),*map(str,arguments)],**extra})
    def native(model,method,label,operations='speed,ppl',batches='1,2,4,8,16,32,64,128'):
        add(f'{model}_{method}_{label}','native_run.py',['--stage',s,'--model',model,'--method',method,'--mode','final','--label',label,'--operations',operations,'--batches',batches])
    def decode(model,method,batches='1,8,32,64,128'):
        add(f'{model}_{method}_decode','decode_quality.py',['--stage',s,'--model',model,'--method',method,'--batches',batches])
    def compare(model,reference,method,engineering=False):
        extra=['--engineering-gate'] if engineering else []
        add(f'{model}_{reference}_vs_{method}_decode','compare_decode.py',[
            '--reference',s/'results'/model/reference/'decode_quality/quality.json',
            '--candidate',s/'results'/model/method/'decode_quality/quality.json',
            '--out',s/'comparisons'/model/f'{reference}_vs_{method}.json',*extra])
    # Rotate method order by model. No quality-dependent order or early stopping.
    for i,model in enumerate(MODELS):
        order=METHODS[i:]+METHODS[:i]
        for method in order:native(model,method,'main')
    # Independent second timing group for claims of small performance differences.
    for model in MODELS:
        for method in ['Hy-F-Q','Mix-I','Hy-I']:native(model,method,'repeat','speed')
    for model in ['3B','7B']:
        for method in ['K0-I','K0-F']:native(model,method,'ablation',batches='1,8,32,64,128')
        native(model,'Hy-I-Q','quantizer',batches='1,8,64')
        for method in methods[model]:
            if method.startswith('B-') or method=='W8A8':native(model,method,'budget',batches='1,8,64')
            if method.startswith('R-'):native(model,method,'robustness',batches='1,8,64')
        for method in METHODS+['K0-I','K0-F-Q','Hy-I-Q']:decode(model,method)
        for method in METHODS[1:]+['Hy-I-Q']:compare(model,'FP16',method)
        compare(model,'K0-I','Hy-I',True);compare(model,'K0-F-Q','Hy-F-Q',True)
    for method in ['A-no-pack','A-no-producer','A-serial','A-no-light-high']:
        native('7B',method,'mechanism','speed','1,8,32,64')
        decode('7B',method,'1,8,32,64');compare('7B','Hy-F-Q',method,True)
    for method in METHODS:
        for op in ['prefill','long','nonbucket']:
            add(f'7B_{method}_{op}','workloads.py',['--stage',s,'--method',method,'--operation',op])
    for method in ['K0-F-Q','Hy-F-Q']:
        add('7B_'+method+'_nonbucket_quality','decode_quality.py',['--stage',s,'--model','7B','--method',method,
            '--batches','3,12,24,48','--label','nonbucket_quality'])
    add('7B_nonbucket_engineering','compare_decode.py',[
        '--reference',s/'results/7B/K0-F-Q/nonbucket_quality/quality.json',
        '--candidate',s/'results/7B/Hy-F-Q/nonbucket_quality/quality.json',
        '--out',s/'comparisons/7B/nonbucket_engineering.json','--engineering-gate'])
    for method in ['Startup-FP16','Startup-Mix-I','Startup-Hy-F-Q']:native('7B',method,'startup','startup')
    selected_fp8=[];fp8_status={}
    for model in ['0.5B','7B']:
        path=s/'development'/model/'FP8-native'/'extra_dry/result.json'
        record=s/'queues/extra_dry'/(model+'_FP8-native.json')
        fp8_status[model]={'development_job':read(record),'log':str(record.with_suffix('.log'))}
        if path.exists() and read(path).get('completed'):
            selected_fp8.append(model);native(model,'FP8-native','optional')
    # Full task splits, separate one-sequence scoring engine for all methods.
    for i,model in enumerate(MODELS):
        order=METHODS[i:]+METHODS[:i]
        if model in ['3B','7B']:order=order+['Hy-I-Q']
        if model in selected_fp8:order=order+['FP8-native']
        for method in order:
            native(model,method,'tasks','tasks')
            add(f'{model}_{method}_task_metrics','tasks_aggregate.py',[
                '--stage',s,'--scores',s/'results'/model/method/'tasks'],python=EVAL_PYTHON,
                environment={'HF_DATASETS_OFFLINE':'1','HF_HUB_OFFLINE':'1'})
    protocol=read(s/'protocol_draft.json')
    protocol.update(status='Frozen final protocol pending manifest creation',version=1,
        final_jobs=len(jobs),optional_fp8_models=selected_fp8,optional_fp8_smoke=fp8_status,
        speed_repeat={'methods':['Hy-F-Q','Mix-I','Hy-I'],'all_models':True,'same_full_16_cell_matrix':True},
        tasks_device_batch={'max_num_seqs':1,'capture_sizes':[1],'api_submission_group':32,'reason':'Training-only grouped-vs-single check failed at device batch 32. All methods use single device sequence and unchanged 0.001 mean-logprob tolerance.'},
        family_capacity={'Mistral-7B_batch128_prompt1024':'Predeclared infeasible under common 10 GiB KV; identical exclusion for all methods, no scheduler-preemption timing.'},
        uncertainty={'ppl':'Paired 1000 bootstrap draws of complete 2048-token blocks; report ratio/delta, no mean raw PPL across models.',
                     'tasks':'Paired 1000 document bootstrap draws; acc and acc_norm both reported, acc_norm headline except PIQA acc.',
                     'speed':'All raw groups, five samples each; CV <= 5%; no cherry-picking repetitions.'},
        format_probe_scope='Repeated fixed exact-representable device buffers; independent kernel-only and reference activation pack+GEMM. Format-dependent L2 residency is possible, so this is not intrinsic MMA throughput. Reference pack is not optimized serving pack.',
        selector_quantizer_ablation={'models':['3B','7B'],'extra_method':'Hy-I-Q','same_I_mask_and_plan':True,'same_FQ_clipping_bank':True,
            'metrics':['speed six points','WT2/C4 PPL','full downstream tasks','cached decode quality'],'primary_winner_not_reselected':True},
        nonbucket_quality={'model':'7B','methods':['K0-F-Q','Hy-F-Q'],'batches':[3,12,24,48],'prompts':[128,1024],
            'teacher_forced':True,'same_engineering_tolerances':True},
        short_free_generation={'prompts':4,'prompt_tokens':128,'generated_tokens':32,'enqueue_barrier':True,'recorded_logprobs':5,
            'note':'An initial unbarriered development run had one token-sequence mismatch despite zero paired teacher-forced differences; controlled development repeats are required and retained.'},
        startup_scope='Dedicated instrumented load; CPU elapsed and CUDA event spans include bank I/O/host gaps. Not pure GPU quantizer time.',
        final_cell_policy='Run every declared cell. Hardware-infeasible and unsupported optional paths remain explicit. Any failed numerical or completion gate prevents unconditional acceptance.')
    write(s/'protocol.json',protocol);write(s/'final_jobs.json',jobs)
    print('Declared',len(jobs),'final jobs; no final scores read',flush=True)


if __name__=='__main__':main()
