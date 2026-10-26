"""Require development-only correctness evidence before the final freeze."""
import argparse
from pathlib import Path
from final_common import *

HERE=Path(__file__).resolve().parent


def prepare_jobs(stage):
    jobs=[];registry=read(stage/'methods.json')
    for model in ['3B','7B']:
        for method in registry[model]:
            if method.startswith('R-'):
                jobs.append({'id':model+'_'+method,'command':[PYTHON,str(HERE/'native_run.py'),'--stage',str(stage),
                    '--model',model,'--method',method,'--mode','dry','--label','robustness_dry']})
    for model in ['0.5B','7B','Mistral-7B']:
        for method in ['FP16','Mix-I','Hy-F-Q']:
            jobs.append({'id':model+'_'+method+'_metrics','command':[EVAL_PYTHON,str(HERE/'tasks_aggregate.py'),
                '--stage',str(stage),'--scores',str(stage/'development'/model/method/'tasks_serial_dry'),'--development'],
                'environment':{'HF_DATASETS_OFFLINE':'1','HF_HUB_OFFLINE':'1'}})
    write(stage/'remaining_preflight_jobs.json',jobs)
    print('Prepared',len(jobs),'remaining development gates',flush=True)


def check(stage):
    evidence={};required=['dry_qwen','family_dry','robustness_prepare','actual_shape_tiling','tasks_serial_dry',
        'extra_dry','development_decode','last_preparation','remaining_preflight']
    for name in required:
        file=stage/'queues'/name/'status.json';state=read(file)
        assert state['phase']=='complete',(name,state);evidence[str(file)]=sha(file)
    methods=read(stage/'methods.json')
    for model in MODELS:
        configs={name:read(Path(methods[model][name]['model_path'])/'config.json') for name in METHODS}
        q={name:configs[name].get('quantization_config',{}) for name in METHODS}
        assert q['Hy-I']['selection']==q['Mix-I']['selection'],(model,'I mask mismatch')
        assert q['Hy-F']['selection']==q['Hy-F-Q']['selection'],(model,'F mask mismatch')
        a=read(q['Hy-F']['decode_plans']);b=read(q['Hy-F-Q']['decode_plans'])
        assert a['hygemm']==b['hygemm'] and a.get('layer_plans')==b.get('layer_plans'),(model,'Quantizer comparison changes compute plan')
        cap=methods[model]['Hy-I']['budget']['payload_bits']
        for name,spec in methods[model].items():
            if spec.get('development_only') or name.startswith('tune-') or not spec.get('budget'):continue
            shapes=read(spec['shape_manifest'])['layers'];weights=sum(v['out']*v['k'] for v in shapes)
            expected=sum((4*v['out']+4*v['high'])*v['k'] for v in shapes)
            assert expected==spec['budget']['payload_bits'],(model,name,'Payload accounting mismatch')
            if name!='W8A8':assert expected<=cap and expected/weights<5.1,(model,name,'Budget exceeded')
        for method in METHODS:
            label='family_dry' if model=='Mistral-7B' else 'dry'
            file=stage/'development'/model/method/label/'result.json';r=read(file);assert r['completed']
            assert r['operations']['dry_ppl']['max_batching_delta_nll']<=.001
            evidence[str(file)]=sha(file)
    for model in ['3B','7B']:
        greedy=stage/'development'/model/'controlled_greedy_comparison.json';assert read(greedy)['passed']
        evidence[str(greedy)]=sha(greedy)
        extra=stage/'development'/model/'Hy-I-Q'/'quantizer_dry/result.json';assert read(extra)['completed']
        evidence[str(extra)]=sha(extra)
        iq=read(Path(methods[model]['Hy-I-Q']['model_path'])/'config.json')['quantization_config']
        integer=read(Path(methods[model]['Hy-I']['model_path'])/'config.json')['quantization_config']
        clipped=read(Path(methods[model]['Hy-F-Q']['model_path'])/'config.json')['quantization_config']
        assert iq['selection']==integer['selection'] and iq['q4_bank']==clipped['q4_bank']
        assert read(iq['decode_plans'])['hygemm']==read(integer['decode_plans'])['hygemm']
        for method in methods[model]:
            if method.startswith('R-'):
                file=stage/'development'/model/method/'robustness_dry/result.json';assert read(file)['completed']
                evidence[str(file)]=sha(file)
    for model in ['0.5B','7B','Mistral-7B']:
        for method in ['FP16','Mix-I','Hy-F-Q']:
            folder=stage/'development'/model/method/'tasks_serial_dry';r=read(folder/'result.json');assert r['completed']
            metrics=read(folder/'task_metrics_development.json');assert len(metrics)==3
            assert all(x['documents']==16 for x in metrics.values())
            evidence[str(folder/'result.json')]=sha(folder/'result.json')
            evidence[str(folder/'task_metrics_development.json')]=sha(folder/'task_metrics_development.json')
    checks=read(stage/'formats/checks.json');assert checks['passed'] and len(checks['records'])==28
    assert read(stage/'formats/summary.json')['completed']
    assert read(stage/'data/tasks/contract_checks.json')['passed']
    assert sha(MIX)=='1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da'
    write(stage/'preflight.json',{'passed':True,'evidence':evidence,'mix_unchanged':True,
        'quality_data':'Only validation/training used for native development scores; final C4/tasks unscored',
        'format_checks_sha256':sha(stage/'formats/checks.json'),'format_results_sha256':sha(stage/'formats/summary.json')})
    print('All pre-freeze gates passed',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--prepare-jobs',action='store_true');args=ap.parse_args()
    if args.prepare_jobs:prepare_jobs(args.stage)
    else:check(args.stage)
