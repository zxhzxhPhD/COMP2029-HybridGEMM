"""Freeze primary validation choices and obtain a bounded 7B timing handoff."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
from alloc_utils import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert read(args.out/'q3_complete.json')['passed']
    src=Path(__file__).resolve().parent;protocol=read(args.out/'protocol.json');q1=Path(protocol['q1'])
    def run(label,script,*options):
        write(args.out/'status.json',{'phase':label})
        with (args.out/'logs'/f'{label}.log').open('a') as log:
            subprocess.run([sys.executable,str(src/script),*map(str,options)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Finished',label,flush=True)
    # The main matrix favored fixed Q1 quotas in several models. Reuse the
    # already collected bank scores to separate new row identities from quotas.
    for model in MODEL_NAMES:
        root=args.out/model;mode=read(root/'q3_quantizer_choice.json')['mode']
        bank=root/'banks'/mode/'manifest.json';folder=root/'scores'/f'{mode}_32'
        bundle=torch.load(folder/'scores.pt',map_location='cpu',weights_only=True)
        original=read(root/'original_selection.json');records={}
        for family in ['loss_abs_gain','loss_signed_gain']:
            names=[]
            for kind,chosen_bank in [('scores',bank),('paired_rtn_scores',None)]:
                scores=bundle[kind][family]
                selection={n:sorted(scores[n].argsort(descending=True,stable=True)[:len(ids)].tolist()) for n,ids in original.items()}
                view=f'q3_fixed_reselect_{mode}_32_{family}' if chosen_bank else f'q3_fixed_paired_rtn_32_{family}'
                spec=make_view(root,view,selection,q1/model,'Q3_fixed_reselection' if chosen_bank else 'Q3_fixed_RTN_gradient_control',
                    family,bank=chosen_bank,extra={'score_sha256':sha(folder/'scores.pt'),'calibration_samples':32,
                        'quota_rule':'original per-unfused-projection counts exactly'})
                if not (root/'native'/f'{view}.json').exists():run(f'{model}_{view}','evaluate.py','--root',root,'--view',view)
                if chosen_bank:records[view]=spec
                names.append(view)
            records[names[0]]['paired_rtn_native_control']=names[1]
        write(root/f'fixed_reselection_{mode}_32.json',records)
    write(args.out/'fixed_reselection_complete.json',{'passed':True,'models':MODEL_NAMES,
        'scope':'two loss families, fixed original quotas, chosen quantizer and paired RTN; no new calibration or quantizer parameters'})
    run('analyze','analyze.py','--out',args.out)
    summary=read(args.out/'summary.json');selected={}
    for model,result in summary['models'].items():
        name=result['primary_winner'];root=args.out/model
        if not (root/'views'/name/'view.json').exists():
            assert name.startswith('q1_')
            source=q1/model/'views'/name[3:]/'selection.json'
            make_view(root,name,read(source),q1/model,'Q1_selected',name[3:])
        spec=read(root/'views'/name/'view.json')
        selected[model]={'view':name,'selection':str(root/'views'/name/'selection.json'),
            'selection_sha256':sha(root/'views'/name/'selection.json'),
            'config_sha256':sha(root/'views'/name/'config.json'),
            'bank':spec['bank'],'bank_sha256':sha(spec['bank']) if spec['bank'] else None,
            'spec':spec,'quality':result['rows'][name],'vs_mix':result['primary_winner_vs_mix']}
    write(args.out/'selected.json',{'rule':'lowest full validation mean NLL among registered seed2029 32-window primary configurations, including documented fixed-quota reselection follow-up; sensitivity seeds/sample counts excluded',
        'models':selected,'default_project_mask_changed':False,'kernel_unchanged':True,'mixllm_unchanged':True})
    # Follow up a selected fixed loss-proxy mask with its existing seed2030/31
    # identities. Keep the quantizer bank fixed; never select the best seed.
    robustness={}
    for model,record in selected.items():
        name=record['view'];root=args.out/model
        if '__' not in name or not name.startswith('q3_'):continue
        mask=name.split('__',1)[1]
        if mask not in ['q1_loss_abs_gain','q2_loss_abs_gain','q1_loss_signed_gain','q2_loss_signed_gain']:continue
        family=mask[3:];rows=[]
        for seed in [2030,2031]:
            source=(q1/model/'views'/f'{family}_seed{seed}' if mask.startswith('q1_') else root/'views'/f'q2_{family}_seed{seed}')/'selection.json'
            view=f'q3_mask_robust_seed{seed}'
            make_view(root,view,read(source),q1/model,'Q3_mask_robustness',family,seed,
                bank=Path(record['bank']),extra={'primary_view':name,'quantizer_moments_seed_fixed':2029})
            result=root/'native'/f'{view}.json'
            if not result.exists():run(f'{model}_{view}','evaluate.py','--root',root,'--view',view)
            value=read(result);assert value['completed']
            rows.append({'seed':seed,'view':view,'ppl':value['validation']['ppl'],'mean_nll':value['validation']['mean_nll']})
        robustness[model]={'primary_view':name,'primary_ppl':record['quality']['ppl'],'other_mask_seeds':rows,
            'scope':'Mask-seed sensitivity at fixed primary quantizer bank; not a full quantizer-calibration seed sweep'}
    write(args.out/'q3_mask_robustness.json',robustness)
    extra_sensitivity={}
    for model,record in selected.items():
        fixed=record['view'].startswith('q3_fixed_reselect_')
        free=record['view'].startswith('q3_joint_')
        if not (fixed or free) or (free and model in ['3B','7B']):continue
        root=args.out/model;bank=Path(record['bank']);mode=read(bank)['mode'];family=record['spec']['family']
        folder=root/'scores'/f'{mode}_128'
        if not (folder/'manifest.json').exists():
            run(f'{model}_extra_cal128','collect_scores.py','--root',root,'--formal',Path(protocol['formal'])/model,
                '--samples',128,'--bank',bank,'--out',folder)
        data=torch.load(folder/'scores.pt',map_location='cpu',weights_only=True)
        meta=read(root/'original_calibration_manifest.json');records={};names=[]
        for kind,chosen_bank in [('scores',bank),('paired_rtn_scores',None)]:
            if fixed:
                original=read(root/'original_selection.json');scores=data[kind][family]
                selection={n:sorted(scores[n].argsort(descending=True,stable=True)[:len(ids)].tolist()) for n,ids in original.items()}
                audit={'quota_rule':'original per-unfused-projection counts exactly'}
            else:selection,audit=allocate(data[kind][family],meta['shapes'],meta['high_weights'])
            prefix='q3_fixed_reselect_' if fixed else 'q3_joint_'
            view=f'{prefix}{mode}_128_{family}' if chosen_bank else f'q3_{"fixed_" if fixed else ""}paired_rtn_128_{family}'
            records[view]=make_view(root,view,selection,q1/model,'Q3_calibration_sensitivity',family,bank=chosen_bank,
                extra={'allocation':audit,'calibration_samples':128,'primary_view':record['view'],'score_sha256':sha(folder/'scores.pt')})
            if not (root/'native'/f'{view}.json').exists():run(f'{model}_{view}','evaluate.py','--root',root,'--view',view)
            names.append(view)
        records[names[0]]['paired_rtn_native_control']=names[1]
        write(root/f'{"fixed_reselection" if fixed else "joint"}_{mode}_128.json',records)
        extra_sensitivity[model]={'primary_view':record['view'],'views':names,'primary_unchanged':True}
    write(args.out/'q3_additional_sensitivity.json',extra_sensitivity)
    run('analyze_final','analyze.py','--out',args.out)
    model='7B';root=args.out/model
    methods=['q2_'+x for x in ['integer',*FAMILIES,'uniform_abs']]
    bestq2=min(methods,key=lambda n:read(root/'native'/f'{n}.json')['validation']['mean_nll'])
    roles={'original':'q1_alias','q2_best':bestq2,'selected':selected[model]['view']};results={}
    for role,name in roles.items():
        output=root/'performance'/f'{role}_inherited'
        if not (output/'decode.json').exists():run(f'decode_{role}_inherited','decode_probe.py','--root',root,'--view',name,'--out',output)
        results[role]={'view':name,'inherited':str(output/'decode.json'),'accepted_plans':None}
        if role=='original':continue
        tuning=root/'performance'/f'{role}_tuning'
        if not (tuning/'complete.json').exists():run(f'tune_{role}','tune_decode.py','--root',root,'--view',name,'--out',tuning)
        proposed=root/'performance'/f'{role}_proposed'
        if not (proposed/'decode.json').exists():
            run(f'decode_{role}_proposed','decode_probe.py','--root',root,'--view',name,'--out',proposed,
                '--plans',tuning/'plans.json','--compare',output/'decode.json')
        before=read(output/'decode.json');after=read(proposed/'decode.json')
        ratios=[a['median_ms_per_step']/b['median_ms_per_step'] for a,b in zip(before['cells'],after['cells'])]
        gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios))
        identical=after['same_selection_decode_outputs_match'] and all(x['repeated_outputs_identical'] for x in before['cells']+after['cells'])
        accept=gm>=1.01 and min(ratios)>=1/1.02 and identical
        results[role].update(proposed=str(proposed/'decode.json'),proposal_geomean_speedup=gm,
            min_cell_speedup=min(ratios),decode_tokens_identical=identical,proposal_accepted=accept,
            accepted_plans=str(tuning/'plans.json') if accept else None,
            accepted_measurement=str(proposed/'decode.json') if accept else str(output/'decode.json'))
        write(root/'performance/selection.json',results)
    write(args.out/'performance_complete.json',{'passed':True,'model':'7B','roles':results,
        'rule':'adopt whole proposed plan table only if >=1% GM model gain, no cell >2% slower, identical decode tokens; otherwise inherit Q1 per-layer plans'})
    write(args.out/'status.json',{'phase':'experiments_complete'})


if __name__=='__main__':main()
