"""Serial, resumable Q3 cross ablation and bounded calibration sensitivity."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from alloc_utils import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();src=Path(__file__).resolve().parent
    protocol=read(args.out/'protocol.json');formal=Path(protocol['formal']);q1=Path(protocol['q1'])
    assert read(args.out/'q2_complete.json')['passed']
    logs=args.out/'logs'

    def command(model,label,script,*options):
        cmd=[sys.executable,str(src/script),*map(str,options)]
        write(args.out/'status.json',{'phase':f'{model}:{label}','command':cmd})
        with (logs/f'{model}_{label}.log').open('a') as f:
            status=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if status:
            write(args.out/'status.json',{'phase':'failed','model':model,'job':label,'returncode':status})
            raise RuntimeError((model,label))
        print('Finished',model,label,flush=True)

    def evaluate(model,name):
        root=args.out/model;view=root/'views'/name;target=root/'native'/f'{name}.json'
        spec=read(view/'view.json')
        identity={'script_sha256':sha(src/'evaluate.py'),'config_sha256':sha(view/'config.json'),
            'selection_sha256':sha(view/'selection.json'),'bank_sha256':sha(spec['bank']) if spec['bank'] else None,
            'loader_sha256':sha(src.parent/'hygemm_vllm/offline.py'),
            'weights_loader_sha256':sha(src.parent/'hygemm_vllm/offline_weights.py'),
            'kernel_sha256':sha(os.environ['HYGEMM_LIBRARY'])}
        stamp=logs/f'{model}_{name}.identity.json'
        if target.exists() and read(target).get('completed'):
            assert read(stamp)==identity,('Stale result',target);return
        write(stamp,identity);command(model,name,'evaluate.py','--root',root,'--view',name)

    def collect(model,mode,samples):
        root=args.out/model;folder=root/'scores'/f'{mode}_{samples}'
        bank=root/'banks'/mode/'manifest.json' if mode!='rtn' else None
        manifest=folder/'manifest.json'
        if manifest.exists():
            old=read(manifest)
            assert old['completed'] and old['samples']==samples
            assert old['bank_sha256']==(sha(bank) if bank else None)
            assert old['source_sha256']==sha(src/'collect_scores.py')
        else:
            opts=['--root',root,'--formal',formal/model,'--samples',samples,'--out',folder]
            if bank:opts+=['--bank',bank]
            command(model,f'scores_{mode}_{samples}','collect_scores.py',*opts)
        return folder

    def joint(model,mode,samples,families):
        root=args.out/model;folder=collect(model,mode,samples)
        bundle=torch.load(folder/'scores.pt',map_location='cpu',weights_only=True);data=bundle['scores']
        meta=read(root/'original_calibration_manifest.json')
        target=(read(q1/model/'budget_expected.json')['payload_bits']-4*meta['weights'])//4
        bank=root/'banks'/mode/'manifest.json' if mode!='rtn' else None
        records={}
        for family in families:
            selection,audit=allocate(data[family],meta['shapes'],target)
            name=f'q3_joint_{mode}_{samples}_{family}' if bank else f'q2_cal{samples}_{family}'
            records[name]=make_view(root,name,selection,q1/model,'Q3_joint' if bank else 'Q2_calibration',family,
                bank=bank,extra={'allocation':audit,'score_manifest':str(folder/'manifest.json'),
                    'score_sha256':sha(folder/'scores.pt'),'calibration_samples':samples})
            evaluate(model,name)
            if bank:
                ref_selection,ref_audit=allocate(bundle['paired_rtn_scores'][family],meta['shapes'],target)
                standalone=root/'views'/f'q2_cal{samples}_{family}'/'selection.json'
                if standalone.exists() and read(standalone)==ref_selection:
                    control=f'q2_cal{samples}_{family}'
                else:
                    control=f'q3_paired_rtn_{samples}_{family}'
                    make_view(root,control,ref_selection,q1/model,'Q3_RTN_gradient_control',family,
                        extra={'allocation':ref_audit,'score_sha256':sha(folder/'scores.pt'),'paired_quantizer_view':name})
                    evaluate(model,control)
                records[name]['paired_rtn_native_control']=control
        write(root/f'joint_{mode}_{samples}.json',records)

    control_root=args.out/'controls/original_deterministic';control_root.mkdir(parents=True,exist_ok=True)
    if not (control_root/'tokens.pt').exists():(control_root/'tokens.pt').symlink_to(args.out/'0.5B/tokens.pt')
    if not (control_root/'scores/2029/manifest.json').exists():
        command('0.5B','original_deterministic','deterministic_reference.py','--root',control_root,'--formal',formal/'0.5B')
    collect('0.5B','rtn',32)
    command('0.5B','verify_score_control','verify_score_control.py','--out',args.out)
    for model in ['3B','7B','0.5B','1.5B']:
        root=args.out/model
        if not (root/'banks_complete.json').exists():
            command(model,'build_banks','build_banks.py','--root',root,'--formal',formal/model)
        command(model,'prepare_q3','prepare_q3.py','--root',root)
        design=read(root/'q3_design.json')
        for name in design['views']:evaluate(model,name)
        control=read(root/'native/q3_native_rtn__q1_integer.json')['validation']['blocks']
        old=read(root/'native/q1_alias.json')['validation']['blocks']
        delta=max(abs(a['nll']-b['nll'])/a['tokens'] for a,b in zip(control,old))
        assert delta<=1e-4
        means={mode:float(np.mean([read(root/'native'/f'q3_{mode}__{mask}.json')['validation']['mean_nll']
                                  for mask in design['fixed_mask_keys']])) for mode in ['mse_clip','act_mse_clip']}
        best=min(means,key=means.get)
        families=['loss_abs_gain','loss_signed_gain']
        winner=design['q2_winner'].replace('q2_','',1)
        if winner in FAMILIES and winner not in families:families.append(winner)
        write(root/'q3_quantizer_choice.json',{'mode':best,'mean_nll_over_six_fixed_masks':means,
            'selection_rule':'lowest arithmetic mean NLL over the six predeclared masks',
            'native_rtn_bank_max_block_nll_delta':delta,'joint_families':families})
        joint(model,best,32,families)
    # Predeclared sample-count sensitivity: same seed, length, teacher and first
    # 32 windows; Q3 bank moments remain fixed at 32 samples to isolate scoring.
    for model in ['3B','7B']:
        best=read(args.out/model/'q3_quantizer_choice.json')['mode']
        for mode in ['rtn',best]:joint(model,mode,128,['loss_abs_gain','loss_signed_gain'])
    write(args.out/'q3_complete.json',{'passed':True,'models':MODEL_NAMES,'calibration_sensitivity_models':['3B','7B']})
    write(args.out/'status.json',{'phase':'q3_complete'})


if __name__=='__main__':main()
