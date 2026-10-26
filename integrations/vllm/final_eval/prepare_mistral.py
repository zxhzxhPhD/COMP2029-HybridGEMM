"""Apply the frozen larger-model selection rule to the independent family."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time
from final_common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
    stage=args.stage;folder=stage/'Mistral-7B';folder.mkdir(exist_ok=True)
    src=Path(__file__).resolve().parents[1];env={**os.environ,**environment()}
    def job(label,command,marker):
        if marker.exists():return
        write(folder/'prepare_status.json',{'phase':label,'command':command})
        with (folder/(label+'.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    while True:
        model_meta=read(stage/'mistral_download.json')
        if model_meta['status']=='complete':break
        time.sleep(10)
    model=Path(model_meta['path']);formal=folder/'formal';calibration=folder/'integer_calibration'
    data=folder/'data';data.mkdir(exist_ok=True)
    for name,origin in [('wikitext2_train.txt',FORMAL/'data/wikitext2_train.txt'),
                        ('wikitext2_validation.txt',Q1/'data/wikitext2_validation.txt')]:
        if not (data/name).exists():(data/name).symlink_to(origin.resolve())
    job('integer_calibration',[PYTHON,str(src/'formal/calibrate.py'),'--model',str(model),
        '--train',str(FORMAL/'data/wikitext2_train.txt'),'--test',str(FORMAL/'data/wikitext2_test.txt'),
        '--out',str(calibration)],calibration/'manifest.json')
    job('formal_views',[PYTHON,str(src/'formal/prepare.py'),'--model',str(model),'--calibration',str(calibration),'--out',str(formal)],formal/'cases.json')
    job('initial_tiling',[PYTHON,str(src/'formal/tune.py'),'--out',str(formal)],formal/'tuning_complete.json')
    assert (formal/'plans.json').exists()
    shutil.copy2(formal/'plans.json',folder/'reference_plans.json')
    write(folder/'reference_cases.json',{'layers':read(formal/'cases.json')['layers']})
    import torch
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    old_tokens=torch.load(calibration/'tokens.pt',map_location='cpu',weights_only=True)
    valid=torch.tensor(tokenizer((data/'wikitext2_validation.txt').read_text(),add_special_tokens=False)['input_ids'])
    if not (folder/'tokens.pt').exists():torch.save({'calibration':{2029:old_tokens['calibration']},'validation':valid},folder/'tokens.pt')
    job('banks',[PYTHON,str(src/'allocation/build_banks.py'),'--root',str(folder),'--formal',str(formal)],folder/'banks_complete.json')
    score=folder/'float_scores'
    job('floating_scores',[PYTHON,str(src/'allocation/collect_scores.py'),'--root',str(folder),'--formal',str(formal),'--out',str(score)],score/'manifest.json')
    shapes=read(calibration/'manifest.json')['shapes'];old=read(calibration/'selection.json')
    values=torch.load(score/'scores.pt',map_location='cpu',weights_only=True)['scores']['loss_signed_gain']
    selected={n:sorted(values[n].argsort(descending=True,stable=True)[:len(old[n])].tolist()) for n in old}
    # The rule is fixed before observing any new-family validation or test score.
    write(folder/'selection_rule.json',{'rule':'same per-original-projection counts as integer map; signed RTN promotion score within each layer; frozen input-weighted clipping bank',
        'seed':2029,'samples':32,'length':512,'chosen_before_quality':True,
        'integer_sha256':sha(calibration/'selection.json'),'score_sha256':sha(score/'scores.pt')})
    import sys
    sys.path.insert(0,str(src/'quality'))
    from common import analytic_budget,fused_cases
    config=read(model/'config.json');methods=read(stage/'methods.json');methods['Mistral-7B']={}
    for method in METHODS:
        dest=folder/'models'/method;cfg=deepcopy(config);budget=None;layers=None
        if method=='Mix-I':cfg=read(formal/'models/mixllm/config.json')
        elif method!='FP16':
            selection=old if method=='Hy-I' else selected
            bank=folder/'banks'/('act_mse_clip' if method=='Hy-F-Q' else 'native_rtn')/'manifest.json'
            layers=fused_cases(selection,shapes,config['num_hidden_layers']);budget=analytic_budget(layers)
            weight_count=sum(o*k for o,k in shapes.values());assert budget['payload_bits']/weight_count<5.1
            plans=read(formal/'plans.json');plans={'hygemm':plans['hygemm'],
                'binding':{'library_sha256':sha(NEW),'layout':'MX K32 E8M0 native physical scales v1',
                    'selection_canonical_sha256':canonical_sha(selection),'bank_manifest_sha256':sha(bank)}}
            planpath=folder/'plans'/(method+'.json');write(planpath,plans)
            cfg['quantization_config']={'quant_method':'hygemm_offline','checkpoint_format':'fp16_rtn','selection':selection,
                'reference_plans':str(folder/'reference_plans.json'),'reference_cases':str(folder/'reference_cases.json'),
                'decode_plans':str(planpath),'q4_bank':str(bank)}
            write(folder/'selections'/(method+'.json'),selection)
            write(folder/'shapes'/(method+'.json'),{'layers':layers,'budget':budget,'weights':weight_count})
        view(model,dest,cfg);method_env={**environment(),'HYGEMM_FUSE_LLAMA':'1'}
        methods['Mistral-7B'][method]={'model_path':str(dest),'environment':method_env,'budget':budget,
            'config_sha256':sha(dest/'config.json'),'implementation':'K1','source_checkpoint':str(model),
            'shape_manifest':str(folder/'shapes'/(method+'.json')) if layers else None,
            'status':'prepared; model gates and freeze required'}
    write(stage/'methods.json',methods)
    write(folder/'prepare_status.json',{'phase':'complete','selection_rule_frozen_without_quality_search':True})
    print('Independent-family preparation complete',flush=True)


if __name__=='__main__':main()
