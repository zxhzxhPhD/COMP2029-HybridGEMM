"""Pin train/validation inputs and original model views for fixed-count selection."""
import argparse
from pathlib import Path
import shutil
import torch
from transformers import AutoTokenizer
from common import read, write, sha, tensor_sha, analytic_budget

ap=argparse.ArgumentParser()
ap.add_argument('--formal',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--plans-7b',type=Path,required=True)
args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
data=args.out/'data';data.mkdir(exist_ok=True)
valid=data/'wikitext2_validation.txt'
assets=read(args.formal/'data/assets.json')
if not valid.exists():
    from datasets import load_dataset
    ds=load_dataset(assets['dataset'],assets['configuration'],revision=assets['dataset_revision'])
    assert len(ds['validation'])==assets['split_rows']['validation']
    valid.write_text('\n\n'.join(ds['validation']['text']))
    write(data/'validation_source.json',{'dataset':assets['dataset'],'revision':assets['dataset_revision'],
        'configuration':assets['configuration'],'split':'validation','rows':len(ds['validation']),
        'fingerprint':ds['validation']._fingerprint,'text_sha256':sha(valid),'join':'two newline characters'})
protocol={'scope':'G0 budget audit, Q0 error diagnosis, Q1 fixed per-original-projection high counts',
 'models':['0.5B','1.5B','3B','7B'],'selection_seed':2029,'robustness_seeds':[2030,2031],
 'calibration_samples':32,'calibration_length':512,'score_chunk_rows':1024,
 'selection_methods':['original','random_2029','random_2030','random_2031','weight_gain','output_gain','loss_abs_gain','loss_signed_gain'],
 'primary_selection_metric':'full WT2 validation mean NLL; test untouched',
 'diagnostic_validation_blocks':16,'diagnostic_length':512,
 'native_validation_length':2048,'native_batching':8,
 'robustness_rule':'Select the best two nonrandom score families by mean validation NLL delta across 3B/7B seed2029; evaluate both on seeds2030/2031 for 3B and 7B. Report all seeds, no best-seed selection.',
 'float_quantizer':'frozen native pack; K32 E8M0; E2M1/E4M3 RTN; no clipping/GPTQ',
 'graph':'FULL','inductor':False,'capture_sizes':[1,2,4,8,16,32,64,128],
 'decode_spotcheck':{'model':'7B','batches':[1,8,64],'prompt':128,'warmup':1,'samples':5,'cv_gate':.05},
 'budget':'Exact original high count in EVERY unfused projection; payload and physical packed tensor bytes must match original.',
 'performance_plan':'Frozen original per-model plans; latest 3a26570 optimized 7B plans; no retuning',
 'native_library_sha256':'efea6b22b907a536b95bf343c817fada2fedd9977e6245251c4693e82159ec90'}
write(args.out/'protocol.json',protocol)
for name in protocol['models']:
    src=args.formal/name
    folder=args.out/name;folder.mkdir(exist_ok=True)
    model=src/'models/fp16'
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    train=torch.tensor(tok((args.formal/'data/wikitext2_train.txt').read_text(),add_special_tokens=False)['input_ids'])
    validation=torch.tensor(tok(valid.read_text(),add_special_tokens=False)['input_ids'])
    cals={}
    for seed in [2029,2030,2031]:
        starts=torch.randint(0,train.numel()-512,(32,),generator=torch.Generator().manual_seed(seed))
        cals[seed]=torch.stack([train[i:i+512] for i in starts])
    old=torch.load(src/'calibration/tokens.pt',map_location='cpu',weights_only=True)
    assert torch.equal(cals[2029],old['calibration'])
    # Evenly spaced windows across validation for software diagnostics; full validation for final Q1 comparison.
    n=validation.numel()//512
    diag_ids=torch.linspace(0,n-1,16).round().long()
    torch.save({'calibration':cals,'validation':validation,'diagnostic':validation[:n*512].reshape(n,512)[diag_ids],
                'diagnostic_window_ids':diag_ids},folder/'tokens.pt')
    write(folder/'data_manifest.json',{'train_text_sha256':sha(args.formal/'data/wikitext2_train.txt'),
        'validation_text_sha256':sha(valid),'validation_tokens':validation.numel(),
        'validation_token_sha256':tensor_sha(validation),'calibration_sha256':{s:tensor_sha(x) for s,x in cals.items()},
        'seed2029_matches_formal_calibration':True,'diagnostic_window_ids':diag_ids.tolist(),
        'test_read_for_calibration_identity_only':True,'test_tokens_not_scored_or_used_for_selection':True})
    selection=read(src/'calibration/selection.json')
    write(folder/'original_selection.json',selection)
    shutil.copy2(src/'calibration/manifest.json',folder/'original_calibration_manifest.json')
    shutil.copy2(args.plans_7b if name=='7B' else src/'plans.json',folder/'plans.json')
    cases=read(src/'cases.json')['layers']
    write(folder/'budget_expected.json',analytic_budget(cases))
    view=folder/'views/original';view.mkdir(parents=True,exist_ok=True)
    for file in model.iterdir():
        if file.name!='config.json' and file.is_file() and not (view/file.name).exists():
            (view/file.name).symlink_to(file.resolve())
    cfg=read(src/'models/hygemm/config.json')
    cfg['quantization_config']['tuning_plan']=str(folder/'plans.json')
    write(view/'config.json',cfg)
    print('Prepared',name,validation.numel(),'validation tokens',flush=True)
