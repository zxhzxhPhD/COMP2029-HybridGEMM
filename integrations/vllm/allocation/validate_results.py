"""Final cross-file audit of completed quality, budgets, controls and handoff."""
import argparse
import os
from pathlib import Path
from alloc_utils import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
protocol=read(args.out/'protocol.json');selected=read(args.out/'selected.json')['models']
assert read(args.out/'q2_complete.json')['passed'] and read(args.out/'q3_complete.json')['passed']
assert read(args.out/'score_control.json')['passed'] and read(args.out/'performance_complete.json')['passed']
assert read(args.out/'fixed_reselection_complete.json')['passed']
assert sha(os.environ['HYGEMM_LIBRARY'])==protocol['kernel_sha256']
assert sha(os.environ['MIXLLM_LIBRARY'])==protocol['mixllm_sha256']
counts={};bank_files=0;max_gap=0.;model_summaries={}
for model in MODEL_NAMES:
    root=args.out/model;values={p.stem:read(p) for p in (root/'native').glob('*.json')}
    required={'q1_alias',*read(root/'q2_views.json'),*read(root/'q3_views.json')}
    for path in [*root.glob('joint_*.json'),*root.glob('fixed_reselection_*.json')]:
        rows=read(path);required.update(rows)
        required.update(v['paired_rtn_native_control'] for v in rows.values() if 'paired_rtn_native_control' in v)
    assert required<=values.keys(),(model,required-values.keys())
    alias=values['q1_alias'];assert alias['q1_alias_max_block_nll_delta']==0
    for name,row in values.items():
        assert row['completed'] and row['validation']['tokens']==262016 and len(row['validation']['blocks'])==128
        assert row['validation_token_sha256']==alias['validation_token_sha256']
        assert row['kernel_sha256']==protocol['kernel_sha256'] and row['graph']=='FULL' and not row['inductor']
        assert row['config_sha256']==sha(root/'views'/name/'config.json')
        assert row['selection_sha256']==sha(root/'views'/name/'selection.json')
        assert row['reference_plans_sha256']==sha(root/'reference_plans.json')
        assert row['max_batching_delta_nll']<=.001
        spec=row['spec'];actual=read(root/'budgets'/f'{name}.json')
        assert actual['passed'] and actual['persistent_quantized_storage_bytes']==spec['budget']['persistent_quantized_bytes']
        gap=spec['target_payload_bits_per_weight']-actual['actual_payload_bits_per_weight']
        assert 0<=gap<.001 and actual['actual_payload_bits_per_weight']<5.1;max_gap=max(max_gap,gap)
    choice=read(root/'q3_quantizer_choice.json')
    assert choice['mode']==min(choice['mean_nll_over_six_fixed_masks'],key=choice['mean_nll_over_six_fixed_masks'].get)
    assert choice['native_rtn_bank_max_block_nll_delta']<=1e-4
    for manifest_path in (root/'banks').glob('*/manifest.json'):
        bank=read(manifest_path);assert bank['completed']
        for spec in bank['layers'].values():
            assert sha(manifest_path.parent/spec['file'])==spec['sha256'];bank_files+=1
    primary=selected[model]
    assert primary['spec']['seed']==2029 and '_128_' not in primary['view'] and '_seed' not in primary['view']
    assert primary['selection_sha256']==sha(primary['selection'])
    if primary['bank']:assert primary['bank_sha256']==sha(primary['bank'])
    counts[model]=len(values)
    model_summaries[model]={'primary':primary['view'],'ppl':primary['quality']['ppl'],
        'payload_bits_per_weight':primary['spec']['payload_bits_per_weight'],
        'physical_bytes':primary['spec']['budget']['persistent_quantized_bytes'],
        'rtn_bank_max_block_nll_delta':choice['native_rtn_bank_max_block_nll_delta']}
performance=read(args.out/'performance_complete.json')
timing_cells=0;max_cv=0.
for row in performance['roles'].values():
    for key in ['inherited','proposed']:
        if key not in row:continue
        report=read(row[key]);assert report['completed']
        assert report['kernel_sha256']==protocol['kernel_sha256'] and not report['inductor']
        for cell in report['cells']:
            assert cell['stable'] and cell['cv']<=.05;max_cv=max(max_cv,cell['cv']);timing_cells+=1
            for attempt in cell['attempts']:
                for sample in attempt['samples']:assert sample['graph_steps']==sample['steps']>=32
write(args.out/'validation_complete.json',{'passed':True,'native_quality_runs':sum(counts.values()),
    'native_runs_per_model':counts,'bank_files_verified':bank_files,'max_payload_gap_bits_per_weight':max_gap,
    'timing_cells':timing_cells,'max_timing_cv':max_cv,'models':model_summaries,
    'kernel_and_mixllm_binary_hashes_unchanged':True,'validation_only':True,'test_scored':False})
print('Validated',sum(counts.values()),'native quality runs,',bank_files,'bank files,',timing_cells,'timing cells')
