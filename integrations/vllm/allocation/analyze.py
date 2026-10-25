"""Paired validation analysis; no test-set selection or best-seed selection."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def comparison(candidate,reference):
    a,b=candidate['validation'],reference['validation']
    assert a['tokens']==b['tokens']
    assert len(a['blocks'])==len(b['blocks'])
    assert candidate['validation_token_sha256']==reference['validation_token_sha256']
    weights=np.array([x['tokens'] for x in a['blocks']],dtype=np.float64)
    delta=np.array([x['nll']-y['nll'] for x,y in zip(a['blocks'],b['blocks'])])/weights
    gen=np.random.default_rng(2029);ids=gen.integers(0,len(delta),(10000,len(delta)))
    means=(delta[ids]*weights[ids]).sum(1)/weights[ids].sum(1)
    return {'delta_ppl':a['ppl']-b['ppl'],'relative_ppl_percent':(a['ppl']/b['ppl']-1)*100,
        'delta_mean_nll':a['mean_nll']-b['mean_nll'],
        'paired_block_bootstrap95_nll':np.quantile(means,[.025,.975]).tolist(),
        'bootstrap_note':'10k paired 2048-token-block resamples; descriptive interval, no multiplicity correction'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--q1',type=Path)
    args=ap.parse_args();protocol=read(args.out/'protocol.json');q1=args.q1 or Path(protocol['q1'])
    models={}
    for model in protocol['models']:
        old={p.stem:read(p) for p in (q1/model/'native').glob('*.json')}
        current={p.stem:read(p) for p in (args.out/model/'native').glob('*.json')}
        original=old['original'];rows={}
        for name,result in {**{'q1_'+k:v for k,v in old.items()},**current}.items():
            if 'validation' not in result:continue
            row={'ppl':result['validation']['ppl'],'mean_nll':result['validation']['mean_nll'],
                 'vs_q1_original':comparison(result,original)}
            if 'spec' in result:
                spec=result['spec'];row.update(payload_bits_per_weight=result['actual_payload_bits_per_weight'],
                    physical_bytes=spec['budget']['persistent_quantized_bytes'],
                    physical_byte_delta_vs_q1=spec['physical_byte_delta_vs_q1'],
                    pure4=spec['pure4_fused_layers'],pure8=spec['pure8_fused_layers'],mixed=spec['mixed_fused_layers'])
                groups={}
                for layer in spec['layers']:
                    group=layer['layer'].split('.',3)[-1]
                    entry=groups.setdefault(group,{'weights':0,'high_weights':0,'channels':0,'high_channels':0})
                    entry['weights']+=layer['out']*layer['k'];entry['high_weights']+=layer['high']*layer['k']
                    entry['channels']+=layer['out'];entry['high_channels']+=layer['high']
                row['partition_groups']=groups
            rows[name]=row
        q2={}
        for name in protocol['q2_primary']:
            key='q2_'+name
            if key not in current:continue
            prior=old.get('original' if name=='integer' else name)
            q2[name]={'ppl':current[key]['validation']['ppl'],
                     'vs_q1_same_method':comparison(current[key],prior) if prior else None}
        seed={}
        for family in protocol['q2_robustness_families']:
            seed[family]={}
            for stage,data,prefix in [('Q1',old,''),('Q2',current,'q2_')]:
                keys=[prefix+family]+[prefix+family+f'_seed{s}' for s in [2030,2031]]
                if all(k in data for k in keys):
                    values=[data[k]['validation'] for k in keys]
                    seed[family][stage]={'ppl':[v['ppl'] for v in values],
                        'geomean_ppl':math.exp(sum(v['mean_nll'] for v in values)/3),
                        'all_seeds_better_than_original':all(v['ppl']<original['validation']['ppl'] for v in values)}
        primary={key:value for key,value in current.items() if key=='q1_alias' or key in ['q2_'+n for n in protocol['q2_primary'] if not n.startswith('random_')]
            or key.startswith('q3_') and '_128_' not in key and '_seed' not in key and 'native_rtn' not in key and '_paired_rtn_' not in key}
        primary.update({'q1_'+name:old[name] for name in ['weight_gain','output_gain','loss_abs_gain','loss_signed_gain']})
        best=min(primary,key=lambda k:primary[k]['validation']['mean_nll'])
        q3={}
        for name,result in current.items():
            if not name.startswith('q3_') or '__' not in name:continue
            mask=name.split('__')[1]
            base=old['original'] if mask=='q1_integer' else old[mask[3:]] if mask.startswith('q1_') else current[mask]
            q3[name]={'ppl':result['validation']['ppl'],'vs_same_mask_rtn':comparison(result,base)}
        bank_stats={}
        for mode in ['mse_clip','act_mse_clip']:
            manifest=args.out/model/'banks'/mode/'manifest.json'
            if manifest.exists() and read(manifest)['completed']:
                layers=list(read(manifest)['layers'].values());groups=sum(x['stats']['groups'] for x in layers)
                stats={'groups':groups,'scope':'full-row candidate bank, including rows later assigned to FP8; not deployed-only error'}
                stats['shift_group_fractions']={str(s):sum(x['stats']['shift_groups'][str(s)] for x in layers)/groups for s in [0,1,2]}
                for objective in ['weighted','weight']:
                    before=sum(x['stats'][objective+'_error_before'] for x in layers)
                    after=sum(x['stats'][objective+'_error_after'] for x in layers)
                    stats[objective+'_error_reduction_percent']=(1-after/before)*100
                bank_stats[mode]=stats
        models[model]={'rows':rows,'q1_to_q2':q2,'seed_robustness':seed,'q3_fixed_masks':q3,'bank_statistics':bank_stats,
                       'primary_winner':best,'primary_winner_vs_mix':comparison(primary[best],old['reference_mixllm'])}
        print(model,' '.join(f'{k}={v["ppl"]:.6f}' for k,v in q2.items()),'winner',best,flush=True)
    summary={'protocol':protocol,'models':models,'complete_q2':(args.out/'q2_complete.json').exists(),
             'complete_q3':(args.out/'q3_complete.json').exists()}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
