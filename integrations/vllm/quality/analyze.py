"""Audit fixed-count selection results and compute paired validation deltas."""
import argparse
import hashlib
import json
import math
import itertools
from pathlib import Path
import statistics
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a,b):
    assert len(a['blocks'])==len(b['blocks'])
    assert all(x['tokens']==y['tokens'] for x,y in zip(a['blocks'],b['blocks']))
    # Equal-length blocks here; resample blocks, not individual correlated tokens.
    differences=np.array([(x['nll']-y['nll'])/x['tokens'] for x,y in zip(a['blocks'],b['blocks'])])
    rng=np.random.default_rng(20261025)
    boot=differences[rng.integers(0,len(differences),(5000,len(differences)))].mean(1)
    return {'delta_nll':float(differences.mean()),'paired_block_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'blocks_improved':int((differences<0).sum()),'blocks':len(differences)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    root=args.out;protocol=read(root/'protocol.json');short=read(root/'shortlist.json')
    models={};decodes={};chance={}
    for model in protocol['models']:
        folder=root/model
        expected=read(folder/'budget_expected.json');budget=read(folder/'budget/original.json')
        assert budget['passed'] and budget['persistent_quantized_storage_bytes']==expected['persistent_quantized_bytes']
        assert all(r['no_fp16_weight_parameter'] for r in budget['layers'])
        old=read(folder/'original_selection.json')
        meta=read(folder/'original_calibration_manifest.json')
        chance[model]={'selected_channel_retention':sum(len(ids)**2/meta['shapes'][n][0] for n,ids in old.items())/meta['high_channels'],
                       'selected_weight_retention':sum(len(ids)**2/meta['shapes'][n][0]*meta['shapes'][n][1] for n,ids in old.items())/meta['high_weights']}
        original=read(folder/'native/original.json')
        methods=protocol['selection_methods']+['reference_fp16','reference_mixllm']
        methods += [f'{m}_seed{s}' for s in [2030,2031] for m in short['families']]
        rows=[]
        for method in methods:
            data=read(folder/'native'/f'{method}.json');assert data['completed']
            assert data['validation_token_sha256']==original['validation_token_sha256']
            assert data['max_batching_delta_nll']<=.001
            q=data['validation']
            assert math.isclose(math.exp(q['nll']/q['tokens']),q['ppl'])
            if not method.startswith('reference_'):
                assert data['library_sha256']==protocol['native_library_sha256']
                assert data['plans_sha256']==sha(folder/'plans.json')
                path=folder/'original_selection.json' if method=='original' else folder/'views'/method/'selection.json'
                assert data['selection_sha256']==sha(path)
                mask=read(path)
                assert set(mask)==set(old) and all(len(mask[n])==len(old[n]) for n in old)
                b=read(folder/'budget'/f'{method}.json')
                assert b['passed'] and b['persistent_quantized_storage_bytes']==budget['persistent_quantized_storage_bytes']
                assert [(r['name'],r['out'],r['k'],r['high']) for r in b['layers']]==[(r['name'],r['out'],r['k'],r['high']) for r in budget['layers']]
                if method!='original':assert read(folder/'views'/method/'selection_audit.json')['budget']==expected
                plans=read(folder/'plans.json')['hygemm']
                assert len(data['audit'])==len(budget['layers'])
                by_name={r['name']:r for r in budget['layers']}
                for layer in data['audit']:
                    row=by_name[layer['name']]
                    assert (layer['outputs'],layer['k'],layer['high'])==(row['out'],row['k'],row['high'])
                    candidates=sorted([r for r in plans if (r['out'],r['k'],r['high'])==(row['out'],row['k'],row['high'])],key=lambda r:r['tokens'])
                    assert layer['plans_used']
                    for count,used in layer['plans_used'].items():
                        chosen=next(r for r in candidates if r['tokens']>=int(count))
                        assert used==[chosen['tokens'],chosen['low_id'],chosen['high_id'],chosen['policy']=='concurrent']
            row={'method':method,'ppl':q['ppl'],'mean_nll':q['mean_nll'],**paired(q,original['validation'])}
            row['ppl_relative_change']=q['ppl']/original['validation']['ppl']-1
            rows.append(row)
            if data.get('decode'):
                assert all(r['stable'] and r['cv']<=.05 for r in data['decode'])
                decodes[method]=[{'batch':r['batch'],'ms':r['median_ms_per_step'],'cv':r['cv']} for r in data['decode']]
        diagnostic={}
        for file in ['diagnosis.json','diagnosis_fp32.json']:
            d=read(folder/file);assert d['completed']
            alignment=d['modes']['native_software_alignment']
            assert set(alignment['native_layer_checks'])==set(old)
            assert all(x['passed'] for x in alignment['native_layer_checks'].values())
            base=d['modes']['fp16']['mean_nll']
            modes={k:{'ppl':v['ppl'],'mean_nll':v['mean_nll'],'delta_nll_vs_reference':v['mean_nll']-base}
                   for k,v in d['modes'].items() if isinstance(v,dict) and 'ppl' in v}
            diagnostic[file]={'modes':modes,'software_native_nll_discrepancy':alignment['mean_nll_abs'],
                'interaction_nll':modes['mixed_a8']['mean_nll']-modes['w16_a8']['mean_nll']-modes['mixed_a16']['mean_nll']+base,
                'native_layer_checks':len(alignment['native_layer_checks'])}
        models[model]={'budget':expected,'other_model_parameter_buffer_bytes':budget['remaining_registered_parameter_buffer_bytes'],
                       'validation':rows,'diagnosis':diagnostic}
    robust={}
    mask_stability={}
    for family in short['families']:
        robust[family]={}
        mask_stability[family]={}
        for model in protocol['models']:
            rs=[r for r in models[model]['validation'] if r['method'] in [family,f'{family}_seed2030',f'{family}_seed2031']]
            assert len(rs)==3
            robust[family][model]={'mean_delta_nll':statistics.mean(r['delta_nll'] for r in rs),
                                  'delta_nll_range':[min(r['delta_nll'] for r in rs),max(r['delta_nll'] for r in rs)],
                                  'all_seeds_improve':all(r['delta_nll']<0 for r in rs)}
            folder=root/model;meta=read(folder/'original_calibration_manifest.json')
            masks={seed:read(folder/'views'/(family if seed==2029 else f'{family}_seed{seed}')/'selection.json') for seed in [2029,2030,2031]}
            pairs=[]
            for a,b in itertools.combinations(masks,2):
                intersection={name:len(set(ids)&set(masks[b][name])) for name,ids in masks[a].items()}
                pairs.append({'seeds':[a,b],
                    'selected_channel_retention':sum(intersection.values())/meta['high_channels'],
                    'selected_weight_retention':sum(intersection[n]*meta['shapes'][n][1] for n in intersection)/meta['high_weights']})
            mask_stability[family][model]=pairs
    baseline={r['batch']:r['ms'] for r in decodes['original']}
    for method,rows in decodes.items():
        for row in rows:row['speedup_vs_original']=baseline[row['batch']]/row['ms']
    probes={}
    for model in ['3B','7B']:
        p=read(root/model/'rank_probe.json');assert p['completed'] and p['families']==short['families']
        original=read(root/model/'original_selection.json')
        assert len(p['records'])==len(p['projection_selection'])*(2+2*len(short['families']))
        checks=[]
        for layer in p['projection_selection']:
            records={r['method']:r for r in p['records'] if r['layer']==layer}
            assert all(r['high_count']==len(original[layer]) for r in records.values())
            for family in p['families']:
                top=records[family+'_top'];bottom=records[family+'_bottom']
                assert top['predicted_score_sum'][family]>=bottom['predicted_score_sum'][family]
                checks.append({'layer':layer,'family':family,
                               'top_minus_bottom_nll':top['mean_nll']-bottom['mean_nll'],
                               'top_minus_original_nll':top['mean_nll']-records['original']['mean_nll']})
        probes[model]=checks
    summary={'passed':True,'scope':'G0/Q0/Q1 only; no cross-layer budget redistribution or kernel changes',
             'models':models,'shortlist':short,'robustness':robust,'mask_stability':mask_stability,
             'mask_retention_random_expectation':chance,'decode_7B':decodes,'rank_probe':probes,
             'diagnostic_caveat':'Software references differ slightly at whole-model level despite passing local GEMM tolerances; native validation alone ranks selectors.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'passed':True,'shortlist':short,'robustness':robust,'decode_7B':decodes},indent=2))


if __name__=='__main__':main()
