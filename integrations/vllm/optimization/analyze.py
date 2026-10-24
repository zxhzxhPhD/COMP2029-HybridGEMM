"""Verify frozen controls and report the complete optimized 7B matrix."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def gm(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()
    names = {'baseline': 'hygemm', 'optimized': 'hygemm', 'mixllm': 'mixllm', 'fp16': 'fp16'}
    data = {label: read(args.out / label / f'result_{kind}.json') for label, kind in names.items()}
    old = read(args.formal / 'plans.json')
    new = read(args.out / 'optimized/plans.json')
    assert new['mixllm'] == old['mixllm']
    oldrows = {(r['out'], r['k'], r['high'], r['tokens']): r for r in old['hygemm']}
    newrows = {(r['out'], r['k'], r['high'], r['tokens']): r for r in new['hygemm']}
    assert set(oldrows) == set(newrows)
    assert all(newrows[k] == r for k, r in oldrows.items() if k[-1] >= 64)
    mask = read(args.formal / 'calibration/selection.json')
    layer_cases = {r['layer']: r for r in read(args.formal / 'cases.json')['layers']}
    cells = {}
    for label, result in data.items():
        assert result['completed']
        assert result['protocol'] == data['baseline']['protocol']
        assert result['protocol']['inductor'] is False and result['protocol']['graph'] == 'FULL'
        assert len(result['decode']) == 16
        cells[label] = {(r['batch'], r['context']): r for r in result['decode']}
        assert set(cells[label]) == {(b, c) for b in [1, 2, 4, 8, 16, 32, 64, 128] for c in [128, 1024]}
        for row in result['decode']:
            for attempt in row['attempts']:
                samples = attempt['samples']
                assert len(samples) == 5
                values = [s['decode_ms_per_step'] for s in samples]
                assert math.isclose(statistics.median(values), attempt['median_ms_per_step'])
                assert math.isclose(statistics.pstdev(values) / statistics.mean(values), attempt['cv'], abs_tol=1e-12)
                assert all(s['graph_steps'] == s['steps'] >= 32 for s in samples)
            assert row['stable'] and row['cv'] <= .05
        if label != 'fp16':
            assert result['plans_sha256'] == hashlib.sha256((args.out / label / 'plans.json').read_bytes()).hexdigest()
            cfg = read(args.out / label / 'models' / names[label] / 'config.json')
            assert cfg['quantization_config']['selection'] == mask
            assert len(result['audit']) == 112
            plans = read(args.out / label / 'plans.json')[names[label]]
            for layer in result['audit']:
                case = layer_cases[layer['name']]
                assert (layer['outputs'], layer['k'], layer['high']) == (case['out'], case['k'], case['high'])
                choices = sorted([r for r in plans if (r['out'], r['k'], r['high']) == (case['out'], case['k'], case['high'])], key=lambda r: r['tokens'])
                for token_count, used in layer['plans_used'].items():
                    chosen = next(r for r in choices if r['tokens'] >= int(token_count))
                    expected = ([chosen['tokens'], chosen['stage'], chosen['id']] if label == 'mixllm' else
                                [chosen['tokens'], chosen['low_id'], chosen['high_id'], chosen['policy'] == 'concurrent'])
                    assert used == expected
    base_quality = read(args.out / 'decode_quality_baseline.json')
    new_quality = read(args.out / 'decode_quality_optimized.json')
    max_lp, targets = 0, 0
    assert [r['batch'] for r in base_quality['records']] == [1,2,4,8,16,32]
    assert [r['batch'] for r in new_quality['records']] == [1,2,4,8,16,32]
    for a, b in zip(base_quality['records'], new_quality['records']):
        assert a['batch'] == b['batch']
        assert len(a['outputs']) == len(b['outputs']) == a['batch']
        for x, y in zip(a['outputs'], b['outputs']):
            assert len(x['tokens']) == len(y['tokens']) == len(x['logprobs']) == len(y['logprobs']) == 17
            assert x['tokens'] == y['tokens']
            max_lp = max(max_lp, max(abs(u-v) for u, v in zip(x['logprobs'], y['logprobs'])))
            targets += len(x['tokens']) - 1
    assert max_lp <= .001
    for label in ['baseline', 'optimized']:
        assert data[label]['ppl']['max_batching_delta_nll_per_token'] <= .001
    assert data['baseline']['ppl']['test_token_sha256'] == data['optimized']['ppl']['test_token_sha256']
    assert abs(data['baseline']['ppl']['mean_nll'] - data['optimized']['ppl']['mean_nll']) <= .0001
    rows = []
    for context in [128, 1024]:
        for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
            key = (batch, context)
            timing = {label: d[key]['median_ms_per_step'] for label, d in cells.items()}
            rows.append({'batch': batch, 'context': context, **{k+'_ms': v for k, v in timing.items()},
                'speedup_vs_baseline': timing['baseline']/timing['optimized'],
                'speedup_vs_mixllm': timing['mixllm']/timing['optimized'],
                'speedup_vs_fp16': timing['fp16']/timing['optimized'],
                'optimized_cv': cells['optimized'][key]['cv'],
                'large_regression_gate': batch < 64 or timing['optimized'] <= timing['baseline'] * 1.02})
    summary = {'passed': True, 'rows': rows,
               'geomean_speedup_vs_baseline': {name: gm(r['speedup_vs_baseline'] for r in rows if r['batch'] in batches)
                 for name, batches in [('small_1_to_8', [1,2,4,8]), ('middle_16_32', [16,32]), ('large_64_128', [64,128]), ('all', [1,2,4,8,16,32,64,128])]},
               'geomean_speedup_vs_mixllm': gm(r['speedup_vs_mixllm'] for r in rows),
               'geomean_speedup_vs_fp16': gm(r['speedup_vs_fp16'] for r in rows),
               'large_regression_failures': [r for r in rows if not r['large_regression_gate']],
               'max_cv': max(r['cv'] for d in data.values() for r in d['decode']),
               'quality': {'baseline_ppl': data['baseline']['ppl']['value'], 'optimized_ppl': data['optimized']['ppl']['value'],
                           'decode_logprob_max_abs': max_lp, 'matched_decode_targets': targets,
                           'scope': 'PPL uses preserved prefill plans; the separate chosen-token logprob test exercises actual small-batch FULL decode graphs'},
               'model_search': read(args.out / 'model_search/selection.json')}
    summary['passed'] = not summary['large_regression_failures']
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.out / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
        for ax, context in zip(axes, [128, 1024]):
            rs = [r for r in rows if r['context'] == context]
            for label, color in [('baseline', '#888888'), ('optimized', '#d35b42'), ('mixllm', '#597fa8')]:
                ax.plot([r['batch'] for r in rs], [r['fp16_ms']/r[label+'_ms'] for r in rs], '-o', label=label, color=color)
            ax.set_xscale('log', base=2)
            ax.set_xticks([1,2,4,8,16,32,64,128], labels=[1,2,4,8,16,32,64,128])
            ax.set_title(f'Qwen2.5-7B | prompt {context}')
            ax.set_xlabel('Batch'); ax.grid(alpha=.2)
        axes[0].set_ylabel('Decode speedup vs FP16'); axes[0].legend()
        fig.suptitle('Frozen channel map; FULL Graph; Inductor disabled')
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(args.out / f'comparison.{ext}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
