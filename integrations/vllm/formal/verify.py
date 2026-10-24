"""Audit persisted results without loading models or executing GPU work."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    protocol = read(args.out / 'protocol.json')
    batches = protocol['decode']['batches']
    contexts = protocol['decode']['prompt_lengths']
    buckets = protocol['autotuning']['token_buckets']
    expected_cells = {(b, c) for b in batches for c in contexts}
    audit = {'models': {}, 'passed': False}
    for size in ['0.5B', '1.5B', '3B', '7B']:
        root = args.out / size
        selection = read(root / 'calibration/selection.json')
        calibration = read(root / 'calibration/manifest.json')
        assert digest(root / 'calibration/selection.json') == calibration['selection_sha256']
        cases = read(root / 'cases.json')
        layers = {r['layer']: r for r in cases['layers']}
        expected_plans = {(r['out'], r['k'], r['high'], t) for r in cases['cases'] for t in buckets}
        plans = read(root / 'plans.json')
        for kind in ['mixllm', 'hygemm']:
            assert len(plans[kind]) == len(expected_plans)
            assert {(r['out'], r['k'], r['high'], r['tokens']) for r in plans[kind]} == expected_plans
            assert all(r['validated'] for r in plans[kind])
        data = {k: read(root / f'result_{k}.json') for k in ['fp16', 'mixllm', 'hygemm']}
        token_signatures = set()
        record = {'plans_per_path': len(expected_plans), 'fused_layers': len(layers), 'methods': {}}
        for kind, result in data.items():
            assert result['completed'] and result['kind'] == kind
            p = result['protocol']
            assert p['graph'] == 'FULL' and p['inductor'] is False
            assert p['batches'] == batches and p['contexts'] == contexts and p['repeats'] == 5
            assert p['kv_cache_bytes'] == protocol['execution']['kv_cache_bytes']
            assert len(result['decode']) == len(expected_cells)
            assert {(r['batch'], r['context']) for r in result['decode']} == expected_cells
            steps = 0
            variants = []
            for cell in result['decode']:
                attempts = cell['attempts']
                assert 1 <= len(attempts) <= 3
                assert all(not a['stable'] for a in attempts[:-1])
                for attempt in attempts:
                    samples = attempt['samples']
                    assert len(samples) == 5
                    values = [s['decode_ms_per_step'] for s in samples]
                    cv = statistics.pstdev(values) / statistics.mean(values)
                    assert math.isclose(cv, attempt['cv'], abs_tol=1e-12)
                    assert attempt['stable'] == (cv <= .05)
                    assert statistics.median(values) == attempt['median_ms_per_step']
                    for sample in samples:
                        assert sample['batch'] == cell['batch']
                        assert sample['steps'] == sample['graph_steps'] >= 32
                        assert len(sample['step_gpu_ms']) == sample['steps']
                        assert sample['context_min'] >= cell['context']
                        assert sample['context_max'] <= cell['context'] + 127
                        assert math.isclose(sample['decode_span_ms'] / sample['steps'], sample['decode_ms_per_step'])
                        steps += sample['steps']
                for field in ['stable', 'cv', 'median_ms_per_step']:
                    assert cell[field] == attempts[-1][field]
                assert math.isclose(cell['tokens_per_second'], cell['batch'] * 1000 / cell['median_ms_per_step'])
                n = len({s['output_sha256'] for s in attempts[-1]['samples']})
                if n > 1:
                    variants.append({'batch': cell['batch'], 'context': cell['context'], 'variants': n})
            ppl = result['ppl']
            count = sum(b['tokens'] for b in ppl['blocks'])
            nll = math.fsum(b['nll'] for b in ppl['blocks'])
            assert count == ppl['scored_tokens'] == len(ppl['blocks']) * 2047
            assert all(b['tokens'] == 2047 for b in ppl['blocks'])
            assert math.isclose(nll / count, ppl['mean_nll'])
            assert math.isclose(math.exp(nll / count), ppl['value'])
            assert len(ppl['batch1_control']) == 8
            delta = max(abs(a['nll'] - b['nll']) / 2047 for a, b in zip(ppl['blocks'], ppl['batch1_control']))
            assert delta == ppl['max_batching_delta_nll_per_token'] <= .001
            token_signatures.add((ppl['test_token_sha256'], count, len(ppl['blocks']), ppl['discarded_tail_tokens']))
            if kind != 'fp16':
                cfg = read(root / 'models' / kind / 'config.json')
                assert cfg['quantization_config']['selection'] == selection
                assert result['plans_sha256'] == digest(root / 'plans.json')
                assert len(result['audit']) == len(layers)
                assert {r['name'] for r in result['audit']} == set(layers)
                indexed = {(r['out'], r['k'], r['high'], r['tokens']): r for r in plans[kind]}
                for r in result['audit']:
                    layer = layers[r['name']]
                    assert (r['outputs'], r['k'], r['high']) == (layer['out'], layer['k'], layer['high'])
                    assert r['low'] + r['high'] == r['outputs']
                    assert set(map(str, batches)).issubset(r['plans_used'])
                    for tokens, used in r['plans_used'].items():
                        bucket = next(t for t in buckets if t >= int(tokens))
                        plan = indexed[(r['outputs'], r['k'], r['high'], bucket)]
                        expected = ([bucket, plan['stage'], plan['id']] if kind == 'mixllm' else
                                    [bucket, plan['low_id'], plan['high_id'], plan['policy'] == 'concurrent'])
                        assert used == expected
            record['methods'][kind] = {'decode_cells': len(result['decode']), 'measured_graph_steps': steps,
                                      'all_stable': all(r['stable'] for r in result['decode']),
                                      'maximum_accepted_cv': max(r['cv'] for r in result['decode']),
                                      'attempts': sum(len(r['attempts']) for r in result['decode']),
                                      'repeat_output_variation': variants, 'ppl': ppl['value']}
        assert len(token_signatures) == 1
        record['common_ppl_signature'] = list(token_signatures.pop())
        audit['models'][size] = record
    audit['passed'] = True
    (args.out / 'verification.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
