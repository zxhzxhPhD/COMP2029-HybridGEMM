"""Tune only small-token HyGEMM plans; preserve every MixLLM/large plan."""
import argparse
import copy
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time
import statistics
import torch
from hygemm_vllm.native import DualLinear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'formal'))
from tune import make_reference, input_reference, measure as measure_resident

BUCKETS = [1, 2, 4, 8, 16, 32]


def config(row):
    return {k: row[k] for k in ['low_id', 'high_id', 'policy']}


def measure_cold(r, x, kind, cfg, expected, index, tids, active_ids=None):
    """One cold-weight graph replay per event sample; eviction is untimed."""
    r.low_id, r.high_id, r.concurrent = cfg['low_id'], cfg['high_id'], cfg['policy'] == 'concurrent'
    y = r(x)
    cols, ref = index, expected
    if active_ids is not None:
        mask = torch.isin(index, active_ids)
        cols, ref = index[mask], expected[:, mask]
    assert torch.isfinite(y[:, cols]).all()
    torch.testing.assert_close(y[tids[:, None], cols[None, :]], ref, atol=.01, rtol=.003)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = r(x)
    graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(3):
        group = []
        for _ in range(8):
            # 256 MiB exceeds the 5090 L2. Restore only the current activation
            # to cache: preceding model operators produce that input on GPU.
            measure_cold.eviction.add_(1)
            x.mul_(1)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            group.append(a.elapsed_time(b) * 1000)
        samples.append(statistics.median(group))
    return {**cfg, 'us': statistics.median(samples), 'samples_us': samples,
            'cv': statistics.pstdev(samples) / statistics.mean(samples), 'validated': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', choices=['resident', 'cold'], default='cold')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    measure = measure_cold if args.cache == 'cold' else measure_resident
    if args.cache == 'cold':
        measure_cold.eviction = torch.zeros(256 * 1024**2, device='cuda', dtype=torch.uint8)
    original = json.loads((args.formal / 'plans.json').read_text())
    data = json.loads((args.formal / 'cases.json').read_text())
    baseline = {(r['out'], r['k'], r['high'], r['tokens']): r for r in original['hygemm']}
    result = json.loads((args.out / 'plans.json').read_text()) if (args.out / 'plans.json').exists() else copy.deepcopy(original)
    selected = {(r['out'], r['k'], r['high'], r['tokens']): r for r in result['hygemm']}
    log = (args.out / 'tuning_raw.jsonl').open('a', buffering=1)
    shortfile = args.out / 'shortlists.json'
    shortlists = json.loads(shortfile.read_text()) if shortfile.exists() else {}
    decisions_file = args.out / 'decisions.json'
    decisions = json.loads(decisions_file.read_text()) if decisions_file.exists() else []
    done = {tuple(r['key']) for r in decisions}
    groups = defaultdict(list)
    for case in data['cases']:
        groups[(case['out'], case['k'], case['high'] == 0 or case['high'] == case['out'])].append(case)
    started = time.perf_counter()
    for (outputs, k, pure), cases in groups.items():
        representative = sorted(cases, key=lambda r: r['high'])[len(cases) // 2]
        w = (torch.randn(outputs, k, device='cuda') * .04).half()
        groupkey = f'{outputs}_{k}_{int(pure)}'
        shortlists.setdefault(groupkey, {})
        if any(str(t) not in shortlists[groupkey] for t in BUCKETS):
            r = DualLinear(w, torch.tensor(representative['high_ids'], device='cuda'))
            index, rw = make_reference(r, 'hygemm')
            for tokens in BUCKETS:
                if str(tokens) in shortlists[groupkey]:
                    continue
                x = torch.randn(tokens, k, device='cuda', dtype=torch.float16)
                tids, dx = input_reference(x, 'hygemm')
                expected = (dx @ rw.t()).half()
                lo, hi = r.lo, r.hi
                top = {}
                for part, active in [('low', lo), ('high', hi)]:
                    if not active.numel():
                        top[part] = [0]
                        continue
                    r.lo, r.hi = (lo, hi[:0]) if part == 'low' else (lo[:0], hi)
                    scores = []
                    for candidate in range(28):
                        cfg = {'low_id': candidate if part == 'low' else 0,
                               'high_id': candidate if part == 'high' else 0, 'policy': 'serial'}
                        score = measure(r, x, 'hygemm', cfg, expected, index, tids, active)
                        scores.append(score)
                        log.write(json.dumps({'phase': 'isolated', 'group': groupkey, 'tokens': tokens, 'part': part, **score}) + '\n')
                    top[part] = [s[part + '_id'] for s in sorted(scores, key=lambda r: r['us'])[:3]]
                r.lo, r.hi = lo, hi
                configs = [{'low_id': a, 'high_id': b, 'policy': policy} for a in top['low'] for b in top['high']
                           for policy in (['serial'] if pure else ['serial', 'concurrent'])]
                old = config(baseline[(outputs, k, representative['high'], tokens)])
                if old not in configs:
                    configs.append(old)
                scores = []
                for cfg in configs:
                    score = measure(r, x, 'hygemm', cfg, expected, index, tids)
                    scores.append(score)
                    log.write(json.dumps({'phase': 'screen', 'group': groupkey, 'tokens': tokens, **score}) + '\n')
                shortlists[groupkey][str(tokens)] = sorted(scores, key=lambda r: r['us'])[:3]
                shortfile.write_text(json.dumps(shortlists, indent=2))
                print('screen', groupkey, tokens, shortlists[groupkey][str(tokens)][0], flush=True)
            r.close()
            del r, rw, x, expected, dx
            gc.collect()
            torch.cuda.empty_cache()
        for case in cases:
            if all((outputs, k, case['high'], t) in done for t in BUCKETS):
                continue
            r = DualLinear(w, torch.tensor(case['high_ids'], device='cuda'))
            index, rw = make_reference(r, 'hygemm')
            for tokens in BUCKETS:
                key = (outputs, k, case['high'], tokens)
                if key in done:
                    continue
                x = torch.randn(tokens, k, device='cuda', dtype=torch.float16)
                tids, dx = input_reference(x, 'hygemm')
                expected = (dx @ rw.t()).half()
                old = config(baseline[key])
                configs = [old]
                for row in shortlists[groupkey][str(tokens)]:
                    if config(row) not in configs:
                        configs.append(config(row))
                scores = [measure(r, x, 'hygemm', cfg, expected, index, tids) for cfg in configs]
                for score in scores:
                    log.write(json.dumps({'phase': 'refine', 'key': key, **score}) + '\n')
                best = min(scores, key=lambda r: r['us'])
                # Require an actual margin over a fresh measurement of the old
                # plan; otherwise retain it instead of switching on noise.
                improvement = scores[0]['us'] / best['us']
                chosen = best if improvement >= 1.02 else scores[0]
                selected[key] = {**{q: baseline[key][q] for q in ['out', 'k', 'high', 'tokens']}, **chosen}
                decisions.append({'key': list(key), 'baseline': scores[0], 'best': best, 'chosen': chosen,
                                  'proxy_speedup': improvement, 'changed': config(chosen) != old})
                done.add(key)
                result['hygemm'] = [selected[(p['out'], p['k'], p['high'], p['tokens'])] for p in original['hygemm']]
                (args.out / 'plans.json').write_text(json.dumps(result, indent=2))
                decisions_file.write_text(json.dumps(decisions, indent=2))
            r.close()
            del r, rw, x, expected, dx
            gc.collect()
            torch.cuda.empty_cache()
            print('refined', outputs, k, case['high'], 'plans', len(done), flush=True)
        del w
        gc.collect()
        torch.cuda.empty_cache()
    assert len(done) == len(data['cases']) * len(BUCKETS)
    assert result['mixllm'] == original['mixllm']
    assert all(row == baseline[(row['out'], row['k'], row['high'], row['tokens'])] for row in result['hygemm'] if row['tokens'] > 32)
    (args.out / 'tuning_complete.json').write_text(json.dumps({'passed': True, 'elapsed_s': time.perf_counter() - started,
        'tuned_buckets': BUCKETS, 'candidates': list(range(28)), 'changed_plans': sum(r['changed'] for r in decisions),
        'plans_considered': len(done), 'mixllm_unchanged': True, 'tokens_64_and_above_unchanged': True,
        'search': 'median actual partition per O/K/purity; 28 isolated candidates, top3 pairs, exact-shape refinement plus original plan',
        'measurement': ('cold weights: untimed 256MiB read-modify-write eviction + activation identity warmup; 3x8 single-Graph event samples'
                        if args.cache == 'cold' else 'resident weights: 3x32 Graph calls'),
        'cache': args.cache,
        'switch_threshold': 1.02}, indent=2))


if __name__ == '__main__':
    main()
