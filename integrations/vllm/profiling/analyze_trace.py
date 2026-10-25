"""Attribute captured graph nodes and account for concurrent GPU intervals."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def union_ns(intervals):
    total = 0
    stop = None
    for start, end in sorted(intervals):
        if stop is None or start > stop:
            total += end - start
            stop = end
        elif end > stop:
            total += end - stop
            stop = end
    return total


def occupancy_budget(activities, begin, end):
    """Additive exclusive category time; cross-category overlap stays explicit."""
    events = []
    for r in activities:
        a, b = max(begin, r['start']), min(end, r['end'])
        if a < b:
            events.extend([(a, 1, r['category']), (b, -1, r['category'])])
    active = Counter()
    total = Counter()
    previous = begin
    for timestamp, change, category in sorted(events):
        keys = [key for key, count in active.items() if count]
        if len(keys) == 1:
            label = keys[0]
        elif len(keys) > 1:
            label = 'cross_category_overlap'
        else:
            label = 'gpu_idle'
        total[label] += timestamp - previous
        active[category] += change
        previous = timestamp
    total['gpu_idle'] += end - previous
    assert sum(total.values()) == end - begin
    return {k: v / 1e6 for k, v in total.items()}


def classify(row):
    name, projection, table = row['name'], row.get('projection'), row['table']
    if projection:
        if table == 'memset':
            assert row['value'] == 127
            return 'activation_quant', 'scale_init'
        if 'pack_mx<' in name or 'quantize_fg_sym_f16s8' in name:
            return 'activation_quant', 'quantize'
        if 'cutlass::device_kernel<' in name:
            return 'gemm', 'low' if 'float_e2m1_t' in name else 'high'
        if 'Gemm' in name or 'gemm' in name:
            return 'gemm', 'low' if 'int4b_t' in name or 'integer_subbyte<' in name else 'high'
        if table == 'kernel' and any(s in name for s in ['copy', 'transpose', 'elementwise']):
            return 'output_layout', 'transpose_copy'
        raise ValueError(('Unclassified projection activity', row))
    if table in ['memcpy', 'memset']:
        return 'framework_gpu', table
    if any(s in name for s in ['flash_fwd', 'reshape_and_cache', 'reshape_and_cache_flash', 'paged_attention']):
        return 'attention_kv', 'attention_kv'
    if any(s in name for s in ['gemm', 'gemv', 'Gemm', 'cublas']):
        return 'lm_head', 'dense_linear'
    if any(s in name for s in ['rms_norm', 'rotary_embedding', 'silu_and_mul', 'CUDAFunctor_add', 'indexSelect']):
        return 'other_model_gpu', 'norm_rope_silu_bias_embedding'
    return 'framework_gpu', 'preprocess_sampling_other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    conn = sqlite3.connect(args.trace)
    conn.row_factory = sqlite3.Row
    strings = {r['id']: r['value'] for r in conn.execute('select * from StringIds')}
    nvtx = [dict(r) for r in conn.execute('select * from NVTX_EVENTS where end is not null')]
    for r in nvtx:
        r['text'] = r['text'] or strings.get(r['textId'], '')
    projections = defaultdict(list)
    steps = []
    for r in nvtx:
        if r['text'].startswith('projection|'):
            projections[r['globalTid']].append(r)
        if r['text'].startswith('decode_step|'):
            r['ordinal'] = int(r['text'].split('|')[1])
            steps.append(r)
    steps.sort(key=lambda r: r['ordinal'])
    assert [r['ordinal'] for r in steps] == list(range(32))
    starts = {}
    for tid, ranges in projections.items():
        ranges.sort(key=lambda r: r['start'])
        starts[tid] = [r['start'] for r in ranges]
    nodes = defaultdict(list)
    for r in conn.execute('select * from CUDA_GRAPH_NODE_EVENTS'):
        nodes[r['graphNodeId']].append(dict(r))
    cache = {}

    def projection(node):
        if node is None:
            return None
        if node in cache:
            return cache[node]
        rows = nodes[node]
        original = next((r['originalGraphNodeId'] for r in rows if r['originalGraphNodeId'] is not None), None)
        if original is not None and original != node:
            result = projection(original)
        else:
            result = None
            for r in rows:
                tid = r['globalTid']
                if tid not in starts:
                    continue
                idx = bisect_right(starts[tid], r['start']) - 1
                if idx >= 0 and projections[tid][idx]['end'] >= r['start']:
                    result = projections[tid][idx]['text']
                    break
        cache[node] = result
        return result

    runtime = {}
    launches = {}
    for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        r = dict(r)
        r['name'] = strings[r['nameId']]
        runtime[r['correlationId']] = r
        if r['name'].startswith('cudaGraphLaunch') and r['start'] >= 0:
            match = [s for s in steps if s['globalTid'] == r['globalTid'] and s['start'] <= r['start'] < s['end']]
            assert len(match) == 1
            launches[match[0]['ordinal']] = r
    assert len(launches) == 32
    activities = []
    for table, short in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'), ('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')]:
        if not conn.execute("select count(*) from sqlite_master where name=?", (table,)).fetchone()[0]:
            continue
        for row in conn.execute(f'select * from {table}'):
            r = dict(row)
            r['table'] = short
            r['name'] = strings[r['demangledName']] if short == 'kernel' else short
            r['projection'] = projection(r.get('graphNodeId'))
            r['category'], r['subtype'] = classify(r)
            activities.append(r)
    graph_starts = {}
    graph_ends = {}
    for ordinal, launch in launches.items():
        rows = [r for r in activities if r.get('graphNodeId') and r['correlationId'] == launch['correlationId']]
        graph_starts[ordinal] = min(r['start'] for r in rows)
        graph_ends[ordinal] = max(r['end'] for r in rows)
        packs = [r for r in rows if r['subtype'] == 'quantize']
        assert len(packs) == 112
        assert len({r['projection'].split('|')[1] for r in packs}) == 112
        for pack in packs:
            label = pack['projection']
            dims = dict(x.split('=') for x in label.split('|')[2:])
            branches = [r['subtype'] for r in rows if r['projection'] == label and r['category'] == 'gemm']
            assert branches.count('high') == int(int(dims['H']) > 0)
            assert branches.count('low') == int(int(dims['H']) < int(dims['O']))
    gap_transitions = defaultdict(list)
    for i in range(2, 30):
        a, b = graph_starts[i], graph_ends[i]
        spans = sorted([r for r in activities if r['start'] < b and r['end'] > a], key=lambda r: r['start'])
        right, previous = a, None
        for r in spans:
            if r['start'] > right and previous is not None:
                key = f"{previous['category']}.{previous['subtype']} -> {r['category']}.{r['subtype']}"
                gap_transitions[key].append(r['start'] - right)
            if r['end'] > right:
                right, previous = r['end'], r
    # Steady graph-start to graph-start periods retain sampling/next-step work.
    first, last = 2, 30
    begin, end = graph_starts[first], graph_starts[last]
    selected = [r for r in activities if r['start'] < end and r['end'] > begin]
    budget = occupancy_budget(selected, begin, end)
    period_rows = []
    for i in range(first, last):
        a, b = graph_starts[i], graph_starts[i + 1]
        inside = occupancy_budget(activities, a, graph_ends[i])
        whole = occupancy_budget(activities, a, b)
        assert graph_ends[i] <= b
        period_rows.append({'ordinal': i, 'period_ms': (b - a) / 1e6,
                            'graph_span_ms': (graph_ends[i] - a) / 1e6,
                            'budget_ms': whole,
                            'graph_budget_ms': inside,
                            'graph_internal_idle_ms': inside.get('gpu_idle', 0),
                            'outside_graph_idle_ms': whole.get('gpu_idle', 0) - inside.get('gpu_idle', 0),
                            'graph_launch_cpu_ms': (launches[i]['end'] - launches[i]['start']) / 1e6})
    # Per-layer metrics use whole graph launches, not clipped cross-step periods.
    correlations = {launches[i]['correlationId'] for i in range(first, last)}
    layers = defaultdict(list)
    for r in activities:
        if r['projection'] and r['correlationId'] in correlations:
            layers[r['projection']].append(r)
    per_layer = []
    for label, rows in layers.items():
        result = {'label': label, 'name': label.split('|')[1], 'categories_us': {}, 'kernel_resources': []}
        by_launch = defaultdict(list)
        for r in rows:
            by_launch[r['correlationId']].append(r)
        gaps = Counter()
        for group in by_launch.values():
            parts = {cat: [r for r in group if r['category'] == cat]
                     for cat in ['activation_quant', 'gemm', 'output_layout']}
            bounds = {cat: (min(r['start'] for r in rs), max(r['end'] for r in rs))
                      for cat, rs in parts.items() if rs}
            gaps['internal_idle_us'] += max(r['end'] for r in group) - min(r['start'] for r in group) - union_ns([(r['start'], r['end']) for r in group])
            gaps['quant_to_gemm_gap_us'] += bounds['gemm'][0] - bounds['activation_quant'][1]
            for cat, (a, b) in bounds.items():
                gaps[cat + '_stage_idle_us'] += b - a - union_ns([(r['start'], r['end']) for r in parts[cat]])
            if parts['output_layout']:
                gaps['gemm_to_layout_gap_us'] += bounds['output_layout'][0] - bounds['gemm'][1]
        result.update({k: v / 1e3 / (last - first) for k, v in gaps.items()})
        for category in ['activation_quant', 'gemm', 'output_layout']:
            spans = [(r['start'], r['end']) for r in rows if r['category'] == category]
            result['categories_us'][category] = union_ns(spans) / 1e3 / (last - first)
        for subtype in ['quantize', 'scale_init', 'low', 'high', 'transpose_copy']:
            sub = [r for r in rows if r['subtype'] == subtype]
            result[subtype + '_sum_us'] = sum(r['end'] - r['start'] for r in sub) / 1e3 / (last - first)
            result[subtype + '_count_per_step'] = len(sub) / (last - first)
        result['gemm_overlap_us'] = result['low_sum_us'] + result['high_sum_us'] - result['categories_us']['gemm']
        resources = {}
        for r in rows:
            if r['category'] != 'gemm':
                continue
            key = (r['name'], r['gridX'], r['gridY'], r['gridZ'])
            if key not in resources:
                resources[key] = {k: r[k] for k in ['name', 'subtype', 'registersPerThread', 'gridX', 'gridY', 'gridZ',
                                                   'blockX', 'blockY', 'blockZ', 'staticSharedMemory', 'dynamicSharedMemory', 'localMemoryPerThread']}
        result['kernel_resources'] = list(resources.values())
        per_layer.append(result)
    kernels = defaultdict(list)
    for r in selected:
        if r['table'] == 'kernel':
            kernels[(r['category'], r['name'])].append(r)
    kernel_summary = [{'category': cat, 'name': name, 'count': len(rs),
                       'sum_us_per_step': sum(min(end, r['end']) - max(begin, r['start']) for r in rs) / 1e3 / (last - first)}
                      for (cat, name), rs in kernels.items()]
    summary = {'trace': str(args.trace), 'steps': last - first, 'window_ns': [begin, end],
               'period_mean_ms': (end - begin) / 1e6 / (last - first),
               'period_cv': statistics.pstdev(r['period_ms'] for r in period_rows) / statistics.mean(r['period_ms'] for r in period_rows),
               'budget_ms_per_step': {k: v / (last - first) for k, v in budget.items()},
               'graph_budget_ms_per_step': {k: sum(r['graph_budget_ms'].get(k, 0) for r in period_rows) / (last - first)
                                            for k in budget},
               'graph_launch_cpu_mean_ms': statistics.mean(r['graph_launch_cpu_ms'] for r in period_rows),
               'graph_span_mean_ms': statistics.mean(r['graph_span_ms'] for r in period_rows),
               'graph_internal_idle_mean_ms': statistics.mean(r['graph_internal_idle_ms'] for r in period_rows),
               'outside_graph_idle_mean_ms': statistics.mean(r['outside_graph_idle_ms'] for r in period_rows),
               'graph_gap_transitions': sorted([{'transition': k, 'count_per_step': len(v) / (last - first),
                                                 'us_per_step': sum(v) / 1e3 / (last - first),
                                                 'median_gap_us': statistics.median(v) / 1e3}
                                                for k, v in gap_transitions.items()], key=lambda r: -r['us_per_step']),
               'periods': period_rows, 'layers': sorted(per_layer, key=lambda r: r['name']),
               'kernels': sorted(kernel_summary, key=lambda r: -r['sum_us_per_step']),
               'accounting': 'Exclusive GPU category intervals + explicit cross-category overlap + no-activity intervals sum exactly to graph-start periods; GEMM includes fused scatter.'}
    assert len(per_layer) == 112
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ['periods', 'layers', 'kernels']}, indent=2))


if __name__ == '__main__':
    main()
