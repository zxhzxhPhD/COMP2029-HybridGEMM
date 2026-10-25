"""Join independent controls with trace attribution; save tables and figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()
    plans = read(args.formal / 'plans.json')
    cases = {r['layer']: r for r in read(args.formal / 'cases.json')['layers']}
    plan_hash = hashlib.sha256((args.formal / 'plans.json').read_bytes()).hexdigest()
    rows, layers, resources = [], [], []
    for kind in ['mixllm', 'hygemm']:
        control = read(args.out / f'control_{kind}.json')
        meter = read(args.out / f'meter_{kind}.json')
        for document in [control, meter, *[read(args.out / f'profile_{kind}_b{b}.json') for b in [1, 8, 64]]]:
            assert document['plans_sha256'] == plan_hash
            assert document['configuration'] == control['configuration']
            assert len(document['audit']) == 112
            for layer in document['audit']:
                case = cases[layer['name']]
                assert (layer['outputs'], layer['k'], layer['high']) == (case['out'], case['k'], case['high'])
                for batch in [1, 8, 64]:
                    plan = next(p for p in plans[kind] if (p['out'], p['k'], p['high'], p['tokens']) == (case['out'], case['k'], case['high'], batch))
                    expected = ([batch, plan['stage'], plan['id']] if kind == 'mixllm' else
                                [batch, plan['low_id'], plan['high_id'], plan['policy'] == 'concurrent'])
                    assert layer['plans_used'][str(batch)] == expected
        for batch in [1, 8, 64]:
            c = next(r for r in control['cells'] if r['batch'] == batch)
            m = next(r for r in meter['cells'] if r['batch'] == batch)
            assert c['stable'] and m['stable']
            samples = m['attempts'][-1]['samples']
            for doc in [c, m]:
                for attempt in doc['attempts']:
                    for s in attempt['samples']:
                        assert s['steps'] == s['graph_steps'] >= 32
            trace = read(args.out / f'attribution_{kind}_b{batch}.json')
            for layer in trace['layers']:
                dims = {k: int(v) for k, v in (part.split('=') for part in layer['label'].split('|')[2:])}
                case = cases[layer['name']]
                assert dims == {'O': case['out'], 'K': case['k'], 'H': case['high'], 'T': batch}
            profile = read(args.out / f'profile_{kind}_b{batch}.json')['cells'][0]['sample']
            hashes = {s['output_sha256'] for cell in [c, m] for attempt in cell['attempts'] for s in attempt['samples']}
            hashes.add(profile['output_sha256'])
            assert len(hashes) == 1, ('Instrumentation changed repeated output', kind, batch)
            assert profile['steps'] == profile['graph_steps']
            assert profile['profile_interior_context'] == samples[0]['roi_context']
            assert abs(sum(trace['budget_ms_per_step'].values()) - trace['period_mean_ms']) < 1e-9
            assert abs(sum(trace['graph_budget_ms_per_step'].values()) - trace['graph_span_mean_ms']) < 1e-9
            graph_ms = statistics.mean(s['roi_graph_ms'] for s in samples)
            row = {'kind': kind, 'batch': batch, 'control_ms': c['median_ms_per_step'], 'control_cv': c['cv'],
                   'meter_full_decode_ms': m['median_ms_per_step'], 'meter_roi_cv': m['attempts'][-1]['roi_cv'],
                   'roi_period_ms': statistics.mean(s['roi_period_ms'] for s in samples),
                   'roi_graph_ms': graph_ms, 'roi_outside_graph_ms': statistics.mean(s['roi_outside_graph_ms'] for s in samples),
                   'graph_launch_cpu_ms': statistics.mean(s['roi_graph_cpu_ms'] for s in samples),
                   'traced_period_ms': trace['period_mean_ms'], 'traced_graph_ms': trace['graph_span_mean_ms'],
                   'trace_graph_relative_error': trace['graph_span_mean_ms'] / graph_ms - 1,
                   'graph_internal_gap_ms': trace['graph_internal_idle_mean_ms']}
            assert abs(row['trace_graph_relative_error']) <= .02
            for category in ['activation_quant', 'gemm', 'output_layout', 'attention_kv', 'other_model_gpu', 'framework_gpu']:
                row[category + '_ms'] = trace['graph_budget_ms_per_step'].get(category, 0)
            row['scale_init_ms'] = sum(r['scale_init_sum_us'] for r in trace['layers']) / 1000
            row['quantize_kernel_ms'] = sum(r['quantize_sum_us'] for r in trace['layers']) / 1000
            row['lm_head_traced_ms'] = trace['budget_ms_per_step']['lm_head']
            rows.append(row)
            for layer in trace['layers']:
                layers.append({'kind': kind, 'batch': batch, 'name': layer['name'],
                               **{k: v for k, v in layer.items() if k.endswith('_us') or k.endswith('_count_per_step')},
                               **{k + '_us': v for k, v in layer['categories_us'].items()}})
                for kernel in layer['kernel_resources']:
                    resources.append({'kind': kind, 'batch': batch, 'layer': layer['name'], 'path': kernel['subtype'],
                                      'ctas': kernel['gridX'] * kernel['gridY'] * kernel['gridZ'],
                                      'threads_per_cta': kernel['blockX'] * kernel['blockY'] * kernel['blockZ'],
                                      'registers_per_thread': kernel['registersPerThread'],
                                      'shared_bytes': kernel['staticSharedMemory'] + kernel['dynamicSharedMemory'],
                                      'reported_local_bytes_per_thread': kernel['localMemoryPerThread']})
    summary = {'passed': True, 'plan_sha256': plan_hash, 'rows': rows,
               'definitions': {'control': 'Unprofiled median of five full-batch decode generations',
                               'roi': 'Unprofiled CUDA events outside existing Graph; pure-decode ordinals 10:38',
                               'trace': '32 decode steps, discard two at each edge; exclusive GPU interval union',
                               'graph_internal_gap': 'No traced kernel/memcpy/memset activity inside Graph; not identified as CPU overhead',
                               'gemm': 'Both branches combined by interval union, includes fused scatter',
                               'output_layout': 'Additional projection transpose/copy only; common attention layout remains in framework_gpu'}}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    for filename, records in [('breakdown.csv', rows), ('layers.csv', layers), ('resources.csv', resources)]:
        fields = list(dict.fromkeys(key for r in records for key in r))
        with (args.out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(rows, indent=2))
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cats = [('gemm_ms', 'GEMM incl. scatter', '#416b91'), ('activation_quant_ms', 'Activation quant/init', '#e7a04b'),
                ('output_layout_ms', 'Extra output copy', '#b7669c'), ('other', 'Other model/aux GPU', '#86b7a2'),
                ('graph_internal_gap_ms', 'Graph internal gaps', '#b4b4b4')]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
        for ax, batch in zip(axes, [1, 8, 64]):
            for i, kind in enumerate(['mixllm', 'hygemm']):
                row = next(r for r in rows if r['batch'] == batch and r['kind'] == kind)
                bottom = 0
                for key, label, color in cats:
                    value = sum(row[k] for k in ['attention_kv_ms', 'other_model_gpu_ms', 'framework_gpu_ms']) if key == 'other' else row[key]
                    ax.bar(i, value, bottom=bottom, color=color, width=.6, label=label if i == 0 and batch == 1 else None)
                    bottom += value
                ax.text(i, bottom + .05, f'{bottom:.3f}', ha='center', fontsize=10)
            ax.set_xticks([0, 1], ['MixLLM', 'HyGEMM'])
            ax.set_title(f'Batch {batch}')
            ax.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Model Graph time (ms / decode step)')
        fig.suptitle('Qwen2.5-7B | prompt 128 | FULL Graph, Inductor disabled')
        fig.legend(loc='lower center', ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .14, 1, .96))
        for ext in ['png', 'svg']:
            fig.savefig(args.out / f'graph_breakdown.{ext}', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        labels = ['GEMM', 'Quant/init', 'Output copy', 'Graph gaps', 'Other graph GPU']
        for ax, batch in zip(axes, [1, 8, 64]):
            mix, hy = [next(r for r in rows if r['batch'] == batch and r['kind'] == k) for k in ['mixllm', 'hygemm']]
            delta = [(hy[key] - mix[key]) * 1000 if key != 'other' else
                     sum(hy[k] - mix[k] for k in ['attention_kv_ms', 'other_model_gpu_ms', 'framework_gpu_ms']) * 1000
                     for key in ['gemm_ms', 'activation_quant_ms', 'output_layout_ms', 'graph_internal_gap_ms', 'other']]
            bars = ax.barh(labels, delta, color=['#d16450' if x >= 0 else '#4e957b' for x in delta])
            ax.bar_label(bars, fmt='%+.0f', padding=3)
            ax.axvline(0, color='gray', linewidth=1)
            ax.set_title(f'Batch {batch}')
            ax.set_xlabel('HyGEMM minus MixLLM (microseconds)')
            ax.set_xlim(min(-1100, min(delta) - 150), 450)
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('Positive = HyGEMM cost; negative = HyGEMM saving')
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(args.out / f'difference_breakdown.{ext}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
