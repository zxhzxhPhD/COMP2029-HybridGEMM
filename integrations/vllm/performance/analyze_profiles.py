"""Re-account initial/final traces with one classifier and unprofiled controls."""
import argparse
from pathlib import Path
import statistics
import subprocess
import sys
from prepare import read,write
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parents[1];rows=[]
for phase in ['initial','final']:
    folder=stage/('profile_'+phase)
    for kind in ['hygemm','mixllm']:
        controls=read(folder/f'control_{kind}.json')['cells'];meter=read(folder/f'meter_{kind}.json')['cells']
        for batch in [1,8,64]:
            target=folder/f'accounting_v2_{kind}_b{batch}.json'
            subprocess.run([sys.executable,str(src/'profiling/analyze_trace.py'),'--trace',str(folder/f'{kind}_b{batch}.sqlite'),
                            '--out',str(target)],stdout=subprocess.DEVNULL,check=True)
            value=read(target);summary=value.get('summary',value)
            control=next(r for r in controls if r['batch']==batch);m=next(r for r in meter if r['batch']==batch)
            samples=m['attempts'][-1]['samples'];graph=statistics.median(s['roi_graph_ms'] for s in samples)
            outside=statistics.median(s['roi_outside_graph_ms'] for s in samples)
            rows.append({'phase':phase,'kind':kind,'batch':batch,'control_ms':control['median_ms_per_step'],
                'control_cv':control['cv'],'unprofiled_graph_ms':graph,'unprofiled_outside_graph_ms':outside,
                'trace_graph_ms':summary['graph_span_mean_ms'],
                'trace_vs_meter_relative_delta':summary['graph_span_mean_ms']/graph-1,
                'exclusive_graph_budget_ms':summary['graph_budget_ms_per_step'],
                'trace_file':str(target.relative_to(stage))})
write(stage/'profile_comparison.json',{'records':rows,
    'scope':'GEMM includes scatter; fused_producer_quant combines SiLU and pack; overlapping GPU intervals are union-accounted.',
    'performance_source':'Unprofiled controls, not traced request latency',
    'hardware_counters':'Unavailable: ERR_NVGPUCTRPERM; no achieved-occupancy/stall claim'})
