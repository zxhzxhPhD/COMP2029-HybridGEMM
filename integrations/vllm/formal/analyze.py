"""Aggregate the matrix with explicit stability gates and separate quality."""
import argparse
import csv
import json
import math
from pathlib import Path


def geomean(values):return math.exp(sum(map(math.log,values))/len(values)) if values else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--partial',action='store_true');ap.add_argument('--plots',action='store_true')
    args=ap.parse_args();sizes=['0.5B','1.5B','3B','7B'];methods=['fp16','mixllm','hygemm']
    rows=[];quality=[];models={};all_data={}
    for size in sizes:
        data={kind:json.loads(path.read_text()) for kind in methods if (path:=args.out/size/f'result_{kind}.json').exists()}
        all_data[size]=data
        if not args.partial:assert len(data)==3 and all(d.get('completed') for d in data.values()),size
        cal_path=args.out/size/'calibration/manifest.json'
        cal=json.loads(cal_path.read_text()) if cal_path.exists() else None
        cells={kind:{(r['batch'],r['context']):r for r in d['decode']} for kind,d in data.items()}
        speedups={kind:[] for kind in ['mixllm','hygemm']};direct=[]
        for key,base in cells.get('fp16',{}).items():
            b,context=key
            for kind in methods:
                if key not in cells.get(kind,{}):continue
                x=cells[kind][key]
                stable=base['stable'] and x['stable']
                ratio=base['median_ms_per_step']/x['median_ms_per_step']
                row={'model':size,'method':kind,'batch':b,'context':context,'ms_per_step':x['median_ms_per_step'],
                    'tokens_per_second':x['tokens_per_second'],'cv':x['cv'],'attempts':len(x['attempts']),
                    'stable':x['stable'],'baseline_stable':base['stable'],'speedup_vs_fp16':ratio,
                    'steps_min':min(s['steps'] for s in x['attempts'][-1]['samples']),
                    'steps_max':max(s['steps'] for s in x['attempts'][-1]['samples']),
                    'repeat_output_variants':len({s['output_sha256'] for s in x['attempts'][-1]['samples']})}
                rows.append(row)
                if kind!='fp16' and stable:speedups[kind].append(ratio)
            if key in cells.get('mixllm',{}) and key in cells.get('hygemm',{}):
                a,z=cells['mixllm'][key],cells['hygemm'][key]
                if a['stable'] and z['stable']:direct.append(a['median_ms_per_step']/z['median_ms_per_step'])
        for kind,d in data.items():
            if 'ppl' not in d:continue
            q=d['ppl'];base=data.get('fp16',{}).get('ppl')
            quality.append({'model':size,'method':kind,'ppl':q['value'],'mean_nll':q['mean_nll'],
                'scored_tokens':q['scored_tokens'],'blocks':len(q['blocks']),
                'ppl_ratio_vs_fp16':q['value']/base['value'] if base else None,
                'delta_nll_vs_fp16':q['mean_nll']-base['mean_nll'] if base else None,
                'max_batching_delta_nll_per_token':q['max_batching_delta_nll_per_token']})
        models[size]={'completed_methods':[k for k,d in data.items() if d.get('completed')],
            'speedup_geomean_vs_fp16':{k:geomean(v) for k,v in speedups.items()},
            'stable_comparisons':{k:len(v) for k,v in speedups.items()},'hygemm_vs_mixllm_geomean':geomean(direct),
            'high_weight_fraction':cal['high_weights']/cal['weights'] if cal else None}
    summary={'models':models,'decode_cells':len(rows),'unstable_cells':[r for r in rows if not r['stable']],
        'nonidentical_repeat_cells':[r for r in rows if r['repeat_output_variants']>1],
        'ppl':quality,'interpretation':'Speedup>1 is faster. PPL lower is better. Equal channel maps do not imply equal numerical quality.'}
    name='progress_summary.json' if args.partial else 'summary.json'
    (args.out/name).write_text(json.dumps(summary,indent=2))
    if not args.partial:
        assert len(rows)==4*3*8*2 and len(quality)==12
        for filename,data in [('decode.csv',rows),('quality.csv',quality)]:
            with (args.out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(summary,indent=2))
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
        for col,size in enumerate(sizes):
            for row,context in enumerate([128,1024]):
                ax=axes[row,col]
                for kind,color in [('mixllm','#597fa8'),('hygemm','#d35b42')]:
                    points=sorted([r for r in rows if r['model']==size and r['method']==kind and r['context']==context],key=lambda r:r['batch'])
                    ax.plot([r['batch'] for r in points],[r['speedup_vs_fp16'] for r in points],'-o',color=color,label=kind)
                    for r in points:
                        if not (r['stable'] and r['baseline_stable']):ax.plot(r['batch'],r['speedup_vs_fp16'],'x',color='black',markersize=10)
                ax.axhline(1,color='gray',linestyle='--',linewidth=1);ax.set_xscale('log',base=2)
                ax.set_title(f'Qwen2.5-{size} | prompt {context}')
                ax.set_xticks([1,2,4,8,16,32,64,128],labels=[1,2,4,8,16,32,64,128]);ax.grid(alpha=.2)
                if col==0:ax.set_ylabel('Decode speedup vs FP16')
                if row==1:ax.set_xlabel('Batch')
        axes[0,0].legend();fig.suptitle('FULL decode graphs, Inductor disabled; median of 5 samples')
        fig.tight_layout();fig.savefig(args.out/'decode_speedups.png',dpi=180);fig.savefig(args.out/'decode_speedups.svg')
        plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(14,3.6))
        for ax,size in zip(axes,sizes):
            q={r['method']:r for r in quality if r['model']==size}
            vals=[q[k]['ppl'] for k in methods]
            bars=ax.bar(methods,vals,color=['#999999','#597fa8','#d35b42'])
            ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_title('Qwen2.5-'+size);ax.set_ylim(0,max(vals)*1.18)
            ax.set_ylabel('WikiText-2 PPL');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(args.out/'ppl.png',dpi=180);fig.savefig(args.out/'ppl.svg');plt.close(fig)


if __name__=='__main__':main()
