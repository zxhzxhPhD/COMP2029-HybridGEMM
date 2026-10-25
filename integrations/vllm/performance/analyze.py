"""Final model tables and scientific plots from accepted, frozen configurations."""
import argparse
from collections import Counter
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import read,write,sha

MODELS=['0.5B','1.5B','3B','7B'];BATCHES=[1,2,4,8,16,32,64,128]
def gm(xs):return math.exp(sum(math.log(x) for x in xs)/len(xs))
def rows(path):return {(r['batch'],r['context']):r for r in read(path)['cells']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
    figures=stage/'figures';figures.mkdir(exist_ok=True);models={};table=[];all_cv=[]
    for model in MODELS:
        folder=stage/model;manifest=read(folder/'final/manifest.json')
        labels={'final':manifest['candidate'],'repeat':manifest['repeat'],'baseline':manifest['baseline'],
            'handoff':'hy_pre_f_inherited' if model!='7B' else 'baseline_full','mix':manifest['mix_baseline'],
            'fp16':'fp16','old_mask':'hy_pre_i'}
        data={kind:rows(folder/label/'result.json') for kind,label in labels.items()}
        keys=sorted(data['final']);assert all(set(v)==set(keys) for v in data.values())
        for values in data.values():all_cv.extend(r['cv'] for r in values.values())
        summary={'model':model,'baseline':manifest['baseline'],'mix_baseline':manifest['mix_baseline'],
            'speedup_vs_retuned_old':gm([data['baseline'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys]),
            'repeat_speedup_vs_retuned_old':gm([data['baseline'][k]['median_ms_per_step']/data['repeat'][k]['median_ms_per_step'] for k in keys]),
            'speedup_vs_stage1_handoff':gm([data['handoff'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys]),
            'speedup_vs_mix':gm([data['mix'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys]),
            'speedup_vs_fp16':gm([data['fp16'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys]),
            'small_batch_speedup':gm([data['baseline'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys if k[0]<=8]),
            'large_batch_speedup':gm([data['baseline'][k]['median_ms_per_step']/data['final'][k]['median_ms_per_step'] for k in keys if k[0]>=64]),
            'max_latency_ratio':max(data['final'][k]['median_ms_per_step']/data['baseline'][k]['median_ms_per_step'] for k in keys),
            'ppl':manifest['ppl'],'budget':manifest['budget'],'quality':read(folder/manifest['quality']/'quality.json')['comparison']}
        audit=read(folder/manifest['candidate']/'result.json')['audit'];dispatch={};families={}
        for t in BATCHES:
            dispatch[str(t)]=dict(Counter(str(r['plans_used'][str(t)][1:]) for r in audit))
            active=Counter()
            for layer in audit:
                used=layer['plans_used'][str(t)]
                if used[-1]=='fused_tma':active['mixed_TMA']+=1;continue
                for part,index in [('low',1),('high',2)]:
                    if not layer[part]:continue
                    variant=used[index]
                    family='TMA' if variant<8 else 'SIMT' if 28<=variant<=34 else 'cp_async_cached_scale' if variant>=35 else 'cp_async'
                    active[part+'_'+family]+=1
            families[str(t)]=dict(active)
        summary['dispatch_by_tokens']=dispatch
        summary['active_kernel_families_by_tokens']=families
        models[model]={'summary':summary,'data':{kind:[{'batch':k[0],'context':k[1],
            **{field:v[k][field] for field in ['median_ms_per_step','cv','stable']}} for k in keys] for kind,v in data.items()},'manifest':manifest}
        table.append(summary)
    write(stage/'summary.json',{'models':table,'max_model_cv':max(all_cv),'primary_sampling':'first accepted group; repeat is independent confirmation'})
    write(stage/'plot_data.json',models)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for reference,title,name in [('baseline','Speedup over retuned old HyGEMM','speedup_hy_pre_f'),
                                 ('mix','Speedup over MixLLM (unchanged device kernels)','speedup_mix'),
                                 ('fp16','Speedup over FP16','speedup_fp16')]:
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,model in zip(axes.flat,MODELS):
            value=models[model]['data'];final={(r['batch'],r['context']):r for r in value['final']}
            baseline={(r['batch'],r['context']):r for r in value[reference]}
            for context,color in [(128,'#2868A9'),(1024,'#D07922')]:
                y=[baseline[(b,context)]['median_ms_per_step']/final[(b,context)]['median_ms_per_step'] for b in BATCHES]
                ax.plot(BATCHES,y,'o-',color=color,label=f'Prompt {context}',markersize=4)
            ax.axhline(1,color='#777777',linestyle='--',linewidth=1)
            ax.set_xscale('log',base=2);ax.set_xticks(BATCHES,[str(b) for b in BATCHES]);ax.grid(alpha=.2)
            ax.set_title('Qwen2.5-'+model);ax.set_xlabel('Decode batch');ax.set_ylabel('Speedup (higher is better)')
        axes.flat[0].legend();fig.suptitle(title,fontsize=13)
        fig.savefig(figures/(name+'.png'),dpi=200);plt.close(fig)
    lines=['| Model | vs stage-1 handoff | vs retuned old Hy | repeat | vs Mix | vs FP16 | validation PPL |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['model']} | {r['speedup_vs_stage1_handoff']:.4f}x | {r['speedup_vs_retuned_old']:.4f}x | {r['repeat_speedup_vs_retuned_old']:.4f}x | {r['speedup_vs_mix']:.4f}x | {r['speedup_vs_fp16']:.4f}x | {r['ppl']:.6f} |")
    (stage/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
