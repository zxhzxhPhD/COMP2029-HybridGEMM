"""Post-freeze analysis only: no model execution or candidate selection."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics
import numpy as np
from final_common import *


def geomean(values):
    values=list(values);return math.exp(sum(map(math.log,values))/len(values)) if values else None


def speed_map(result):
    return {(r['batch'],r['context']):r for r in result['operations'].get('speed',[]) if r.get('feasible',True)}


def paired_interval(x,y,seed=3071):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert x.shape==y.shape
    d=y-x;rng=np.random.default_rng(seed)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(1000)])
    return {'candidate_minus_reference':float(d.mean()),'paired_95':np.quantile(boot,[.025,.975]).tolist(),'units':len(d)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();s=args.stage
    out=s/'analysis';out.mkdir(exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    methods=read(s/'freeze/v1/methods.json');results={}
    for f in (s/'results').glob('*/*/*/result.json'):
        model,method,label=f.relative_to(s/'results').parts[:3];r=read(f)
        if r.get('completed'):results[(model,method,label)]=r
    summary={'main':{},'budget':{},'robustness':{},'ablation':{},'mechanism':{},'memory':{},'tasks':{},'paired_quality':{},'paired_quality_mix':{},'startup':{},'optional_fp8':{},'observed_pareto':{},'selector_quantizer':{}}
    for model in MODELS:
        summary['main'][model]={};summary['memory'][model]={};summary['paired_quality'][model]={};summary['paired_quality_mix'][model]={}
        reference=results.get((model,'FP16','main'))
        if not reference:continue
        base=speed_map(reference)
        for method in METHODS:
            r=results.get((model,method,'main'))
            if not r:continue
            cells=speed_map(r);mix=speed_map(results.get((model,'Mix-I','main'),reference))
            entry={'speedup_fp16':geomean(base[k]['median_ms_per_step']/v['median_ms_per_step'] for k,v in cells.items()),
                'speedup_mix':geomean(mix[k]['median_ms_per_step']/v['median_ms_per_step'] for k,v in cells.items()),
                'speed_cells':len(cells),'max_cv':max(v['cv'] for v in cells.values()),
                'unstable':sum(not v['stable'] for v in cells.values()),
                'infeasible':[v for v in r['operations']['speed'] if not v.get('feasible',True)],
                'ppl':{split:r['operations'][split]['ppl'] for split in ['test','c4']},
                'subgroups':{name:geomean(mix[k]['median_ms_per_step']/v['median_ms_per_step'] for k,v in cells.items() if predicate(k[0]))
                    for name,predicate in [('small_1_8',lambda b:b<=8),('medium_16_32',lambda b:16<=b<=32),('large_64_128',lambda b:b>=64)]}}
            summary['main'][model][method]=entry
            summary['memory'][model][method]=r['runtime_audit']
            intervals={}
            for split in ['test','c4']:
                x=[b['nll']/b['tokens'] for b in reference['operations'][split]['blocks']]
                y=[b['nll']/b['tokens'] for b in r['operations'][split]['blocks']]
                value=paired_interval(x,y);value['ppl_ratio']=math.exp(value['candidate_minus_reference'])
                value['ppl_ratio_95']=list(map(math.exp,value['paired_95']));intervals[split]=value
            summary['paired_quality'][model][method]=intervals
            mix_result=results.get((model,'Mix-I','main'))
            if mix_result:
                intervals={}
                for split in ['test','c4']:
                    x=[b['nll']/b['tokens'] for b in mix_result['operations'][split]['blocks']]
                    y=[b['nll']/b['tokens'] for b in r['operations'][split]['blocks']]
                    value=paired_interval(x,y);value['ppl_ratio']=math.exp(value['candidate_minus_reference'])
                    value['ppl_ratio_95']=list(map(math.exp,value['paired_95']));intervals[split]=value
                summary['paired_quality_mix'][model][method]=intervals
        for method in ['Hy-F-Q','Mix-I','Hy-I']:
            r=results.get((model,method,'repeat'));reference_repeat=results.get((model,'Mix-I','repeat'))
            if r and reference_repeat:
                mix=speed_map(reference_repeat)
                summary['main'][model][method]['repeat_speedup_mix']=geomean(mix[k]['median_ms_per_step']/v['median_ms_per_step'] for k,v in speed_map(r).items())
    for model in ['3B','7B']:
        weights=sum(x['out']*x['k'] for x in read(methods[model]['Hy-F-Q']['shape_manifest'])['layers'])
        cross={name:results.get((model,name,'quantizer' if name=='Hy-I-Q' else 'main')) for name in ['Hy-I','Hy-F','Hy-I-Q','Hy-F-Q']}
        if all(cross.values()):
            block_vectors={split:{name:np.array([v['nll']/v['tokens'] for v in r['operations'][split]['blocks']]) for name,r in cross.items()} for split in ['test','c4']}
            summary['selector_quantizer'][model]={'methods':{name:{'ppl':{x:r['operations'][x]['ppl'] for x in ['test','c4']},
                'six_cell_gm_ms':geomean(v['median_ms_per_step'] for k,v in speed_map(r).items() if k[0] in [1,8,64])} for name,r in cross.items()},
                'paired_nll':{split:{'F_minus_I_RTN':paired_interval(v['Hy-I'],v['Hy-F']),
                    'F_minus_I_clipped':paired_interval(v['Hy-I-Q'],v['Hy-F-Q']),
                    'interaction':paired_interval(np.zeros_like(v['Hy-I']),v['Hy-F-Q']-v['Hy-I-Q']-v['Hy-F']+v['Hy-I'])} for split,v in block_vectors.items()}}
        for category,prefix,label in [('budget','B-','budget'),('robustness','R-','robustness')]:
            rows=[]
            for method,spec in methods[model].items():
                chosen=method.startswith(prefix) or category=='budget' and method in ['Hy-F-Q','W8A8']
                if not chosen:continue
                r=results.get((model,method,'main' if method=='Hy-F-Q' else label))
                if not r:continue
                cells=speed_map(r);budget=spec['budget']
                rows.append({'method':method,'payload_bit_per_weight':budget['payload_bits']/weights,
                    'physical_bit_per_weight':8*budget['persistent_quantized_bytes']/weights,
                    'persistent_bytes':budget['persistent_quantized_bytes'],
                    'ppl':{split:r['operations'][split]['ppl'] for split in ['test','c4']},
                    'decode_ms':{f'b{k[0]}_c{k[1]}':v['median_ms_per_step'] for k,v in cells.items()},
                    'decode_gm_ms_common6':geomean(v['median_ms_per_step'] for k,v in cells.items() if k[0] in [1,8,64]),
                    'within_primary_payload':method!='W8A8'})
            summary[category][model]=rows
        curve=[r for r in summary['budget'][model] if r['within_primary_payload']]
        summary['observed_pareto'][model]={}
        for split in ['test','c4']:
            def coordinates(r):return (r['persistent_bytes'],r['ppl'][split],r['decode_gm_ms_common6'])
            summary['observed_pareto'][model][split]=[r['method'] for r in curve if not any(
                all(a<=b for a,b in zip(coordinates(t),coordinates(r))) and any(a<b for a,b in zip(coordinates(t),coordinates(r)))
                for t in curve if t is not r)]
        summary['ablation'][model]={}
        for mask in ['I','F']:
            old=results.get((model,'K0-'+mask,'ablation'));new=results.get((model,'Hy-'+mask,'main'))
            if old and new:
                a=speed_map(old);b=speed_map(new)
                summary['ablation'][model][mask]={'kernel_speedup':geomean(v['median_ms_per_step']/b[k]['median_ms_per_step'] for k,v in a.items()),
                    'cells':{f'b{k[0]}_c{k[1]}':v['median_ms_per_step']/b[k]['median_ms_per_step'] for k,v in a.items()},
                    'old_ppl':{x:old['operations'][x]['ppl'] for x in ['test','c4']},'new_ppl':{x:new['operations'][x]['ppl'] for x in ['test','c4']}}
    main7=results.get(('7B','Hy-F-Q','main'))
    if main7:
        base=speed_map(main7)
        for method in ['A-no-pack','A-no-producer','A-serial','A-no-light-high']:
            r=results.get(('7B',method,'mechanism'))
            if r:summary['mechanism'][method]={f'b{k[0]}_c{k[1]}':v['median_ms_per_step']/base[k]['median_ms_per_step'] for k,v in speed_map(r).items()}
    for method in ['Startup-FP16','Startup-Mix-I','Startup-Hy-F-Q']:
        r=results.get(('7B',method,'startup'))
        if r:
            records=r['runtime_audit']['startup_quantization_layout_timing']
            summary['startup'][method]={'load_capture_s':r['load_and_capture_s'],'quantized_projections':len(records),
                'quant_layout_host_s':sum(v['host_elapsed_ms'] for v in records)/1000,
                'event_span_s_including_host_gaps':sum(v['gpu_event_span_ms'] for v in records)/1000}
    for model in ['0.5B','7B']:
        r=results.get((model,'FP8-native','optional'));base=results.get((model,'FP16','main'))
        if r and base:
            reference=speed_map(base)
            summary['optional_fp8'][model]={'speedup_fp16':geomean(reference[k]['median_ms_per_step']/v['median_ms_per_step'] for k,v in speed_map(r).items()),
                'ppl':{x:r['operations'][x]['ppl'] for x in ['test','c4']},
                'backend_counts':dict(Counter(v['fp8_linear'] for v in r['runtime_audit']['quantization_modules'] if v.get('fp8_linear'))),
                'actual_kv_cache_dtype':r['runtime_audit']['actual_kv_cache_dtype'],
                'marlin_modules':sum(bool(v.get('use_marlin')) for v in r['runtime_audit']['quantization_modules'])}
    for f in (s/'results').glob('*/*/tasks/task_metrics.json'):
        model,method=f.relative_to(s/'results').parts[:2];summary['tasks'].setdefault(model,{})[method]=read(f)
    for reference_name,key in [('FP16','paired_tasks'),('Mix-I','paired_tasks_mix')]:
        task_pairs={}
        for model,rows in summary['tasks'].items():
            task_pairs[model]={}
            for method in rows:
                if method==reference_name or reference_name not in rows:continue
                task_pairs[model][method]={}
                for task in ['hellaswag','arc_challenge','piqa']:
                    def samples(name):
                        return {v['doc_id']:v for v in map(json.loads,(s/'results'/model/name/'tasks'/(task+'_samples.jsonl')).read_text().splitlines())}
                    a=samples(reference_name);b=samples(method);assert a.keys()==b.keys()
                    task_pairs[model][method][task]={metric:paired_interval([a[k]['metrics'][metric] for k in sorted(a)],
                        [b[k]['metrics'][metric] for k in sorted(a)]) for metric in rows[method][task]['metrics']}
        summary[key]=task_pairs
    summary['selector_quantizer_tasks']={}
    for model in ['3B','7B']:
        names=['Hy-I','Hy-F','Hy-I-Q','Hy-F-Q']
        if not all(name in summary['tasks'].get(model,{}) for name in names):continue
        summary['selector_quantizer_tasks'][model]={}
        for task,metric in [('hellaswag','acc_norm'),('arc_challenge','acc_norm'),('piqa','acc')]:
            values={}
            for name in names:
                path=s/'results'/model/name/'tasks'/(task+'_samples.jsonl')
                values[name]={v['doc_id']:v['metrics'][metric] for v in map(json.loads,path.read_text().splitlines())}
            keys=sorted(values['Hy-I']);assert all(set(v)==set(keys) for v in values.values())
            a={name:np.array([value[k] for k in keys]) for name,value in values.items()}
            summary['selector_quantizer_tasks'][model][task]={'metric':metric,
                'F_minus_I_RTN':paired_interval(a['Hy-I'],a['Hy-F']),
                'F_minus_I_clipped':paired_interval(a['Hy-I-Q'],a['Hy-F-Q']),
                'interaction':paired_interval(np.zeros_like(a['Hy-I']),a['Hy-F-Q']-a['Hy-I-Q']-a['Hy-F']+a['Hy-I'])}
    summary['decode_comparisons']={}
    for file in (s/'comparisons').glob('*/*.json'):
        r=read(file)
        if 'distributions' not in r:continue
        # Prompt zero is deliberately reused across batches. Forty records are
        # eight token histories evaluated at five batch sizes, not forty IID docs.
        histories={(v['context'],v['target_index']) for v in r['distributions']}
        summary['decode_comparisons'].setdefault(file.parent.name,{})[file.stem]={
            **{k:r[k] for k in ['targets','mean_nll_delta','mean_kl','mean_abs_target_logprob','max_abs_target_logprob','greedy_tokens_equal']},
            'full_vocabulary_records':len(r['distributions']),'unique_sampled_histories':len(histories),
            'top1_agree_records':sum(v['top1_agree'] for v in r['distributions']),
            'engineering_gate':r.get('passed'),'interpretation':'Grid-averaged paired diagnostics; prompts and histories are reused across batch sizes.'}
    summary['workloads']={};summary['offline_calibration']={}
    for operation in ['prefill','long','nonbucket']:
        summary['workloads'][operation]={}
        for method in METHODS:
            r=results.get(('7B',method,'workload_'+operation))
            if r:summary['workloads'][operation][method]=r['rows']
    for model in ['3B','7B']:
        summary['offline_calibration'][model]={}
        for path in (s/model/'robustness').glob('*/complete.json'):
            r=read(path);teacher=read(path.parent/'teacher.json')
            summary['offline_calibration'][model][path.parent.name]={'elapsed_s':r['elapsed_s'],
                'mask_jaccard_vs_primary':r['mask_jaccard_vs_primary'],'same_per_layer_high_counts':r['same_per_layer_high_counts'],
                'teacher_score_elapsed_s':teacher['score_elapsed_s'],'teacher_peak_gpu_bytes':teacher['peak_gpu_bytes']}
    if (s/'formats/summary.json').exists():summary['formats']=read(s/'formats/summary.json')
    dispatch={}
    for model in MODELS:
        dispatch[model]={}
        for method in ['Hy-I','Hy-F','Hy-F-Q']:
            cfg=read(Path(methods[model][method]['model_path'])/'config.json')['quantization_config']
            plan=read(cfg['decode_plans']);rows=plan.get('layer_plans',[])
            dispatch[model][method]={str(t):dict(Counter(f"low={r['low_id']},high={r['high_id']},{r['policy']}" for r in rows if r['tokens']==t)) for t in BATCHES}
    summary['planned_layer_dispatch_counts']=dispatch
    allocation={}
    import re
    for model in ['3B','7B']:
        allocation[model]={}
        for name in ['Hy-I','Hy-F-Q','B-integer-a100','B-uniform-a100','B-random-a100']:
            layers=read(methods[model][name]['shape_manifest'])['layers'];totals=defaultdict(lambda:[0,0])
            for row in layers:
                index=int(re.search(r'\.layers\.(\d+)\.',row['layer']).group(1))
                totals[index][0]+=row['high']*row['k'];totals[index][1]+=row['out']*row['k']
            allocation[model][name]=[{'layer':i,'high_weight_fraction':h/w,'high_weights':h,'weights':w} for i,(h,w) in sorted(totals.items())]
    summary['layer_allocation']=allocation
    summary['primary_budget']={}
    for model in MODELS:
        summary['primary_budget'][model]={}
        for method in ['Hy-I','Hy-F-Q']:
            spec=methods[model][method];budget=spec['budget'];layers=read(spec['shape_manifest'])['layers']
            weights=sum(r['out']*r['k'] for r in layers);high=sum(r['high']*r['k'] for r in layers)
            summary['primary_budget'][model][method]={**budget,'quantized_weight_elements':weights,
                'high_precision_weight_fraction':high/weights,'payload_bit_per_weight':budget['payload_bits']/weights,
                'physical_bit_per_weight':8*budget['persistent_quantized_bytes']/weights}
    # Reuse the existing matching-implementation development trace explicitly;
    # never relabel its timing as a final C4 measurement.
    import shutil
    old_cfg=read(S2/'profile_final_view/models/hygemm/config.json')['quantization_config']
    new_cfg=read(Path(methods['7B']['Hy-F-Q']['model_path'])/'config.json')['quantization_config']
    assert old_cfg['selection']==new_cfg['selection'] and sha(old_cfg['q4_bank'])==sha(new_cfg['q4_bank'])
    old_plan=read(old_cfg['decode_plans']);new_plan=read(new_cfg['decode_plans'])
    assert old_plan['layer_plans']==new_plan['layer_plans']
    control=read(S2/'profile_final/control_hygemm.json')
    assert control['library_sha256']==sha(methods['7B']['Hy-F-Q']['environment']['HYGEMM_LIBRARY'])
    assert all(methods['7B']['Hy-F-Q']['environment'][k]==v for k,v in control['runtime_options'].items())
    reuse=s/'reused_stage2_profile';reuse.mkdir(exist_ok=True)
    evidence=[S2/'profile_comparison.json',S2/'profile_final/control_hygemm.json',S2/'profile_final/control_mixllm.json',
        *sorted((S2/'profile_final').glob('accounting_v2_*.json'))]
    source_records={}
    for path in evidence:
        target=reuse/path.name;shutil.copy2(path,target);source_records[path.name]={'source':str(path),'sha256':sha(path)}
    write(reuse/'provenance.json',{'scope':'Stage-two development profiling, same accepted 7B Hy-F-Q weights/mask/compute and native library; not final holdout timing',
        'sources':source_records,'same_selection_bank_layer_plans_and_library':True})
    summary['reused_stage2_profile']=read(S2/'profile_comparison.json')
    write(out/'summary.json',summary)
    make_plots(s,fig,results,summary)
    make_report(s,summary,results)
    print('Report updated from',len(results),'completed result files',flush=True)


def make_plots(stage,folder,results,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.bbox':'tight'})
    for model in MODELS:
        if (model,'FP16','main') not in results:continue
        fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
        for ax,context in zip(axes,[128,1024]):
            for method in METHODS:
                r=results.get((model,method,'main'))
                if not r:continue
                rows=sorted((k[0],v['median_ms_per_step']) for k,v in speed_map(r).items() if k[1]==context)
                ax.plot([x for x,y in rows],[y for x,y in rows],'-o',ms=3,label=method)
            ax.set_xscale('log',base=2);ax.set_xticks(BATCHES,labels=BATCHES);ax.set_title(f'{model}, prompt {context}')
            ax.set_xlabel('Decode batch');ax.grid(alpha=.2)
            if model=='Mistral-7B' and context==1024:
                ax.text(.98,.04,'B128 infeasible\n10 GiB KV budget',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        axes[0].set_ylabel('Milliseconds / decode step');axes[1].legend(fontsize=8)
        fig.savefig(folder/(model+'_decode.png'));plt.close(fig)
    for model,rows in summary['budget'].items():
        if not rows:continue
        fig,axes=plt.subplots(2,2,figsize=(11,7));storage_fig,storage_ax=plt.subplots(figsize=(6,3.8))
        for row_index,(split,label) in enumerate([('test','WT2 test'),('c4','C4 subset')]):
            for rule,color in [('integer','C0'),('float','C1'),('uniform','C2')]:
                points=[r for r in rows if r['method'].startswith('B-'+rule) or r['method']=='B-zero' or rule=='float' and r['method']=='Hy-F-Q']
                points.sort(key=lambda r:r['payload_bit_per_weight'])
                if not points:continue
                axes[row_index,0].plot([r['payload_bit_per_weight'] for r in points],[r['ppl'][split] for r in points],'-o',ms=3,label=rule,color=color)
                axes[row_index,1].plot([r['ppl'][split] for r in points],[r['decode_gm_ms_common6'] for r in points],'-o',ms=3,color=color)
                if row_index==0:storage_ax.plot([r['persistent_bytes']/1e9 for r in points],[r['decode_gm_ms_common6'] for r in points],'-o',ms=3,label=rule,color=color)
            for r in rows:
                if r['method']=='B-random-a100':
                    axes[row_index,0].scatter(r['payload_bit_per_weight'],r['ppl'][split],marker='x',c='black',label='random')
                    axes[row_index,1].scatter(r['ppl'][split],r['decode_gm_ms_common6'],marker='x',c='black')
                    if row_index==0:storage_ax.scatter(r['persistent_bytes']/1e9,r['decode_gm_ms_common6'],marker='x',c='black',label='random')
            axes[row_index,0].set_xlabel('Projection weight-code bits / weight');axes[row_index,0].set_ylabel(label+' PPL');axes[row_index,0].legend(fontsize=8)
            axes[row_index,1].set_xlabel(label+' PPL');axes[row_index,1].set_ylabel('Decode latency, common 6-cell GM (ms)')
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(model+' within-budget measured points; no equal-quality interpolation')
        fig.tight_layout();fig.savefig(folder/(model+'_budget.png'));plt.close(fig)
        storage_ax.set_xlabel('Quantized projection storage (GB)');storage_ax.set_ylabel('Decode latency, common 6-cell GM (ms)')
        storage_ax.set_title(model+' within-budget storage / latency');storage_ax.grid(alpha=.2);storage_ax.legend(fontsize=8)
        storage_fig.tight_layout();storage_fig.savefig(folder/(model+'_budget_storage.png'));plt.close(storage_fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for ax,model in zip(axes.flat,MODELS):
        for label,style in [('main','-'),('repeat','--')]:
            a=results.get((model,'Mix-I',label));b=results.get((model,'Hy-F-Q',label))
            if not a or not b:continue
            left=speed_map(a);right=speed_map(b)
            for context,color,marker in [(128,'C0','o'),(1024,'C1','s')]:
                xs=sorted(k[0] for k in right if k[1]==context)
                ax.plot(xs,[left[(x,context)]['median_ms_per_step']/right[(x,context)]['median_ms_per_step'] for x in xs],
                    linestyle=style,marker=marker,ms=3,color=color,label=f'prompt {context}, {label}')
        ax.axhline(1,color='gray',lw=1);ax.set_xscale('log',base=2);ax.set_xticks(BATCHES,labels=BATCHES)
        ax.set_title(model);ax.set_xlabel('Batch');ax.grid(alpha=.2)
        if model=='Mistral-7B':ax.text(.02,.97,'B128 / prompt 1024 infeasible',transform=ax.transAxes,va='top',fontsize=7)
    axes[0,0].set_ylabel('Hy-F-Q speedup / Mix-I');axes[1,0].set_ylabel('Hy-F-Q speedup / Mix-I')
    axes[1,2].axis('off');handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:axes[1,2].legend(handles,labels,loc='center',fontsize=9)
    fig.tight_layout();fig.savefig(folder/'relative_speed.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,split in zip(axes,['test','c4']):
        for offset,method in enumerate(['Mix-I','Hy-I','Hy-F','Hy-F-Q']):
            xs=[];values=[];lower=[];upper=[]
            for i,model in enumerate(MODELS):
                pair=summary['paired_quality'].get(model,{}).get(method,{}).get(split)
                if not pair:continue
                xs.append(i+(offset-1.5)*.19);values.append(pair['ppl_ratio'])
                lower.append(max(0,pair['ppl_ratio']-pair['ppl_ratio_95'][0]));upper.append(max(0,pair['ppl_ratio_95'][1]-pair['ppl_ratio']))
            ax.errorbar(xs,values,label=method,yerr=[lower,upper],capsize=2,fmt=['o','s','^','D'][offset],color=f'C{offset+1}',ms=5)
        ax.axhline(1,color='black',lw=1);ax.set_xticks(range(len(MODELS)),labels=MODELS,rotation=15)
        ax.set_ylim(.975,1.13);ax.set_title('WT2 test' if split=='test' else 'Preregistered C4 subset');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('PPL ratio / FP16 (lower is better)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'quality_ratios.png');plt.close(fig)
    if any('Hy-F-Q' in r for r in summary['paired_tasks_mix'].values()):
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
        for ax,(task,metric,title) in zip(axes,[('hellaswag','acc_norm','HellaSwag acc_norm'),
                ('arc_challenge','acc_norm','ARC-Challenge acc_norm'),('piqa','acc','PIQA acc')]):
            xs=[];ys=[];lower=[];upper=[]
            for i,model in enumerate(MODELS):
                pair=summary['paired_tasks_mix'].get(model,{}).get('Hy-F-Q',{}).get(task,{}).get(metric)
                if pair is None:continue
                value=100*pair['candidate_minus_reference'];lo,hi=[100*v for v in pair['paired_95']]
                xs.append(i);ys.append(value);lower.append(max(0,value-lo));upper.append(max(0,hi-value))
            ax.errorbar(xs,ys,yerr=[lower,upper],fmt='o',capsize=3,color='C4')
            ax.axhline(0,color='black',lw=1);ax.set_xticks(range(len(MODELS)),labels=MODELS,rotation=25)
            ax.set_title(title);ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Hy-F-Q minus Mix-I (percentage points)')
        fig.suptitle('Full splits; paired document bootstrap 95% intervals, no multiplicity correction')
        fig.tight_layout();fig.savefig(folder/'task_differences.png');plt.close(fig)
    if 'formats' in summary:
        rows=summary['formats']['records'];shapes=sorted({(r['m_out'],r['k_in']) for r in rows})
        fig,axes=plt.subplots(len(shapes),2,figsize=(10,9),sharex=True)
        for ai,(m,k) in enumerate(shapes):
            fp16={r['n_tokens']:r['median_us'] for r in rows if r['m_out']==m and r['k_in']==k and r['path']=='fp16'}
            for col,online in enumerate([False,True]):
                ax=axes[ai,col]
                for name in ['w4a8','w8a8_mx','w4a4']:
                    values=sorted((r['n_tokens'],fp16[r['n_tokens']]/r['median_us']) for r in rows if
                        (r['m_out'],r['k_in'],r['path'],r['online_activation_pack'])==(m,k,name,online))
                    ax.plot([x for x,y in values],[y for x,y in values],'-o',label=name)
                ax.axhline(1,color='gray',ls='--',lw=1);ax.set_xscale('log',base=2);ax.set_xticks([1,8,64],labels=[1,8,64])
                ax.set_title(f'M={m}, K={k}; '+('pack + GEMM' if online else 'GEMM only'));ax.grid(alpha=.2)
                ax.set_ylabel('Speedup / FP16');ax.set_xlabel('Token count')
        axes[0,1].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'formats.png');plt.close(fig)
    if summary['mechanism']:
        fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharey='row')
        for row in range(2):
            for ax,context in zip(axes[row],[128,1024]):
                for color,(method,values) in enumerate(summary['mechanism'].items()):
                    if row==1 and method=='A-serial':continue
                    xs=[b for b in [1,8,32,64] if f'b{b}_c{context}' in values]
                    ax.plot(xs,[100*(values[f'b{b}_c{context}']-1) for b in xs],'-o',label=method,color=f'C{color}')
                ax.axhline(0,color='gray',lw=1);ax.set_xscale('log',base=2);ax.set_xticks([1,8,32,64],labels=[1,8,32,64])
                ax.set_title(f'7B, prompt {context}'+(' (without serial control)' if row else ''));ax.set_xlabel('Batch');ax.grid(alpha=.2)
            axes[row,0].set_ylabel('Extra latency when disabled (%)')
        axes[0,1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/'mechanism.png');plt.close(fig)
    for model,methods in summary['layer_allocation'].items():
        fig,ax=plt.subplots(figsize=(9,3.6))
        for name,rows in methods.items():
            same=methods['Hy-I']==methods['Hy-F-Q']
            if same and name=='Hy-I':continue
            label='Hy-I / Hy-F-Q (same quotas)' if same and name=='Hy-F-Q' else name
            ax.plot([r['layer'] for r in rows],[100*r['high_weight_fraction'] for r in rows],label=label,lw=1.3)
        ax.set_xlabel('Decoder layer');ax.set_ylabel('High-precision weight elements (%)');ax.set_title(model+' allocation at the original payload cap')
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/(model+'_allocation.png'));plt.close(fig)
    profile=summary['reused_stage2_profile']['records'];chosen=[r for r in profile if r['phase']=='final']
    chosen.sort(key=lambda r:(r['batch'],r['kind']))
    fig,ax=plt.subplots(figsize=(10,4));xs=np.arange(len(chosen));bottom=np.zeros(len(chosen))
    for key in ['gemm','activation_quant','fused_producer_quant','output_layout','attention_kv','other_model_gpu','framework_gpu','gpu_idle']:
        values=np.array([r['exclusive_graph_budget_ms'].get(key,0) for r in chosen])
        ax.bar(xs,values,bottom=bottom,label=key);bottom+=values
    ax.set_xticks(xs,labels=[f"B{r['batch']}\n{r['kind']}" for r in chosen]);ax.set_ylabel('Exclusive Graph span (ms)')
    ax.set_title('Reused stage-two 7B development trace, prompt 128');ax.legend(fontsize=8,ncol=3)
    fig.tight_layout();fig.savefig(folder/'profile_decomposition.png');plt.close(fig)


def finding_lines(stage,summary):
    lines=['## 主要发现','']
    if all('Hy-F-Q' in summary['main'].get(m,{}) and 'Mix-I' in summary['main'][m] for m in MODELS):
        rows=summary['main'];q7=rows['7B']['Hy-F-Q']
        lines += [
            f"**性能收益集中在较大 batch。** 7B 最终 Hy-F-Q 相对 Mix-I 的全点速度几何平均为 {q7['speedup_mix']:.4f}×，独立重复组为 {q7.get('repeat_speedup_mix',float('nan')):.4f}×；B1–8 为 {q7['subgroups']['small_1_8']:.4f}×，B64–128 为 {q7['subgroups']['large_64_128']:.4f}×。不能把大 batch 收益外推为全面领先。各模型与各点结果保留在下表及图中。",'',
            '**质量收益存在数据域依赖。** 下表为最终 Hy-F-Q 相对原 Mix-I RTN 的 PPL 百分比变化，负数更好；这些结果没有用于重新选择主方法。','',
            '| 模型 | WT2 PPL 变化 [配对95%] | C4 子集 PPL 变化 [配对95%] |','|---|---:|---:|']
        for model in MODELS:
            fq=rows[model]['Hy-F-Q']['ppl'];mix=rows[model]['Mix-I']['ppl']
            values=[]
            for split in ['test','c4']:
                lo,hi=summary['paired_quality_mix'][model]['Hy-F-Q'][split]['ppl_ratio_95']
                values.append(f"{100*(fq[split]/mix[split]-1):+.3f}% [{100*(lo-1):+.3f}%, {100*(hi-1):+.3f}%]")
            lines.append('| '+model+' | '+' | '.join(values)+' |')
        lines += ['','C4 子集上主方案在五个模型的 PPL 均高于 Mix-I，因而当前结果不支持通用的质量优势。Mistral 的 WT2 点估计虽可低于 FP16，也不据此声称量化普遍提高模型能力。']
    if all(m in summary['selector_quantizer'] for m in ['3B','7B']):
        lines += ['', '**选择与 clipping 存在交互。** 固定 RTN 的 Hy-F−Hy-I 与固定 clipping 的 Hy-F-Q−Hy-I-Q 分开计算。在 3B，浮点选择在 RTN 下使 WT2 PPL 上升，在 clipping 下则降低 WT2 PPL；3B/7B 的 clipping 对照中，浮点选择均改善 WT2、却使 C4 PPL 上升。不能把最终质量变化全部归于选择或全部归于 clipping。']
    if all(len(summary['robustness'].get(m,[]))==6 for m in ['3B','7B']):
        lines += ['', '**校准域迁移进一步复现了这种取舍。** 在两个模型中，C4 训练语料校准均降低 C4 holdout PPL、提高 WT2 PPL。seed、样本数与长度的完整结果一并呈现；没有按最终分数挑选最佳 seed。R-base 是独立重算的全流程控制，与历史主方案的浮点 score/边界通道不保证逐位相同，不能把两者的小差异误称为工程优化改变了量化结果。']
    if all(len(summary['ablation'].get(m,{}))==2 for m in ['3B','7B']):
        gains=[v['kernel_speedup'] for rows in summary['ablation'].values() for v in rows.values()]
        lines += ['', f"**实现提速可与算法质量变化隔离。** 固定 mask 和 RTN 的四组 K1/K0 对照，在各模型共同十点的速度 GM 提升范围为 {100*(min(gains)-1):.2f}%–{100*(max(gains)-1):.2f}%；对应 WT2/C4 PPL 一致。该十点集合与阶段二全十六点汇总不同，百分比不能直接混用。真实缓存 decode 的工程门禁另列。"]
    if summary.get('paired_tasks_mix'):
        lines += ['', '### 下游任务相对 Mix-I 的变化','',
            '| 模型 | HellaSwag acc_norm Δpp [95%] | ARC-Challenge acc_norm Δpp [95%] | PIQA acc Δpp [95%] |',
            '|---|---:|---:|---:|']
        for model in MODELS:
            pairs=summary['paired_tasks_mix'].get(model,{}).get('Hy-F-Q')
            if not pairs:continue
            values=[]
            for task,metric in [('hellaswag','acc_norm'),('arc_challenge','acc_norm'),('piqa','acc')]:
                v=pairs[task][metric];lo,hi=v['paired_95']
                values.append(f"{100*v['candidate_minus_reference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
            lines.append('| '+model+' | '+' | '.join(values)+' |')
        lines += ['', '区间为同任务文档的配对 bootstrap，不作跨任务多重比较校正；不以单个任务的点估计替代总体质量结论。']
        if all('Hy-F-Q' in summary['paired_tasks_mix'].get(model,{}) for model in MODELS):
            pairs={m:summary['paired_tasks_mix'][m]['Hy-F-Q'] for m in MODELS}
            h=[p['hellaswag']['acc_norm'] for p in pairs.values()]
            if all(v['paired_95'][1]<0 for v in h):
                decreases=[-100*v['candidate_minus_reference'] for v in h]
                lines += ['', f"**HellaSwag上的下降延续到独立家族。** 五模型Hy-F-Q相对Mix-I的acc_norm均下降，幅度为{min(decreases):.2f}–{max(decreases):.2f}个百分点，各自未经多重校正的配对区间均低于零。这与C4结果一起限制了WT2收益的泛化解释。"]
            arc_positive=[m for m,p in pairs.items() if p['arc_challenge']['acc_norm']['paired_95'][0]>0]
            if arc_positive:lines += ['', 'ARC的正向配对区间出现在'+ '、'.join(arc_positive)+'；其他模型仍需结合各自区间判断，不将局部收益写成统一排名。']
            if all(p['piqa']['acc']['paired_95'][0]<=0<=p['piqa']['acc']['paired_95'][1] for p in pairs.values()):
                lines += ['', 'PIQA上五模型的配对区间均跨零，本轮不能给出一致优劣判断。主方案保持冻结，未根据这些最终分数改选mask或量化器。']
    if (stage/'acceptance.json').exists():
        a=read(stage/'acceptance.json')
        lines += ['', '### 验收覆盖','',
            f"已登记 {a['expected_jobs']} 个最终作业；普通 decode 速度 {a['speed_cells']} 个可行单元，验证计时 FULL 步数 {a['verified_timed_full_steps']:,}；PPL {len(a['ppl_evaluations'])} 项，下游任务 {len(a['task_evaluations'])} 项，工程对照 {len(a['engineering_gates'])} 项。缺失/失败作业、稳定度与不可行单元的完整列表见 `acceptance.json`。长上下文、非 bucket 和 TTFT 扩展另列，不计入这里的普通 decode 单元数。"]
        if 'sampling_audit' in a:
            lines += ['', '| 计时类别 | 单元数 | 总采样组数 | 只用一组的单元 | 所有尝试组的最大CV |','|---|---:|---:|---:|---:|']
            for kind,r in a['sampling_audit'].items():lines.append(f"| {kind} | {r['cells']} | {r['groups']} | {r['single_group_cells']} | {r['maximum_attempt_cv']:.3%} |")
            lines += ['', '重试单元与逐组CV保存在`acceptance.json:retried_cells`。按原协议采用首个达标组，不按最快时间选组；最终组通过不抹去早期未达标组。扩展类别包含TTFT、长上下文与非bucket。']
        if a['unstable_cells']:
            lines += ['', '**严格稳定度门禁未全部通过。** 下列单元用完原定最多三组采样后仍未达标；全部样本保留，未删除或追加尝试来获得通过结果。它们不用于稳定性能优劣的结论：', '']
            lines += ['- '+ ' / '.join(map(str,row)) for row in a['unstable_cells']]
    if (stage/'completion.json').exists():
        completion=read(stage/'completion.json')
        lines += ['', '最终源码提交：`'+completion['commit']+'`。冻结后的执行源码与二进制保持原身份；最终分析源码单独归档在 `final_analysis_source/`。']
    return lines+['']


def make_report(stage,summary,results):
    accepted=(stage/'acceptance.json').exists() and read(stage/'acceptance.json')['passed']
    queue_file=stage/'queues/final/status.json'
    finished=queue_file.exists() and read(queue_file)['phase']=='complete'
    title='# 阶段三：冻结后的最终评测' if accepted else '# 阶段三：冻结后的最终评测（含未通过项目）' if finished else '# 阶段三：冻结后的评测（进行中）'
    lines=[title,'',
        '状态：完整验收通过。' if accepted else '状态：全部预登记作业执行完毕，严格验收未全部通过；未通过项目与原始样本完整保留。' if finished else '状态：部分结果已生成，预登记队列仍在执行；这不是最终验收报告。','',
        '此报告由已完成结果生成；是否全部验收以 `acceptance.json` 为准。原始逐点、逐块和逐任务样本均保留。',
        '', '## 实现与评测边界','',
        '主方法为 FP16、Mix-I（原 INT4/INT8 device + RTN）、Hy-I（整数选择 + RTN）、Hy-F（浮点选择 + RTN）、Hy-F-Q（浮点选择 + clipping）。Mix-I 不是论文式 GPTQ/clipping 的质量最优复现。',
        '', 'F不是五模型统一的配额规则：Qwen 0.5B沿用阶段一选出的clipping-aware absolute loss proxy及层间自由分配；Qwen 1.5B/3B/7B沿用Q1 signed RTN promotion proxy与原逐投影高精度通道数量。候选在阶段一按完整WT2 validation选择，主校准为seed2029、32×512训练窗口；最终WT2/C4/任务分数不参与重选。Mistral在观察自身质量分数之前固定采用较大模型的signed RTN、固定逐投影配额规则。score是教师梯度构造的近似收益，不是真实重跑模型后的精确loss差。方法选择历史见[阶段一报告](../../../01_quality/runs/20261025_211540_q2_q3_allocation/report.md)。',
        '', 'Q分支仅离线改变低精度FP4组的code/scale：每K32组比较原E8M0指数及-1/-2偏移，以输入量化后二阶矩加权的平方重构误差选择，目标相同时保留原候选。这是对角近似，不是完整输出重构或GPTQ；高精度FP8权重与在线A8规则不变，不增加scale数量或在线算子。精确实现见`integrations/vllm/allocation/build_banks.py`及`docs/floating_allocation_quantizer.md`。',
        '', '```mermaid', 'flowchart LR',
        '  W[原始权重] --> S[离线通道选择与量化]',
        '  S --> W4[低精度权重 MXFP4 + scales]',
        '  S --> W8[高精度权重 MXFP8 + scales]',
        '  X[FP16 输入] --> Q[在线 A8 pack；down 分支可融合 SiLU]',
        '  W4 --> G4[FP4 × FP8 GEMM]', '  Q --> G4',
        '  W8 --> G8[FP8 × FP8 GEMM]', '  Q --> G8',
        '  G4 --> Y[按原输出通道 scatter 到同一 FP16 输出]',
        '  G8 --> Y', '```',
        '', '两个 GEMM 按冻结 plan 并发或串行，共享同一份量化激活；两组输出通道互不重叠，无需求和归约。全局权重保持 packed FP4，当前 K32 混合路径的 shared/operand 使用 byte container。',
        '', '主 decode 固定 FULL Graph、Inductor mode0、TP1、FP16 KV、10 GiB KV、129 个生成 token；计时为纯固定 batch decode 的 CUDA event span，包含在线 pack、输出布局和引擎间隙。每组 1 warmup + 5 samples，CV ≤ 5%，最多 3 组。',
        '', '质量使用完整 WT2 test 非重叠 2048-token 块、预登记 C4 子集 256 块及 HellaSwag/ARC-Challenge/PIQA 全 split。WT2 test 在历史阶段看过，不能称盲测。任务为0-shot、无chat template、无额外BOS，以单序列设备 batch 求分，避免训练样本上发现的 batching 差异；该任务口径不用于 decode 速度。任务输入上限2048 tokens，冻结token清单中五模型三个任务均无上下文截断；长度和请求hash见`data/<model>/manifest.json`。',
        '', 'Mistral B128/prompt1024 超过共同 10 GiB KV 容量，五方法一致标记不可行，不纳入速度均值。协议、环境、源码、模型、tokenizer、mask、bank、plans 与二进制 hash 见 `freeze/v1/manifest.json` 和 `file_ledger.json`。',
        '', '## 主矩阵','',
        '| 模型 | 方法 | 对 FP16 速度 GM | 对 Mix 速度 GM | WT2 PPL | C4 子集 PPL | 稳定点/可行点 |',
        '|---|---|---:|---:|---:|---:|---:|']
    # Interpretation is derived only after measurements; it never selects a
    # method, changes a gate, or alters the immutable execution registry.
    insertion=lines.index('## 实现与评测边界')
    lines[insertion:insertion]=finding_lines(stage,summary)
    for model,rows in summary['main'].items():
        for method,r in rows.items():lines.append(f"| {model} | {method} | {r['speedup_fp16']:.4f} | {r['speedup_mix']:.4f} | {r['ppl']['test']:.5f} | {r['ppl']['c4']:.5f} | {r['speed_cells']-r['unstable']}/{r['speed_cells']} |")
    lines += ['', '| 模型 | Hy-F-Q/Mix B1–8 | B16–32 | B64–128 | 独立重复组全点 GM |','|---|---:|---:|---:|---:|']
    for model,rows in summary['main'].items():
        if 'Hy-F-Q' in rows:
            r=rows['Hy-F-Q'];v=r['subgroups'];repeat=r.get('repeat_speedup_mix')
            repeat_text=f'{repeat:.4f}' if repeat is not None else 'pending'
            lines.append(f"| {model} | {v['small_1_8']:.4f} | {v['medium_16_32']:.4f} | {v['large_64_128']:.4f} | {repeat_text} |")
    lines+=['','速度 GM 在同一模型的共同可行点计算。小/中/大 batch、独立重复组及配对 PPL 比率置信区间见 `analysis/summary.json`。不同模型的原始 PPL 不直接平均。PPL区间按2048-token块配对重采样，是固定语料上的描述性区间，未建模相邻块或同文档相关性，也不作多重比较校正；不将其当作全C4总体显著性证明。','', '## 下游任务','',
        '| 模型 | 方法 | HellaSwag acc_norm | ARC-Challenge acc_norm | PIQA acc |','|---|---|---:|---:|---:|']
    for model in MODELS:
        rows=summary['tasks'].get(model,{})
        for method in METHODS+['Hy-I-Q','FP8-native']:
            if method not in rows:continue
            r=rows[method]
            lines.append(f"| {model} | {method} | {r['hellaswag']['metrics']['acc_norm']:.4f} | {r['arc_challenge']['metrics']['acc_norm']:.4f} | {r['piqa']['metrics']['acc']:.4f} |")
    lines+=['','两种 acc 指标、逐样本分数和 1000 次配对 document bootstrap 区间均保存；此处只选任务常用指标展示。', '', '## 选择与实现的 2×2 消融','',
        '| 模型 | 固定 mask | K1/K0 速度 GM | K0 WT2 | K1 WT2 | K0 C4 | K1 C4 |','|---|---|---:|---:|---:|---:|---:|']
    for model,rows in summary['ablation'].items():
        for mask,r in rows.items():lines.append(f"| {model} | {mask} | {r['kernel_speedup']:.4f} | {r['old_ppl']['test']:.5f} | {r['new_ppl']['test']:.5f} | {r['old_ppl']['c4']:.5f} | {r['new_ppl']['c4']:.5f} |")
    lines += ['', '### 真正缓存 decode 的质量','', '| 模型 | 参考 → 候选 | 共同目标观测数 | 平均 NLL 差 | 小样本 KL | top-1 相同/记录数 | 独特采样历史 |','|---|---|---:|---:|---:|---:|---:|']
    for model,rows in summary['decode_comparisons'].items():
        for name,r in rows.items():lines.append(f"| {model} | {name} | {r['targets']} | {r['mean_nll_delta']:.6f} | {r['mean_kl']:.6f} | {r['top1_agree_records']}/{r['full_vocabulary_records']} | {r['unique_sampled_histories']} |")
    lines += ['', '以上是固定网格上的配对诊断。不同batch复用prompt和真实续写，目标观测并非全部独立；默认40个全词表记录对应8个独特token历史在5个batch下重复测量。KL/top-1仅作这些位置的诊断，不据此推断总体任务质量。工程对照保持原误差门禁，不要求不同量化算法输出相同。']
    lines+=['','该 2×2 固定 RTN。Hy-F 与 Hy-F-Q 另外隔离 clipping 的贡献，不能将其收益归于通道选择。真实缓存 decode 的 paired NLL、KL、top-1 及工程误差门禁另见 `comparisons/`；KL 仅覆盖预登记的少量全词表采样，不等于全部目标。',
        '', '### 选择与 clipping 的交互对照','', '| 模型 | 方法 | WT2 | C4 | 共同六点延迟 GM ms |','|---|---|---:|---:|---:|']
    for model,value in summary['selector_quantizer'].items():
        for method,r in value['methods'].items():lines.append(f"| {model} | {method} | {r['ppl']['test']:.5f} | {r['ppl']['c4']:.5f} | {r['six_cell_gm_ms']:.4f} |")
    lines += ['', 'Hy-I-Q 保留 I 的 mask/compute plan，只切换为与 Hy-F-Q 相同的 clipping bank。固定 RTN 与固定 clipping 的 F−I 配对 NLL 差异及交互项 `(FQ−IQ)−(F−I)` 见 `summary.json:selector_quantizer`；负值表示浮点选择在对应量化器下降低 NLL。Hy-I-Q 是预登记消融，不按最终结果替换主方案。']
    if summary['selector_quantizer_tasks']:
        lines += ['', '### 任务上的选择与clipping交互','',
            '| 模型 | 任务/指标 | F−I RTN Δpp [95%] | FQ−IQ clipped Δpp [95%] | 交互项 Δpp [95%] |',
            '|---|---|---:|---:|---:|']
        for model,rows in summary['selector_quantizer_tasks'].items():
            for task,r in rows.items():
                cells=[]
                for key in ['F_minus_I_RTN','F_minus_I_clipped','interaction']:
                    v=r[key];lo,hi=v['paired_95'];cells.append(f"{100*v['candidate_minus_reference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
                lines.append('| '+model+' | '+task+'/'+r['metric']+' | '+' | '.join(cells)+' |')
        lines += ['', '这里是准确率百分点差，正数更好，与上方NLL差的方向相反。交互项仍为`(FQ−IQ)−(F−I)`；各列使用相同文档的配对bootstrap，不作多重比较校正。完整四格均保留，不只展示最有利的对照。']
    lines += ['', '## 预算与校准泛化','']
    for category,title in [('budget','预算曲线'),('robustness','全过程校准变化')]:
        lines += ['### '+title,'','| 模型 | 方法 | payload bit/w | 含 scale/index/padding bit/w | WT2 | C4 | 共同六点延迟 GM ms |','|---|---|---:|---:|---:|---:|---:|']
        for model,rows in summary[category].items():
            for r in rows:lines.append(f"| {model} | {r['method']} | {r['payload_bit_per_weight']:.5f} | {r['physical_bit_per_weight']:.5f} | {r['ppl']['test']:.5f} | {r['ppl']['c4']:.5f} | {r['decode_gm_ms_common6']:.4f} |")
        lines+=['']
    lines+=['payload 只针对量化 projection 的 weight code，不包括 scale、index、padding、未量化模块、KV 或 Graph pool。W8A8 是超预算端点，在表中报告，不压缩预算内曲线的坐标轴。随机选择和所有校准设置均报告，不挑最佳 seed。预算曲线使用相同 clipping bank，不能与主矩阵的 RTN 选择消融混为一谈。观察到的三维非支配点见 `summary.json:observed_pareto`，其逐点支配关系不自动等于统计显著性。',
        '', '预算曲线的float使用冻结Q1 signed score和按alpha缩放的固定逐层配额；integer采用成本受限分配，uniform采用其自身的层间分配规则。因此曲线比较完整分配策略，不是只替换score的控制试验。`B-integer-a100`不等于原I mask或Hy-I-Q；浮点alpha=1则在冻结前断言与主Hy-F-Q完全相同，并复用该主点。残余预算填充可使相邻预算点的mask不严格嵌套。',
        '', '### 普通 vLLM FP8 辅助基线','', '| 模型 | 对 FP16 速度 GM | WT2 | C4 | 实际 backend |','|---|---:|---:|---:|---|']
    for model,r in summary['optional_fp8'].items():lines.append(f"| {model} | {r['speedup_fp16']:.4f} | {r['ppl']['test']:.5f} | {r['ppl']['c4']:.5f} | {r['backend_counts']} |")
    lines += ['', '### 主方案预算账本','',
        '| 模型 | 选择 | 高精度权重元素比例 | code bit/w | 含 scale/index/padding bit/w | code GB | scale GB | index GB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model,rows in summary['primary_budget'].items():
        for method,r in rows.items():lines.append(f"| {model} | {method} | {100*r['high_precision_weight_fraction']:.3f}% | {r['payload_bit_per_weight']:.6f} | {r['physical_bit_per_weight']:.6f} | {r['weight_bytes']/1e9:.6f} | {r['weight_scale_bytes']/1e9:.6f} | {r['channel_index_bytes']/1e9:.6f} |")
    lines += ['', 'Hy-F 与 Hy-F-Q 的 mask 和存储格式相同；Mix-I 与 Hy-I 共用 I mask，因此 weight-code bits 相同，但量化 metadata 和物理总字节不同。表中 Hy-I 的物理存储不能当成 Mix-I 的物理存储。两者实测常驻 tensor storage 见下一表。','',
        '### 显存口径','', '| 模型 | 方法 | 模型 tensor storage GB | 量化 runtime tensor GB | Graph pool reserved MiB | Torch peak allocated GB |','|---|---|---:|---:|---:|---:|']
    for model,rows in summary['memory'].items():
        for method,r in rows.items():lines.append(f"| {model} | {method} | {r['model_representation_bytes']/1e9:.4f} | {r['quantized_runtime_tensor_storage_bytes']/1e9:.4f} | {(r['graph_pool_reserved_bytes'] or 0)/2**20:.2f} | {r['torch_peak_allocated_bytes']/1e9:.4f} |")
    lines += ['', '### 7B 工作负载扩展','', '| 协议 | 方法 | batch | prompt | 实际 Graph tokens | 中位时间 ms | 单 token 请求组 wall ms | CV | 稳定度门禁 |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for op,methods in summary['workloads'].items():
        for method,rows in methods.items():
            for r in rows:
                if r.get('feasible',True):
                    graph='—' if op=='prefill' else str(r['attempts'][-1]['samples'][0]['first_pure_decode_graph_descriptor']['num_tokens'])
                    wall=f"{r['median_request_group_wall_ms']:.4f}" if op=='prefill' else '—'
                    stable='通过' if r['stable'] else '**未通过**'
                    lines.append(f"| {op} | {method} | {r['batch']} | {r['context']} | {graph} | {r.get('median_ttft_ms',r.get('median_ms_per_step')):.4f} | {wall} | {r['cv']:.4%} | {stable} |")
    lines += ['', 'prefill 行为 TTFT；long/nonbucket 行为每 decode step 时间，两种单位含义不混排比较。真实 Graph descriptor 和所有原始样本在相应 result 文件。',
        '', '### 准备成本','', '| 专用启动方法 | 加载与 capture s | 被计量的量化投影 | 量化/layout host s | CUDA event span s |','|---|---:|---:|---:|---:|']
    for method,r in summary['startup'].items():lines.append(f"| {method} | {r['load_capture_s']:.3f} | {r['quantized_projections']} | {r['quant_layout_host_s']:.3f} | {r['event_span_s_including_host_gaps']:.3f} |")
    lines += ['', '当前部署仍从 FP16 checkpoint 加载，并在载入时生成布局、导入 FP4 bank；这些数字不代表已优化的序列化量化 checkpoint 启动时间。专用计时逐层同步，会影响加载重叠，仅作准备成本描述，不当作无扰动的启动性能排名。',
        '', '| 模型 | 校准设置 | 总准备 s | score s | teacher peak GB | mask Jaccard 对主方案 |','|---|---|---:|---:|---:|---:|']
    for model,settings in summary['offline_calibration'].items():
        for name,r in settings.items():lines.append(f"| {model} | {name} | {r['elapsed_s']:.2f} | {r['teacher_score_elapsed_s']:.2f} | {r['teacher_peak_gpu_bytes']/1e9:.3f} | {r['mask_jaccard_vs_primary']:.5f} |")
    lines += [
        '', '## 机制、工作负载与内存','',
        '逐项恢复旧 pack 选择、关闭 producer、统一两路串行调度和关闭轻量高精度路径的逐点时延比例见 `analysis/summary.json:mechanism`。优化影响可交互，不相加百分比。预登记长 prompt、非 capture bucket、prefill/TTFT 分别保存在 `results/7B/*/workload_*`；TTFT 使用引擎时间戳，不冒称孤立 GPU prefill 时间。',
        '', '消融名称按实际操作解释：`A-no-pack` 将 PACK_MODE=3 改为0，即 T≥16 的向量化 pack 恢复为旧选择，仍保留旧实现已有的 padding 初始化融合；T<16 本来就选择mode0，因此B1/8不是该向量化优化的有效处理组。`A-no-producer` 关闭精确SiLU+pack融合；`A-serial` 将全部计划统一为serial，可能同时改变原执行顺序；`A-no-light-high` 仅将high_id≥8替换为0。名称中的no-pack不表示省略激活量化，也不表示撤销所有历史pack优化。两路并发在旧双kernel设计中已存在；串行消融解释当前设计依赖，不能归入阶段二新增收益。',
        '', '去重模型 tensor storage、allocator pool、峰值和设备总占用见 `analysis/summary.json:memory`，它们是不同统计口径，不能直接相加。离线校准的完整过程记录于 `robustness/*/complete.json`；专用启动实验给出加载/capture 与量化/layout 时间，CUDA event span 包含 host 间隙和 bank I/O。',
        '', '格式微基准见 `formats/summary.json`：这是3A冻结前完成并随输入账本固定的独立硬件实验，不属于最终模型holdout评分。比较FP16、MX-W4A8、MX-W8A8、MX-W4A4；分开预打包 kernel-only 与 reference activation pack+GEMM，重复固定 device buffers 做 Graph replay。L2 为 96 MiB，各格式的 weight payload 可能处在不同缓存容量区间，不能据此单独推导 MMA 指令效率。输入数值可精确表示，因此不推导模型量化质量；参考 pack 也不是格式专属最优实现。SASS 与资源见 `environment/`。',
        '', '原 Mix INT4 packed 权重在软件中展开为 INT8 MMA operands；浮点 K32 W4A8 使用 native 混合 MMA 和 byte container。它省掉软件 unpack 指令链，但不能直接推出 shared/register 减半或端到端必胜。当前 W4A4 的 K64 packed 行为和低精度 activation 与 W4A8 不同。阶段二已有分项 Nsight 证据，本轮以冻结后的机制关闭实验验证实际收益；没有硬件 counter 权限，不推断未测 stall 原因。',
        '', '| 成本项 | 本轮能够确认的边界 |','|---|---|',
        '| Mix软件unpack、scale/zero处理、GEMM epilogue | 源码/SASS可识别；时间合并在原Mix GEMM中，未得到各自独立的时间份额 |',
        '| Hy block-scale处理、MMA及scatter epilogue | 时间合并在Hy GEMM中，scatter不是额外copy kernel |',
        '| 独立activation pack、producer、输出布局copy | 由节点trace分类，另有冻结机制关闭实验；producer同时包含SiLU与pack |',
        '', '因此没有把GEMM总时间差直接解释为unpack时间，也没有将静态指令数换算成已测cycle占比。更细的Mix内部归因仍未完成；原Mix device保持不动，硬件counter权限限制也未改变。',
        '', '分项图明确复用阶段二开发集 trace，已核对相同7B mask、bank、逐层compute plan、runtime选项及native库，不冒称新的C4测量。GEMM按并发区间的并集计并包含scatter；fused producer包含SiLU和pack。Graph外还包含LM head等GPU工作，不能把Graph外总时长全部称为框架开销。图中的trace时长只用于分解，性能排名采用本轮无profiler计时。',
        '', '### 未进入最终实现的路线','',
        '下表是[阶段二开发实验](../../../02_performance/runs/20261026_015237_stage2_performance/report.md)的决策记录，未作为本轮新增的最终测试数据。', '',
        '| 路线 | 已观察到的限制 | 本轮处理 |','|---|---|---|',
        '| SIMT小batch GEMV | 有速度潜力，但改变累加顺序，未通过既定decode数值等价门禁；不等于已证明PPL劣化 | 未进入冻结主方案 |',
        '| 近似exp producer | 未通过数值/字节一致性检查 | 仅采用保留vLLM舍入与signed-zero行为的math0 |',
        '| 分区内reverse/block_reverse | 微基准收益未延续到整模型 | 未采用 |',
        '| 单kernel TMA精度切换 | 数值通过，但阶段二7B十四点约0.9977×，没有速度收益 | 不进入最终双kernel方案 |',
        '| M64 TMA构建探针 | 当前CUTLASS布局等价性静态断言失败 | 记为实现约束，不称硬件不支持 |',
        '', '## 复现与限制','',
        '执行顺序：`preflight.py` → `campaign.py` → `freeze.py` → `serial_jobs.py --jobs final_jobs.json` → `accept.py` / `report.py`。完整命令保存在冻结的 `final_jobs.json`。源码在本里程碑提交，权重/bank/构建产物保存在远端 tmp，哈希账本随结果归档。',
        '', '普通 FP8 是否纳入由冻结前 native 冒烟决定，实际 backend 保存在 runtime audit。当前 vLLM dense MXFP4 入口退回非量化层；NVFP4/MXFP8 序列化 checkpoint、AWQ/GPTQ 与 Mix 论文 PTQ 未作为主基线，具体 installed-source 证据见 `baseline_compatibility.json`。',
        '', '限制：单卡 RTX5090/SM120、TP1、FP16 KV、固定纯 batch decode；C4 是固定子集；所有性能差异应结合逐点稳定度和重复组。全过程校准变化在 3B/7B 进行，不能推出所有模型/语料都稳健。不同 PPL 的方案只比较测得 Pareto 关系，不称等质量加速。','']
    for model in MODELS:
        if (stage/'analysis/figures'/(model+'_decode.png')).exists():lines += [f'![{model} decode](analysis/figures/{model}_decode.png)','']
    for model in ['3B','7B']:
        if (stage/'analysis/figures'/(model+'_budget.png')).exists():lines += [f'![{model} budget](analysis/figures/{model}_budget.png)','']
        if (stage/'analysis/figures'/(model+'_budget_storage.png')).exists():lines += [f'![{model} storage and latency](analysis/figures/{model}_budget_storage.png)','']
    if (stage/'analysis/figures/formats.png').exists():lines += ['![Formats](analysis/figures/formats.png)','']
    if (stage/'analysis/figures/mechanism.png').exists():lines += ['![Mechanism](analysis/figures/mechanism.png)','']
    if (stage/'analysis/figures/relative_speed.png').exists():lines += ['![Relative speed with independent repeat](analysis/figures/relative_speed.png)','']
    if (stage/'analysis/figures/quality_ratios.png').exists():lines += ['![Paired PPL ratios](analysis/figures/quality_ratios.png)','']
    if (stage/'analysis/figures/task_differences.png').exists():lines += ['![Paired downstream task differences](analysis/figures/task_differences.png)','']
    if (stage/'analysis/figures/profile_decomposition.png').exists():lines += ['![Reused development profile](analysis/figures/profile_decomposition.png)','']
    for model in ['3B','7B']:
        if (stage/'analysis/figures'/(model+'_allocation.png')).exists():lines += [f'![{model} allocation](analysis/figures/{model}_allocation.png)','']
    (stage/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
