"""Standalone research plots from the audited native validation summary."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
summary=json.loads((args.out/'summary.json').read_text());models=list(summary['models'])
folder=args.out/'figures';folder.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
titles=['Weight MSE benefit','Output MSE benefit','Absolute loss proxy','Signed loss proxy']
for ax,family,title in zip(axes.flat,['weight_gain','output_gain','loss_abs_gain','loss_signed_gain'],titles):
    x=np.arange(len(models))
    for offset,stage,color in [(-.2,'q1','#5586b8'),(.2,'q2','#d88142')]:
        y=[summary['models'][m]['rows'][stage+'_'+family]['vs_q1_original']['relative_ppl_percent'] for m in models]
        bars=ax.bar(x+offset,y,.38,label='Q1: fixed quotas' if stage=='q1' else 'Q2: free allocation',color=color)
        ax.bar_label(bars,labels=[f'{v:+.2f}' for v in y],fontsize=8,padding=3)
    ax.axhline(0,color='#555555',linewidth=.8);ax.set_xticks(x,models);ax.set_title(title)
    ax.set_ylabel('PPL change vs original Hy (%)');ax.margins(y=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Q1 benefits do not necessarily survive cross-layer allocation\nWT2 validation, seed2029, matched weight-code budget')
for ext in ['png','svg']:fig.savefig(folder/f'q1_q2_comparison.{ext}',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');x=np.arange(len(models))
for i,(label,color) in enumerate([('Original Hy','#b0b4ba'),('Mix-I RTN','#5586b8'),('Best Q2','#d88142'),('Selected Q1/Q2/Q3','#488b62')]):
    y=[]
    for model in models:
        data=summary['models'][model];rows=data['rows'];fp=rows['q1_reference_fp16']['ppl']
        key={'Original Hy':'q1_original','Mix-I RTN':'q1_reference_mixllm','Selected Q1/Q2/Q3':data['primary_winner']}.get(label)
        if key is None:key=min(['q2_'+n for n in ['integer','weight_gain','output_gain','loss_abs_gain','loss_signed_gain','uniform_abs']],key=lambda k:rows[k]['ppl'])
        y.append((rows[key]['ppl']/fp-1)*100)
    bars=ax.bar(x+(i-1.5)*.21,y,.2,label=label,color=color);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
ax.axhline(0,color='#555555',linewidth=.8);ax.set_xticks(x,models);ax.set_ylabel('PPL change vs FP16 (%)')
ax.set_title('Quality-stage validation choices (not final test results)');ax.margins(y=.2)
ax.legend(ncols=2,fontsize=9)
for ext in ['png','svg']:fig.savefig(folder/f'selected_quality.{ext}',dpi=180)
plt.close(fig)
diagnostics=json.loads((args.out/'allocation_diagnostics.json').read_text())['7B']
groups=list(diagnostics['loss_abs_gain']['groups']);x=np.arange(len(groups))
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
for i,(label,family,key,color) in enumerate([
        ('Q1 quotas','loss_abs_gain','old_high_weights','#b0b4ba'),
        ('Q2 absolute','loss_abs_gain','new_high_weights','#5586b8'),
        ('Q2 signed','loss_signed_gain','new_high_weights','#d88142')]):
    values=[100*diagnostics[family]['groups'][g][key]/diagnostics[family]['groups'][g]['weights'] for g in groups]
    bars=ax.bar(x+(i-1)*.25,values,.24,label=label,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
ax.set_xticks(x,[g.split('.')[-1].replace('_proj','') for g in groups]);ax.set_ylim(0,115)
ax.set_ylabel('FP8 weight share within projection type (%)');ax.set_title('7B: quota movement at the same total weight-code budget')
ax.legend(ncols=3,loc='upper center',fontsize=9)
for ext in ['png','svg']:fig.savefig(folder/f'allocation_7b.{ext}',dpi=180)
print('Saved figures in',folder)
