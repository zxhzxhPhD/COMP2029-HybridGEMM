"""Compact read-only experiment summary, including incomplete model screens."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage
def read(p):return json.loads(p.read_text())
for name in ['round1_status.json','round2_status.json','round3_status.json']:
    if (stage/name).exists():print(name,read(stage/name).get('phase'))
for model in ['7B','0.5B','1.5B','3B']:
    folder=stage/model
    if not folder.exists():continue
    basefile=folder/'hy_pre_f_inherited/result.json'
    base={(r['batch'],r['context']):r['median_ms_per_step'] for r in read(basefile)['cells']} if basefile.exists() else {}
    for file in sorted(folder.glob('*/result.json')):
        value=read(file);rows=value['cells'];ratios=[base[(r['batch'],r['context'])]/r['median_ms_per_step'] for r in rows if (r['batch'],r['context']) in base]
        if not ratios:continue
        print(model,file.parent.name,'cells',len(rows),'GM',round(math.exp(sum(map(math.log,ratios))/len(ratios)),6),
              'max_regression',round(1/min(ratios)-1,6),'ms',[round(r['median_ms_per_step'],4) for r in rows])
    for file in sorted(folder.glob('quality*/quality.json')):
        value=read(file);print(model,file.parent.name,value.get('comparison','baseline'))
for name in ['producer.json','variants.json','pack_policy.json']:
    file=stage/'checks'/name
    if file.exists():
        val=read(file);print(name,val.get('passed'),val.get('passing_math_modes'))
        if name=='producer.json':
            print('producer differences',[(m,*[sum(r[k] for r in val['records'] if r['math']==m)
                for k in ['half_differences','q_differences','sf_differences']]) for m in range(3)])
file=stage/'probes/results.json'
if file.exists():
    groups={}
    for r in read(file):
        if 'baseline' not in r:continue
        c=r['candidate'];speed=r['baseline']['us']/c.get('us',c.get('median_us'))
        groups.setdefault(r['probe'],[]).append((r['shape'],r.get('order',r.get('variant')),round(speed,4)))
    for key,vals in groups.items():print(key,'best',sorted(vals,key=lambda x:x[2],reverse=True)[:6])
