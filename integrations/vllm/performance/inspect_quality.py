import argparse
import json
import math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);args=ap.parse_args()
old=json.loads(args.old.read_text());new=json.loads(args.new.read_text());rows=[]
for a,b in zip(new['records'],old['records']):
    delta=[];signed=[]
    for i,(x,y) in enumerate(zip(a['logprobs'],b['logprobs'])):
        indices=sorted(set(a['full_target_indices'][i])&set(b['full_target_indices'][i]))[:32]
        for j in indices:delta.append(abs(x[j]-y[j]));signed.append(y[j]-x[j])
    rows.append({'batch':a['batch'],'context':a['context'],'targets':len(delta),'max':max(delta),
                 'mean_abs':math.fsum(delta)/len(delta),'nll_delta':math.fsum(signed)/len(signed),
                 'exact':sum(x==0 for x in delta)})
print(json.dumps(rows,indent=2))
