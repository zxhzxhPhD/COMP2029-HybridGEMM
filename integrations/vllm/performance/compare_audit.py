import argparse
import json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);args=ap.parse_args()
a=json.loads(args.old.read_text());b=json.loads(args.new.read_text());diffs=[]
for x,y in zip(a['audit'],b['audit']):
    assert x['name']==y['name']
    for t,p in x['plans_used'].items():
        q=y['plans_used'].get(t)
        if q is not None and p!=q:diffs.append({'layer':x['name'],'tokens':t,'old':p,'new':q})
print('Plan differences',len(diffs));print(json.dumps(diffs[:50],indent=2))
