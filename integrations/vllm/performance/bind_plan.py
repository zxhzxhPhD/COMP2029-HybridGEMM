"""Materialize per-layer old/new choices and bind them to frozen model artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare import read,write,sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model',required=True)
ap.add_argument('--proposal',type=Path);ap.add_argument('--buckets',default='')
ap.add_argument('--library',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();meta=read(args.stage/'inputs.json')[args.model];root=Path(meta['source_root'])
view=Path(meta['shape_manifest']).parent;spec=read(view/'view.json');selection=read(view/'selection.json')
oldcases={r['layer']:r for r in read(root/'reference_cases.json')['layers']}
oldplans=read(root/'reference_plans.json')['hygemm']
proposals=read(args.proposal)['hygemm'] if args.proposal else []
buckets=set(map(int,filter(None,args.buckets.split(','))))
layers=[];unique={}
for case in spec['layers']:
    for t in [1,2,4,8,16,32,64,128]:
        if t in buckets:
            rows=[r for r in proposals if (r['out'],r['k'],r['high'])==(case['out'],case['k'],case['high'])]
        else:
            rows=[r for r in oldplans if (r['out'],r['k'],r['high'])==(case['out'],case['k'],oldcases[case['layer']]['high'])]
        chosen=next(r for r in sorted(rows,key=lambda x:x['tokens']) if r['tokens']>=t)
        rec={k:case[k] for k in ['layer','out','k','high']}
        rec.update(tokens=t,**{k:chosen[k] for k in ['low_id','high_id','policy']})
        layers.append(rec);unique.setdefault((rec['out'],rec['k'],rec['high'],t),{k:v for k,v in rec.items() if k!='layer'})
binding={'library_sha256':sha(args.library),'selection_canonical_sha256':hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'selection_file_sha256':sha(view/'selection.json'),'bank_manifest_sha256':sha(spec['bank']),
    'shape_manifest_sha256':sha(view/'view.json'),'layout':'MX K32 E8M0 native physical scales v1',
    'bucket_rule':'smallest covering bucket; per-layer plans take precedence; >128 original inherited plan',
    'proposal_sha256':sha(args.proposal) if args.proposal else None,'proposal_buckets':sorted(buckets),
    'reference_plans_sha256':sha(root/'reference_plans.json')}
write(args.out,{'binding':binding,'hygemm':list(unique.values()),'layer_plans':layers})
