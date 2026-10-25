"""CPU-only profile accounting and compiler inspection of stage-two artifacts."""
import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parents[1];profile=stage/'profile_initial'
records=[]
for b in [1,8,64]:
    for kind in ['hygemm','mixllm']:
        output=profile/f'analysis_{kind}_b{b}.json'
        if not output.exists():
            subprocess.run([sys.executable,str(src/'profiling/analyze_trace.py'),'--trace',str(profile/f'{kind}_b{b}.sqlite'),
                            '--out',str(output)],stdout=subprocess.DEVNULL,check=True)
        analysis=json.loads(output.read_text());control=json.loads((profile/f'control_{kind}.json').read_text())
        meter=json.loads((profile/f'meter_{kind}.json').read_text())
        c=next(r for r in control['cells'] if r['batch']==b);m=next(r for r in meter['cells'] if r['batch']==b)
        records.append({'kind':kind,'batch':b,'unprofiled_ms':c['median_ms_per_step'],'unprofiled_cv':c['cv'],
            'graph_span_ms':analysis['summary']['graph_span_mean_ms'] if 'summary' in analysis else analysis.get('graph_span_mean_ms'),
            'analysis_file':str(output),'meter':m})
(profile/'summary.json').write_text(json.dumps(records,indent=2))
library=stage/'build/libhygemm_serving.so'
sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(library)],text=True)
with gzip.open(stage/'build/sass.txt.gz','wt') as f:f.write(sass)
functions=[]
for chunk in re.split(r'Function\s*:\s*',sass)[1:]:
    name=chunk.splitlines()[0].strip()
    if 'light' not in name and 'gemv' not in name and 'pack_mx' not in name:continue
    instructions=re.findall(r'/\*[0-9a-f]+\*/\s*(?:@[!A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)',chunk)
    functions.append({'symbol':name,'static_instruction_counts':dict(Counter(instructions))})
(stage/'build/sass_counts.json').write_text(json.dumps(functions,indent=2))
