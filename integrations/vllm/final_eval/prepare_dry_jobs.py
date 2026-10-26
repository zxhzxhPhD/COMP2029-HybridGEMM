import argparse
from pathlib import Path
from final_common import *
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
jobs=[]
for model,methods in read(args.stage/'methods.json').items():
    for method in methods:
        jobs.append({'id':model+'_'+method,'command':[PYTHON,str(Path(__file__).with_name('native_run.py')),
            '--stage',str(args.stage),'--model',model,'--method',method,'--mode','dry']})
write(args.stage/'dry_jobs.json',jobs)
