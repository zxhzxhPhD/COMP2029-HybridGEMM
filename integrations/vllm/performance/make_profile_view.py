"""Build an isolated profiling view of the accepted model and Mix baseline."""
import argparse
from pathlib import Path
import shutil
from prepare import read,write
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--mix',default='mix_original');args=ap.parse_args()
stage=args.stage;folder=stage/'7B';out=stage/'profile_final_view';out.mkdir(exist_ok=True)
for kind,source in [('hygemm',folder/'combination_full/model'),('mixllm',folder/args.mix/'model')]:
    target=out/'models'/kind;target.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        dest=target/p.name
        if p.name=='config.json':shutil.copy2(p,dest)
        elif p.is_file() and not dest.exists():dest.symlink_to(p.resolve())
(out/'calibration').mkdir(exist_ok=True)
token=out/'calibration/tokens.pt'
if not token.exists():token.symlink_to((folder/'calibration/tokens.pt').resolve())
write(out/'plans.json',{'hygemm':read(folder/'combination_plans.json'),
    'mixllm':read(folder/'tune_mix/plans.json') if args.mix=='mix_tuned' else read(folder/'plans.json').get('mixllm')})
