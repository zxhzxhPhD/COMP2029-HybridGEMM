"""Paired-scheduler controls replace attribution from unpaired prefill runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args()
stage=args.stage;src=Path(__file__).resolve().parent;folder=stage/'7B'
old='/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so';new=str(stage/'build/libhygemm_serving.so')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',HYGEMM_LIBRARY=old,HYGEMM_PACK_MODE='0',
    HYGEMM_FUSE_SILU='0',HYGEMM_ROW_ORDER='none',HYGEMM_FUSED_GATEUP='0',HYGEMM_SILU_MATH='0',
    HYGEMM_PRODUCER_LIBRARY=str(stage/'build/libhygemm_producer_v2.so'))
def run(label,plans=None,extra=None,required=True):
    cmd=[sys.executable,str(src/'decode_quality.py'),'--stage',str(stage),'--model','7B','--label',label,
        '--compare',str(folder/'quality_old_paired/quality.json')]
    if plans:cmd+=['--plans',str(plans)]
    (stage/'quality_repair_status.json').write_text(json.dumps({'phase':label,'command':cmd,'updated':time.time()},indent=2))
    with (stage/(label+'.log')).open('w') as f:ret=subprocess.run(cmd,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode and required:
        (stage/'quality_repair_status.json').write_text(json.dumps({'phase':'failed','job':label,'exit':ret.returncode},indent=2));raise SystemExit(ret.returncode)
    print('Completed',label,ret.returncode,flush=True)
run('quality_old_paired_repeat')
run('quality_new_paired',folder/'tune_new/plans.json',{'HYGEMM_LIBRARY':new},False)
run('quality_combination_paired',folder/'combination_plans.json',{'HYGEMM_LIBRARY':new,'HYGEMM_PACK_MODE':'3','HYGEMM_FUSE_SILU':'1'})
for canonical,paired in [('quality_old','quality_old_paired'),('quality_new','quality_new_paired'),('quality_combination','quality_combination_paired')]:
    target=folder/canonical/'quality.json';backup=target.with_name('unpaired_quality.json')
    if target.exists() and not backup.exists():shutil.copy2(target,backup)
    shutil.copy2(folder/paired/'quality.json',target)
(stage/'quality_repair_status.json').write_text(json.dumps({'phase':'complete','updated':time.time()},indent=2))
(stage/'round3_status.json').write_text(json.dumps({'phase':'complete_7B_gate','quality':'paired-scheduler controls',
    'quality_comparison':str(folder/'quality_combination_paired/quality.json'),'updated':time.time()},indent=2))
