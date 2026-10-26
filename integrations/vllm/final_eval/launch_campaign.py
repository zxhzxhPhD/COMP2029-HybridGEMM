"""Advance only after all preflight queues pass; stop on any failed gate."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import time
from final_common import *

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();s=args.stage
src=Path(__file__).resolve().parent
names=['dry_qwen','family_dry','robustness_prepare','actual_shape_tiling','tasks_serial_dry',
       'extra_dry','development_decode','last_preparation','remaining_preflight']
while True:
    states={name:read(s/'queues'/name/'status.json') if (s/'queues'/name/'status.json').exists() else {'phase':'waiting'} for name in names}
    failed={n:v for n,v in states.items() if v['phase']=='failed'}
    if failed:
        write(s/'campaign_status.json',{'phase':'preflight_failed','queues':failed});raise SystemExit(1)
    if all(v['phase']=='complete' for v in states.values()):break
    write(s/'campaign_status.json',{'phase':'waiting_for_preflight','queues':{n:v['phase'] for n,v in states.items()}})
    time.sleep(30)
def run(label,command):
    write(s/'campaign_status.json',{'phase':label,'command':command})
    with (s/(label+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write(s/'campaign_status.json',{'phase':'failed','job':label,'returncode':result.returncode});raise SystemExit(result.returncode)
    print('Completed campaign transition',label,flush=True)
with (s/'gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not (s/'freeze/current.json').exists():
        run('preflight',[PYTHON,str(src/'preflight.py'),'--stage',str(s)])
        run('declare_final',[PYTHON,str(src/'campaign.py'),'--stage',str(s)])
        run('freeze',[PYTHON,str(src/'freeze.py'),'--stage',str(s)])
    else:run('verify_before_resume',[PYTHON,str(src/'freeze.py'),'--stage',str(s),'--verify'])
run('final_queue',['/root/miniconda3/bin/python',str(src/'serial_jobs.py'),'--stage',str(s),
    '--jobs',str(s/'final_jobs.json'),'--name','final'])
run('acceptance',[PYTHON,str(src/'accept.py'),'--stage',str(s),'--full-hashes'])
run('report',['/root/miniconda3/bin/python',str(src/'report.py'),'--stage',str(s)])
write(s/'campaign_status.json',{'phase':'complete','finished':time.time()})
