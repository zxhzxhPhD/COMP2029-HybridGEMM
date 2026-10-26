"""Sequential, resumable stage-three jobs with one GPU lock."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
from final_common import read,write,sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--jobs',type=Path,required=True)
ap.add_argument('--name',required=True);args=ap.parse_args()
stage=args.stage;out=stage/'queues'/args.name;out.mkdir(parents=True,exist_ok=True)
with (stage/'gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    jobs=read(args.jobs)
    for index,job in enumerate(jobs):
        status=out/(job['id']+'.json')
        if status.exists() and read(status).get('passed'):
            assert read(status)['job']==job,'Changed queued command';continue
        started=time.time();record={'job':job,'index':index,'total':len(jobs),'started':started,'passed':False}
        write(out/'status.json',{'phase':'running','index':index,'total':len(jobs),'job':job})
        with (out/(job['id']+'.log')).open('w') as log:
            ret=subprocess.run(job['command'],env={**os.environ,**job.get('environment',{})},stdout=log,stderr=subprocess.STDOUT)
        record.update(exit_code=ret.returncode,elapsed_s=time.time()-started,passed=ret.returncode==0)
        write(status,record)
        if ret.returncode and not job.get('allow_failure',False):
            write(out/'status.json',{'phase':'failed','job':job,'exit_code':ret.returncode});raise SystemExit(ret.returncode)
        print('Finished',args.name,index+1,'/',len(jobs),job['id'],ret.returncode,flush=True)
    write(out/'status.json',{'phase':'complete','jobs':len(jobs),'finished':time.time(),'jobs_sha256':sha(args.jobs)})
