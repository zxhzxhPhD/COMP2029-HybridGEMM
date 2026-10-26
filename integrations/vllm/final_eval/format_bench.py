"""Bounded actual-shape format comparison, including online activation packing."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
from final_common import read,write,sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
out=stage/'formats';out.mkdir(exist_ok=True)
base=stage/'format_build/hygemm_sm120_format_probe';tuned=stage/'format_build/hygemm_sm120_format_probe_tuned'
assert base.exists() and tuned.exists()
l2_bytes=read(stage/'environment/gpu_l2.json')['l2_cache_bytes']
env={**os.environ,'LD_LIBRARY_PATH':'/usr/local/cuda/lib64'}
variants=[(base,64),(base,128),(tuned,16),(tuned,64)]
configs=[(p,b,t,online) for p in ['w4a4','w4a8','w8a8_mx'] for b,t in variants for online in [False,True]]+ [('fp16',base,64,False)]
def run(shape,config,smoke=False,sanitizer=False):
    m,n,k=shape;path,binary,tile,online=config
    name=f'{m}_{n}_{k}_{path}_{binary.name}_{tile}_{int(online)}'
    command=[str(binary),f'--m={m}',f'--n={n}',f'--k={k}',f'--path={path}',f'--tile-n={tile}',
        '--iterations=30','--repeats=5']
    if online:command.append('--online-pack')
    if smoke:command.append('--full-check')
    if sanitizer:command=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','91',*command]
    process=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/(('memcheck_' if sanitizer else 'smoke_' if smoke else '')+name+'.log')).write_text(process.stdout)
    assert process.returncode==0,(name,process.returncode,process.stdout[-1500:])
    if sanitizer:assert '0 errors' in process.stdout
    row=json.loads(next(line for line in process.stdout.splitlines() if line.startswith('{"path"')))
    assert row['failures']==0
    cv=statistics.pstdev(row['samples_us'])/statistics.mean(row['samples_us'])
    weight_bytes=m*k*(16 if path=='fp16' else 8 if path=='w8a8_mx' else 4)//8
    row.update(binary_sha256=sha(binary),cv=cv,stable=cv<=.05,smoke=smoke,sanitizer=sanitizer,
        weight_payload_bytes=weight_bytes,weight_payload_to_l2_capacity=weight_bytes/l2_bytes)
    return row
checks=[]
for config in configs:checks.append(run((128,3,256),config,smoke=True))
for path in ['w4a4','w4a8','w8a8_mx']:checks.append(run((128,3,256),(path,tuned,16,True),smoke=True,sanitizer=True))
write(out/'checks.json',{'passed':True,'records':checks})
rows=[];rng=random.Random(3071)
for m,k in [(4608,3584),(37888,3584),(3584,18944)]:
    for n in [1,8,64]:
        order=list(configs);rng.shuffle(order)
        for config in order:
            attempts=[]
            for attempt in range(3):
                row=run((m,n,k),config);attempts.append(row)
                if row['stable']:break
            row['attempts']=attempts[:-1];rows.append(row);write(out/'raw.json',rows)
            print('Format',m,n,k,config[0],config[2],config[3],row['median_us'],row['cv'],flush=True)
best={}
for row in rows:
    if not row['stable']:continue
    key=(row['m_out'],row['n_tokens'],row['k_in'],row['path'],row['online_activation_pack'])
    if key not in best or row['median_us']<best[key]['median_us']:best[key]=row
assert len(rows)==225 and len(best)==63,('Incomplete stable format coverage',len(rows),len(best))
write(out/'summary.json',{'completed':True,'records':list(best.values()),'unstable_candidates':sum(not r['stable'] for r in rows),
    'cache':'Repeated fixed device buffers under CUDA Graph replay; not guaranteed L2 residency and not a cold full-model speed prediction',
    'l2_cache_bytes':l2_bytes,'cache_limitation':'Weight payload sizes cross L2 capacity differently by format. No cache hit counters are available; these GEMM times do not isolate intrinsic MMA throughput.',
    'boundary':'Weights prepacked. Kernel-only versus online FP16 activation -> native MX pack (including scale padding init) -> GEMM. FP16 needs no activation quantizer.',
    'pack':'Common simple warp/K32 reference pack, not a claim of optimal format-specific quantization',
    'numerics':'Same exactly representable matrices for every format; no model-quality claim for W4A4',
    'source_sha256':sha(Path(__file__).resolve().parents[3]/'src/kernel/sm120_format_probe.cu'),
    'binaries':{str(p):sha(p) for p in [base,tuned]}})
