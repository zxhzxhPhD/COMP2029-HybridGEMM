"""Build changed units, reusing the frozen CUTE objects without recompiling them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--base',type=Path,default=Path('/root/autodl-tmp/20261025_101734_small_batch/build'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    nvcc='/usr/local/cuda/bin/nvcc'
    common=[nvcc,'-std=c++17','-gencode=arch=compute_120a,code=sm_120a','-Xcompiler=-fPIC',
        '--expt-relaxed-constexpr','-lineinfo','-Xptxas=-v','-I/root/code/cutlass/include',
        '-I/root/code/cutlass/tools/util/include','-I'+str(root/'src/kernel')]
    sources={'gemv':'src/kernel/output_channel_gemv.cu','light':'src/kernel/output_channel_light.cu',
             'runtime':'integrations/vllm/native/runtime.cu','factory':'src/kernel/output_channel_gemm.cpp'}
    log=[]
    for name,src in sources.items():
        cmd=common+['-c',str(root/src),'-o',str(args.out/(name+'.o'))]
        with (args.out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        log.append({'command':cmd,'source_sha256':sha(root/src)})
    archive=args.base/'libhygemm_output_channel.a'
    cmd=[nvcc,'-shared','-o',str(args.out/'libhygemm_serving.so')]+[str(args.out/(n+'.o')) for n in sources]+[str(archive),'-lcudart']
    subprocess.run(cmd,check=True)
    (args.out/'build.json').write_text(json.dumps({'commands':log,'link':cmd,'frozen_archive_sha256':sha(archive),
        'library_sha256':sha(args.out/'libhygemm_serving.so'),
        'reuse':'old archive factory/light objects are not extracted: new units satisfy their symbols'},indent=2))


if __name__=='__main__':main()
