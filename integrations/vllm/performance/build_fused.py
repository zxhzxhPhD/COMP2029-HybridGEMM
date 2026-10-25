"""Recompile the unchanged precision-switch kernels as PIC for a fixed-pointer probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[3]
nvcc='/usr/local/cuda/bin/nvcc'
common=[nvcc,'-std=c++17','-gencode=arch=compute_120a,code=sm_120a','-Xcompiler=-fPIC',
    '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v',
    '-I/root/code/cutlass/include','-I/root/code/cutlass/tools/util/include','-I'+str(root/'src/kernel'),
    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1']
def compile_one(i):
    n=[16,32,64,128,16,32,64,64][i];k=[128,128,128,128,256,256,128,128][i]
    cmd=common+[f'-DHYGEMM_OC_VARIANT_ID={i}',f'-DHYGEMM_OC_TILE_N={n}',f'-DHYGEMM_OC_TILE_K={k}',
        f'-DHYGEMM_OC_STAGES={2 if i==6 else 0}',f'-DHYGEMM_OC_PINGPONG={int(i==7)}',
        '-c',str(root/'src/kernel/output_channel_fused_variant.cu'),'-o',str(out/f'fused_{i}.o')]
    with (out/f'fused_{i}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(2) as pool:list(pool.map(compile_one,range(8)))
subprocess.run(common+['-shared',str(root/'integrations/vllm/native/fused_probe.cu'),
    str(root/'src/kernel/output_channel_fused.cpp'),*[str(out/f'fused_{i}.o') for i in range(8)],
    '-o',str(out/'libfused_probe.so')],check=True)
