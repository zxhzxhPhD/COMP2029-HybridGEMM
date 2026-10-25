"""Record version and artifact provenance without dumping credentials/environment."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
from prepare import write,sha
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage
root=Path(__file__).resolve().parents[3]
def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
versions={}
for name in ['torch','vllm','transformers','triton','flashinfer-python','nvidia-cuda-nvcc-cu13','nvidia-cuda-runtime-cu13']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
libraries=[stage/'build/libhygemm_serving.so',stage/'build/libhygemm_producer_v2.so',stage/'fused_serving_build/libfused_probe.so',
    Path('/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so'),
    Path('/root/autodl-tmp/20261024_235427_formal_decode/mixllm_build/extension/mixllm_sm120.so')]
write(stage/'environment.json',{'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),
    'packages':versions,'nvcc':command(['/usr/local/cuda/bin/nvcc','--version']),
    'gpu':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader']),
    'cutlass_commit':command(['git','-C','/root/code/cutlass','rev-parse','HEAD']),
    'source_parent_commit':command(['git','-C',str(root),'rev-parse','HEAD']),
    'libraries':{str(p):sha(p) for p in libraries},
    'cuda_target':'sm_120a; CUDA 13.0; no compiler/runtime upgrade',
    'gpu_jobs':'serial; CPU builds/analysis may overlap','original_mix_device_modified':False})
