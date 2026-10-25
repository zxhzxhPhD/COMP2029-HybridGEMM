"""Record versions, revisions and frozen-library hashes without a CUDA context."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
repo=Path(__file__).resolve().parents[3]
def command(*args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
data={'python':platform.python_version(),'platform':platform.platform(),
    'packages':{x:importlib.metadata.version(x) for x in ['torch','vllm','transformers','numpy']},
    'source_base_commit':command('git','rev-parse','HEAD',cwd=repo),
    'cutlass_commit':command('git','rev-parse','HEAD',cwd='/root/code/cutlass'),
    'gpu':command('nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'),
    'nvcc':command('/usr/local/cuda/bin/nvcc','--version'),
    'hygemm_library_sha256':sha(os.environ['HYGEMM_LIBRARY']),
    'mixllm_library_sha256':sha(os.environ['MIXLLM_LIBRARY']),
    'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sorted((repo/'integrations/vllm/allocation').glob('*.py'))}}
args.out.write_text(json.dumps(data,indent=2))
print(data['source_base_commit'],data['gpu'])
