"""Record versions, source/binary hashes and compatibility-patch provenance."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
root=Path(__file__).resolve().parents[2]
def capture(command):return subprocess.check_output(command,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[root/'CMakeLists.txt',*sorted((root/'src/kernel').glob('output_channel*')),
         *sorted((root/'integrations/vllm').rglob('*.py')),root/'integrations/vllm/pyproject.toml',
         root/'integrations/vllm/constraints-cu130.txt']
up=args.out/'upstream/mixllm/kernels'
built=args.out/'mixllm_build/source'
device_files=[up/'mma_multistage_testbed.h',up/'mix_mma_config.h',*sorted((up/'cutlass_extension').rglob('*'))]
unchanged={str(p.relative_to(up)):sha(p)==sha(built/p.relative_to(up)) for p in device_files if p.is_file()}
assert unchanged and all(unchanged.values())
meta={'base_commit':capture(['git','-C',str(root),'rev-parse','HEAD']),
      'python':sys.version,'vllm':md.version('vllm'),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
      'transformers':md.version('transformers'),'flashinfer':md.version('flashinfer-python'),
      'gpu':capture(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),
      'nvcc':capture(['/usr/local/cuda/bin/nvcc','--version']),
      'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
      'binaries_sha256':{name:sha(Path(os.environ[name])) for name in ['HYGEMM_LIBRARY','MIXLLM_LIBRARY']},
      'mixllm_device_headers_unchanged':unchanged,
      'mixllm_compatibility_patch_sha256':sha(args.out/'mixllm_build/compatibility.patch'),
      'framework_install':'vLLM release wheel; editable out-of-tree quantization plugin',
      'scope':'Qwen2.5-0.5B TP1 smoke; no performance/quality claim'}
(args.out/'environment.json').write_text(json.dumps(meta,indent=2))
(args.out/'pip_freeze.txt').write_text(capture([sys.executable,'-m','pip','freeze'])+'\n')
(args.out/'pip_check.txt').write_text(capture([sys.executable,'-m','pip','check'])+'\n')
print('Environment recorded; all original MixLLM GEMM device headers match')
