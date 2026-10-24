"""Build an isolated copy of the original MixLLM extension for SM120.

The upstream checkout is read-only. The manifest records any toolchain-only
edits and hashes. No MMA, dequantization, pipeline or epilogue is rewritten.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
src = args.upstream / 'mixllm/kernels'
dst = args.out / 'source'
dst.mkdir(parents=True, exist_ok=True)
for p in src.iterdir():
    if p.suffix in ['.cu', '.cuh', '.h']:
        shutil.copy2(p, dst / p.name)
shutil.copytree(src / 'cutlass_extension', dst / 'cutlass_extension', dirs_exist_ok=True)
# Explicit conversions compile with the modern PyTorch extension flags that
# disable implicit __half conversions. Values and arithmetic are unchanged.
p = dst / 'kernels.cu'
s = p.read_text().replace('__half max_int = 127.0f;', '__half max_int = __float2half(127.0f);')
s = s.replace('__hdiv(1.0, scale)', '__hdiv(__float2half(1.0f), scale)')
s = s.replace('max_val[i] / max_int', '__hdiv(max_val[i], max_int)')
p.write_text(s)
# Original host dispatch waits on both events even for a pure partition. An
# unused event can still belong to a previous CUDA graph capture. Only join
# streams actually launched by this request; device code and work stay intact.
host = dst / 'mix_mma_multistage.cuh'
hs = host.read_text()
for bits in [4, 8]:
    old = f'checkCudaErrors(cudaStreamWaitEvent(stream_for_graph, this->local_gemm_event_int{bits}, 0));'
    assert hs.count(old) == 2
    hs = hs.replace(old, f'if (options.partial_n_int{bits} > 0) '+old)
host.write_text(hs)
patch = ''.join(''.join(difflib.unified_diff((src/name).read_text().splitlines(True),
    (dst/name).read_text().splitlines(True), fromfile='upstream/'+name, tofile='build/'+name))
    for name in ['kernels.cu', 'mix_mma_multistage.cuh'])
(args.out / 'compatibility.patch').write_text(patch)
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['MAX_JOBS'] = '1'
from torch.utils.cpp_extension import load
build = args.out / 'extension'
build.mkdir(exist_ok=True)
library = load(name='mixllm_sm120', sources=[str(p)],
    extra_include_paths=[str(src/'cutlass/include'), str(src/'cutlass/tools/util/include'), str(dst/'cutlass_extension')],
    extra_cuda_cflags=['-O3', '--use_fast_math', '--expt-relaxed-constexpr', '-lineinfo'],
    extra_cflags=['-O3'], build_directory=str(build), verbose=True, is_python_module=False)
manifest = {'library': str(library), 'upstream_commit': '00263510f9760d3a1d958a388e0e7c25961a7d59',
    'cutlass_commit': '4370102f9dacab813282e1d67722fceb0b90a019', 'arch': 'sm_120',
    'original_gemm_dispatch_mma_pipeline_epilogue': True,
    'host_joins_only_launched_streams': True,
    'source_sha256': {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dst.rglob('*') if p.is_file()}}
(args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'}, indent=2))
