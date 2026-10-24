"""Build an isolated copy of the original MixLLM extension for SM120.

The upstream checkout is read-only. The manifest records toolchain and host
dispatch/event edits and hashes. No device MMA, dequantization, pipeline or
epilogue is rewritten.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--planned', action='store_true', help='Expose original candidates and exact-shape host dispatch')
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
if args.planned:
    # Add only host dispatch/metadata. The exact upstream device templates and
    # candidate lists are retained, including resource-heavy candidates that
    # the external tuner will reject before launching.
    hs = hs.replace('namespace mixllm{', 'namespace mixllm{\ninline thread_local int forced_stage=0, forced_id=0;')
    anchor = '    torch::Tensor output;'
    assert hs.count(anchor) == 1
    hs = hs.replace(anchor, anchor+'''\n    if (forced_stage) {
      best_stage=forced_stage;best_config_id=forced_id;
      if(row_major){CALL_GEMM_SWITCH_RM(best_stage,best_config_id)}
      else {CALL_GEMM_SWITCH(best_stage,best_config_id)}
      return output;
    }
''')
    host.write_text(hs)
    config = (src/'mix_mma_config.h').read_text()
    choices = {}
    for rm in [False,True]:
        for stages in [5,11]:
            macro = f'#define CALL_GEMM_SWITCH_{"RM_" if rm else ""}{stages}(stage, config_id)'
            block = config.split(macro,1)[1].split('\n#define',1)[0]
            cases = re.findall(r'CALL_GEMM_CASE(?:_RM)?\(\s*(\d+)\s*,\s*stage\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',block)
            assert cases, f'Empty candidate family: {macro}'
            choices[f'{int(rm)}_{stages}'] = [list(map(int,c)) for c in cases]
    helper = '''
namespace mixllm {
template<int S,int BM,int BN,int WM,int WN,bool RM>
std::vector<int64_t> candidate_info() {
  using TB=cutlass::gemm::GemmShape<BM,BN,64>;
  using W=cutlass::gemm::GemmShape<WM,WN,64>;
  using I=cutlass::gemm::GemmShape<16,8,32>;
  using L=typename std::conditional<RM,cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>::type;
  using M8=typename cutlass::gemm::threadblock::DefaultMmaCore<TB,W,I,ElementA,LayoutA,
      ElementB_INT8,LayoutB,ElementC,L,cutlass::arch::OpClassTensorOp,S,cutlass::arch::OpMultiplyAddSaturate>;
  using M4=typename cutlass::gemm::threadblock::DefaultMmaCore<TB,W,I,ElementA,LayoutA,
      ElementB_INT4,LayoutB,ElementC,L,cutlass::arch::OpClassTensorOp,S,cutlass::arch::OpMultiplyAddMixedAndShuffledInputUpcast>;
  return {S,BM,BN,WM,WN,int64_t(sizeof(typename extension_cpp::Testbed<M4,RM>::SharedStorage)),
          int64_t(sizeof(typename extension_cpp::Testbed<M8,RM>::SharedStorage)),32*(BM/WM)*(BN/WN)};
}
}
std::vector<int64_t> config_info(int64_t stage,int64_t id,bool rm) {
'''
    for key,cases in choices.items():
        rm,stage=map(int,key.split('_'))
        helper += f'  if(rm=={str(bool(rm)).lower()} && stage=={stage}) switch(id) {{\n'
        for cid,bm,bn,wm,wn in cases:
            helper += f'    case {cid}:return mixllm::candidate_info<{stage},{bm},{bn},{wm},{wn},{str(bool(rm)).lower()}>();\n'
        helper += '  }\n'
    helper += '  return {};\n}\n'
    helper += '''
torch::Tensor gemm_planned(torch::Tensor &a,torch::Tensor &sa,torch::Tensor &z,
  torch::Tensor &s8,torch::Tensor &s4,torch::Tensor &i8,torch::Tensor &i4,
  torch::Tensor &b8,torch::Tensor &b4,int64_t stage,int64_t id) {
  TORCH_CHECK(!config_info(stage,id,i8.numel()==0 || i4.numel()==0).empty(),"Invalid original MixLLM candidate");
  struct Guard { ~Guard(){mixllm::forced_stage=0;mixllm::forced_id=0;} } guard;
  mixllm::forced_stage=int(stage);mixllm::forced_id=int(id);
  return gemm(a,sa,z,s8,s4,i8,i4,b8,b4);
}
TORCH_LIBRARY_FRAGMENT(kernels_mixllm,m) {
  m.def("config_info(int stage, int id, bool row_major) -> int[]", &config_info);
  m.def("gemm_planned(Tensor a, Tensor sa, Tensor z, Tensor s8, Tensor s4, Tensor i8, Tensor i4, Tensor b8, Tensor b4, int stage, int id) -> Tensor");
  m.impl("gemm_planned",torch::kCUDA,&gemm_planned);
}
'''
    s += helper
    p.write_text(s)
    (args.out/'original_candidates.json').write_text(json.dumps(choices,indent=2))
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
    'original_device_mma_pipeline_epilogue': True,
    'original_candidate_templates': True,
    'host_joins_only_launched_streams': True,
    'explicit_plan_dispatch':args.planned,
    'source_sha256': {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in dst.rglob('*') if p.is_file()}}
(args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'}, indent=2))
