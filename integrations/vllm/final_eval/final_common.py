"""Stage-three paths, content hashes, method environments and model views."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/root/autodl-tmp/20261025_052049_three_stage_plan')
S1=ROOT/'01_quality/runs/20261025_211540_q2_q3_allocation'
Q1=ROOT/'01_quality/runs/20261025_134149_budget_diagnosis_fixed_counts'
S2=ROOT/'02_performance/runs/20261026_015237_stage2_performance'
FORMAL=Path('/root/autodl-tmp/20261024_235427_formal_decode')
PYTHON='/root/autodl-tmp/hygemm-vllm026/bin/python'
EVAL_PYTHON='/root/autodl-tmp/hygemm-eval-tools/bin/python'
OLD=Path('/root/autodl-tmp/20261025_101734_small_batch/build/libhygemm_serving.so')
NEW=S2/'build/libhygemm_serving.so'
MIX=FORMAL/'mixllm_build/extension/mixllm_sm120.so'
BATCHES=[1,2,4,8,16,32,64,128]
MODELS=['0.5B','1.5B','3B','7B','Mistral-7B']
METHODS=['FP16','Mix-I','Hy-I','Hy-F','Hy-F-Q']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def environment(old=False):
    return {'LD_LIBRARY_PATH':'/usr/local/cuda/lib64','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
        'VLLM_PLUGINS':'hygemm_quantization','VLLM_NO_USAGE_STATS':'1',
        'HYGEMM_LIBRARY':str(OLD if old else NEW),'MIXLLM_LIBRARY':str(MIX),
        'HYGEMM_PACK_MODE':'0' if old else '3','HYGEMM_FUSE_SILU':'0' if old else '1',
        'HYGEMM_SILU_MATH':'0','HYGEMM_ROW_ORDER':'none','HYGEMM_FUSED_GATEUP':'0',
        'HYGEMM_PRODUCER_LIBRARY':str(S2/'build/libhygemm_producer_v2.so')}
def view(source,dest,config):
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.name not in ['config.json','selection.json','view.json'] and not (dest/file.name).exists():
            (dest/file.name).symlink_to(file.resolve())
    write(dest/'config.json',config)
def model_kwargs(path,worker='hygemm_vllm.final_worker.FinalWorkerExtension',max_len=2049):
    return {'model':str(path),'dtype':'float16','tensor_parallel_size':1,'max_model_len':max_len,
        'max_num_seqs':128,'max_num_batched_tokens':2048,'enable_prefix_caching':False,
        'gpu_memory_utilization':.93,'kv_cache_memory_bytes':10*1024**3,'seed':2029,'enforce_eager':False,
        'attention_config':{'backend':'FLASH_ATTN'},'worker_extension_cls':worker,
        'compilation_config':{'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':BATCHES,'max_cudagraph_capture_size':128}}
