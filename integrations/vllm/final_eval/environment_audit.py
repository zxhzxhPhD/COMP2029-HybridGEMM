"""Record installed-source baseline support and compiler/binary provenance."""
import argparse
import ctypes
import gzip
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
from final_common import *

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();s=args.stage
vllm=Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
root=vllm/'model_executor/layers/quantization';evidence={}
for name,patterns in {
    'mxfp4.py':['def get_supported_act_dtypes','if isinstance(layer, LinearBase)','MXFP4 linear layer is not implemented'],
    'modelopt.py':['class ModelOptNvFp4Config','dynamic quantization is not supported','MXFP8 currently only supports serialized'],
    'fp8.py':['class Fp8Config','is_checkpoint_fp8_serialized','use_marlin'],
    'online/fp8.py':['class Fp8PerTensorOnlineLinearMethod','weight_quant_key =','activation_quant_key =','use_marlin ='],
    'auto_awq.py':['class','weight_bits','quantize_config'],
    'auto_gptq.py':['class','weight_bits','quantize_config'],
}.items():
    file=root/name;lines=file.read_text().splitlines();snippets=[]
    for i,line in enumerate(lines):
        if any(p in line for p in patterns):snippets.append({'line':i+1,'text':'\n'.join(lines[max(0,i-2):i+6])})
    evidence[name]={'path':str(file),'sha256':sha(file),'snippets':snippets}
mixroot=Path('/root/autodl-tmp/20261024_223704_vllm_integration/upstream/mixllm')
mixfiles={str(p.relative_to(mixroot)):sha(p) for p in (mixroot/'quantization').glob('*.py')}
write(s/'baseline_compatibility.json',{'vllm_version':importlib.metadata.version('vllm'),'source_evidence':evidence,
    'decisions':{
        'ordinary_fp8':'Selected conditionally on native dry run. Actual runtime backend and activation path will be reported; not MXFP8.',
        'native_mxfp4_dense':'Installed Mxfp4Config returns UnquantizedLinearMethod for dense LinearBase and only declares BF16 activation support. This entry is not a comparable FP16 dense MXFP4 baseline.',
        'modelopt_nvfp4':'Installed dense implementation requires serialized quantized weights; direct dynamic loading of the same FP16 checkpoint is rejected. Separate matched ModelOpt calibration/export was not selected.',
        'modelopt_mxfp8':'Installed implementation requires serialized checkpoint. Our native MX-W8A8 endpoint provides same-bank format control instead.',
        'int4_weight_only':'AWQ/GPTQ classes exist, but a matched externally calibrated checkpoint/PTQ pipeline was not selected. This is scope, not a claim of hardware incompatibility.',
        'mix_paper_ptq':'Original quantizer/searcher sources retained and hashed. Main Mix-I uses the established load-time RTN checkpoint path; paper GPTQ/clipping is outside this controlled deployment comparison. No paper-optimal quality claim.'},
    'mix_original_quantization_sources':mixfiles,'selection_before_final_scoring':True})
out=s/'environment';out.mkdir(exist_ok=True)
driver=ctypes.CDLL('libcuda.so.1');assert driver.cuInit(0)==0
l2=ctypes.c_int();assert driver.cuDeviceGetAttribute(ctypes.byref(l2),38,0)==0
write(out/'gpu_l2.json',{'l2_cache_bytes':l2.value,
    'source':'cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device 0), CUDA header enum 38',
    'measurement_uses_no_CUDA_context':True})
commands={'gpu':['nvidia-smi','-q'],'nvcc':['/usr/local/cuda/bin/nvcc','--version'],
    'gpu_pip':[sys.executable,'-m','pip','freeze'],'cutlass':['git','-C','/root/code/cutlass','rev-parse','HEAD'],
    'cutlass_status':['git','-C','/root/code/cutlass','status','--porcelain'],
    'kernel':['uname','-a']}
for name,command in commands.items():
    p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    (out/(name+'.txt')).write_text(p.stdout)
records={}
for name,binary in [('hy_k1',NEW),('hy_k0',OLD),('mix',MIX),('producer',S2/'build/libhygemm_producer_v2.so'),
                    ('format',s/'format_build/hygemm_sm120_format_probe'),('format_tuned',s/'format_build/hygemm_sm120_format_probe_tuned')]:
    counts={key:0 for key in ['QMMA','OMMA','IMMA','HMMA','LDSM','LDGSTS','UTMALDG','LOP3','SHF']}
    process=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    with gzip.open(out/(name+'_sass.txt.gz'),'wt') as f:
        for line in process.stdout:
            f.write(line)
            for key in counts:
                if re.search(r'\b'+key+r'\b',line):counts[key]+=1
    errors=process.stderr.read();assert process.wait()==0,errors
    resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
    (out/(name+'_resources.txt')).write_text(resource)
    records[name]={'binary':str(binary),'sha256':sha(binary),'instruction_counts':counts,
        'sass_sha256':sha(out/(name+'_sass.txt.gz')),'resources_sha256':sha(out/(name+'_resources.txt'))}
write(out/'binary_evidence.json',records)
print('Installed baseline compatibility and SASS recorded',flush=True)
