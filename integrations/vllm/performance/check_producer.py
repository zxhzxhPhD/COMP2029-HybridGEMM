"""Require parity with the installed vLLM operator, not a mathematical proxy."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import torch
import vllm._custom_ops
from hygemm_vllm.native import pack,library

ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--maths',default='0,1,2')
args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261026)
lib=C.CDLL(args.library);lib.hg_silu_pack.argtypes=[C.c_void_p]*4+[C.c_int]*2+[C.c_void_p,C.c_int]
records=[];modes=list(map(int,args.maths.split(',')))
for math in modes:
    for r in [1,3,8,33,64,128,129,512]:
        for k in [128,896,4864,18944]:
            for scale in [.1,1,4,16]:
                x=(torch.randn(r,2*k,device='cuda')*scale).half()
                ref=torch.empty(r,k,device='cuda',dtype=torch.float16);torch.ops._C.silu_and_mul(ref,x)
                eq,es=pack(ref);q,sf=torch.empty_like(eq),torch.empty_like(es);debug=torch.empty_like(ref)
                assert lib.hg_silu_pack(x.data_ptr(),q.data_ptr(),sf.data_ptr(),debug.data_ptr(),r,k,torch.cuda.current_stream().cuda_stream,math)==0
                records.append({'math':math,'r':r,'k':k,'scale':scale,'half_differences':int((ref.view(torch.int16)!=debug.view(torch.int16)).sum()),
                    'q_differences':int((eq!=q).sum()),'sf_differences':int((es!=sf).sum()),
                    'max_half_error':float((ref.float()-debug.float()).abs().max())})
    print('Math',math,'errors',sum(x['q_differences']+x['sf_differences'] for x in records if x['math']==math),flush=True)
passing=[m for m in modes if all(x['half_differences']==x['q_differences']==x['sf_differences']==0 for x in records if x['math']==m)]
args.out.write_text(json.dumps({'records':records,'tested_modes':modes,'passing_math_modes':passing,'library':args.library},indent=2))
