"""Native pack parity, legal exponent search, bank layout, tails and Graphs."""
import argparse
from pathlib import Path
import torch
from alloc_utils import *
from fp4_quant import codes_rne,pack_codes,decode_logical,quantize
from hygemm_vllm.native import pack,DualLinear
from hygemm_vllm.offline_weights import logical_scales,physical_scales,replace_low_weight
from test_components import reference

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--sanitizer',action='store_true');args=ap.parse_args();configure();torch.manual_seed(20261025)
records=[]
values=torch.tensor([0.,-.0,.25,-.25,.75,-.75,1.25,-1.25,1.75,-1.75,2.5,-2.5,3.5,-3.5,5.,-5.,6.,-6.],device='cuda',dtype=torch.float16)
for rows in ([3] if args.sanitizer else [1,3,127,128,129]):
    for k in ([128] if args.sanitizer else [128,896,3584]):
        w=torch.randn(rows,k,device='cuda',dtype=torch.float16)*.03
        w[0,:len(values)]=values
        q,sf=pack(w,True);logical=logical_scales(sf,rows,k)
        scaled=w.float().reshape(rows,k//32,32)/torch.exp2(logical.float()-127)[:,:,None]
        assert torch.equal(pack_codes(codes_rne(scaled).reshape(rows,k)),q)
        assert torch.equal(physical_scales(logical,k),sf)
        assert torch.equal(decode_logical(q,logical,k),decode_mx(q,sf,k,True))
        for mode in ['native_rtn','mse_clip','act_mse_clip']:
            importance=torch.rand(k,device='cuda')+.1 if mode=='act_mse_clip' else None
            qq,ss,stats=quantize(w,importance,offsets=(0,) if mode=='native_rtn' else (0,-1,-2))
            if mode=='native_rtn':assert torch.equal(qq,q) and torch.equal(ss,logical)
            records.append({'type':'quantizer','rows':rows,'k':k,'mode':mode,'passed':True,'stats':stats})
for k in ([896] if args.sanitizer else [128,896,3584]):
    w=(torch.randn(257,k,device='cuda')*.04).half()
    for mode in ['native_rtn','mse_clip','act_mse_clip']:
        q,sf,_=quantize(w,torch.rand(k,device='cuda')+.1 if mode=='act_mse_clip' else None,
                       offsets=(0,) if mode=='native_rtn' else (0,-1,-2))
        for high in [0,26,257]:
            r=DualLinear(w,torch.randperm(257,device='cuda')[:high])
            replace_low_weight(r,q.cpu(),sf.cpu(),verify_native=mode=='native_rtn')
            for tokens in [1,3,17]:
                x=torch.randn(tokens,k,device='cuda',dtype=torch.float16)
                torch.testing.assert_close(r(x),reference(r,x,'hygemm'),atol=.002,rtol=.002)
                records.append({'type':'gemm','mode':mode,'k':k,'high':high,'tokens':tokens,'passed':True})
            if high==26:
                x=torch.randn(3,k,device='cuda',dtype=torch.float16);r(x);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=r(x)
                for scale in [0,.25,2.]:
                    x.copy_(torch.randn_like(x)*scale);graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(y,reference(r,x,'hygemm'),atol=.002,rtol=.002)
                records.append({'type':'changed_input_graph','mode':mode,'k':k,'passed':True})
            r.close()
write(args.out,{'passed':True,'records':records,'kernel_sha256':sha(__import__('os').environ['HYGEMM_LIBRARY'])})
print('Passed',len(records),'component checks',flush=True)
