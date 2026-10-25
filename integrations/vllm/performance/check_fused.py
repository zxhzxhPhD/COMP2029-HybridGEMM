"""Rebinding and Graph invariants for the reintroduced native TMA switch path."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import torch
from hygemm_vllm.native import DualLinear
from hygemm_vllm import native
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_components import reference

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sanitizer',action='store_true');args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
native.FUSED_GATEUP=128;records=[]
shapes=[(257,k,h) for k in [128,384,896,3584] for h in [0,1,26,256,257]]
if args.sanitizer:shapes=[(257,384,26),(129,128,0),(129,896,129)]
for o,k,h in shapes:
    w=(torch.randn(o,k,device='cuda')*.04).half();ids=torch.randperm(o,device='cuda')[:h]
    r=DualLinear(w,ids);r.layer_name='model.layers.0.mlp.gate_up_proj'
    baseline=DualLinear(w,ids)
    for t in [1,3,8,17,65]:
        x=torch.randn(t,k,device='cuda',dtype=torch.float16);y=r(x)
        expected=reference(r,x,'hygemm');torch.testing.assert_close(y,expected,atol=.01,rtol=.003)
        control=baseline(x);equal=torch.equal(y,control)
        x2=torch.randn_like(x);torch.testing.assert_close(r(x2),reference(r,x2,'hygemm'),atol=.01,rtol=.003)
        r(x);graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):y=r(x)
        cudart=C.CDLL('/usr/local/cuda/lib64/libcudart.so');cudart.cudaGraphGetNodes.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_size_t)]
        count=C.c_size_t();assert cudart.cudaGraphGetNodes(graph.raw_cuda_graph(),None,C.byref(count))==0 and count.value>=2
        for scale in [0,.25,2]:
            x.copy_(torch.randn_like(x)*scale);graph.replay()
            torch.testing.assert_close(y,reference(r,x,'hygemm'),atol=.01,rtol=.003)
        records.append({'out':o,'k':k,'high':h,'tokens':t,'passed':True,'baseline_bitwise_equal':equal,'graph_nodes':count.value})
    r.close();baseline.close()
args.out.write_text(json.dumps({'passed':True,'records':records},indent=2))
