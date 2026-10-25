"""Finite FP16 extremes and subnormal activations with bounded outputs."""
import argparse
import json
from pathlib import Path
import sys
import torch
from hygemm_vllm.native import DualLinear
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_components import reference

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
records=[]
for variant in range(28,45):
    w=(torch.randn(129,896,device='cuda')*1e-4).half()
    r=DualLinear(w,torch.randperm(129,device='cuda')[:19],variant,variant)
    for scale in [0,2**-24,2**-14,1,32752]:
        x=(torch.rand(3,896,device='cuda')*2-1).mul_(scale).half()
        y=r(x);expected=reference(r,x,'hygemm')
        torch.testing.assert_close(y,expected,atol=.01,rtol=.003)
        records.append({'variant':variant,'scale':scale,'max_abs':float((y.float()-expected.float()).abs().max()),'passed':True})
    r.close()
args.out.write_text(json.dumps({'passed':True,'records':records},indent=2))
