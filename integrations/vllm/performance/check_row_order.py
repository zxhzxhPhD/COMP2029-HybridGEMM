import argparse
import json
from pathlib import Path
import torch
from hygemm_vllm.native import DualLinear
from hygemm_vllm.row_order import reorder_runtime_rows

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261026);records=[]
for o,k,h in [(257,128,26),(129,896,0),(129,896,129),(37888,3584,616)]:
    w=(torch.randn(o,k,device='cuda')*.04).half();ids=torch.randperm(o,device='cuda')[:h]
    for mode in ['reverse','block_reverse']:
        for variant in [0,8,28,32,35]:
            old=DualLinear(w,ids,variant,variant);new=DualLinear(w,ids,variant,variant)
            before=sum(getattr(new,n).untyped_storage().nbytes() for n in ['lo','hi','w4','w8','sf4','sf8'])
            reorder_runtime_rows(new,mode)
            after=sum(getattr(new,n).untyped_storage().nbytes() for n in ['lo','hi','w4','w8','sf4','sf8']);assert before==after
            for t in [1,3,8,64]:
                if variant in [28,32] and t>8:continue
                x=torch.randn(t,k,device='cuda',dtype=torch.float16)
                expected=old(x);actual=new(x)
                assert torch.equal(expected,actual),(o,k,h,t,mode,variant)
                records.append({'out':o,'k':k,'high':h,'tokens':t,'mode':mode,'variant':variant,'bitwise_equal':True,'storage_bytes':after})
            old.close();new.close()
args.out.write_text(json.dumps({'passed':True,'records':records},indent=2))
