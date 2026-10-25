"""Every finite FP16 value as group maximum, including scale-boundary ties."""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from hygemm_vllm import native
ap=argparse.ArgumentParser();ap.add_argument('--old-library',required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
torch.set_num_threads(4);old=C.CDLL(args.old_library);old.hg_pack.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]
values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16);values[~torch.isfinite(values)]=0
values=values.cuda();rows=[]
for kind in ['constant','ramp','negative_zero']:
    x=values[:,None].expand(-1,128).clone()
    if kind=='ramp':x=(x.float()*torch.linspace(-1,1,128,device='cuda')[None,:]).half()
    if kind=='negative_zero':x[:,1:]=torch.tensor(-0.,device='cuda',dtype=torch.float16)
    native.PACK_MODE=2;q,sf=native.pack(x);eq,es=torch.empty_like(q),torch.empty_like(sf)
    assert old.hg_pack(x.data_ptr(),eq.data_ptr(),es.data_ptr(),len(x),128,0,torch.cuda.current_stream().cuda_stream)==0
    qi=(q!=eq).nonzero();si=(sf!=es).nonzero()
    examples=[]
    for r,c in qi[:20].tolist():examples.append({'row':r,'column':c,'input':float(x[r,c]),'max':float(x[r].abs().max()),'old_q':int(eq[r,c]),'new_q':int(q[r,c])})
    rows.append({'kind':kind,'q_differences':len(qi),'sf_differences':len(si),'examples':examples})
args.out.write_text(json.dumps({'records':rows,'passed':all(r['q_differences']==r['sf_differences']==0 for r in rows)},indent=2))
print(json.dumps(rows,indent=2))
