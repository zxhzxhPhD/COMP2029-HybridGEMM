"""Stage-two pack byte parity, policy correctness and changed-input graphs."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import torch
from hygemm_vllm import native
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_components import reference


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--old-library',required=True);ap.add_argument('--sanitizer',action='store_true')
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261026)
    torch.backends.cuda.matmul.allow_tf32=False
    old=C.CDLL(args.old_library);old.hg_pack.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]
    rows=[1,3,32,33,64,127,128,129,255,256,257,2048] if not args.sanitizer else [3,33,129]
    pack_records=[];policies=[]
    for mode in [1,2]:
        native.PACK_MODE=mode
        for r in rows:
            for k in ([128,896,3584,18944] if not args.sanitizer else [128,896]):
                x=torch.randn(r,k,device='cuda',dtype=torch.float16)
                for scale in [0,.125,2,4096]:
                    x.copy_(torch.randn_like(x)*scale)
                    q,sf=native.pack(x)
                    eq,es=torch.empty_like(q),torch.empty_like(sf)
                    assert old.hg_pack(x.data_ptr(),eq.data_ptr(),es.data_ptr(),r,k,0,torch.cuda.current_stream().cuda_stream)==0
                    assert torch.equal(q,eq) and torch.equal(sf,es),(mode,r,k,scale)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):q,sf=native.pack(x)
                for scale in [0,.25,32]:
                    x.copy_(torch.randn_like(x)*scale);sf.fill_(255);graph.replay()
                    assert old.hg_pack(x.data_ptr(),eq.data_ptr(),es.data_ptr(),r,k,0,torch.cuda.current_stream().cuda_stream)==0
                    assert torch.equal(q,eq) and torch.equal(sf,es)
                pack_records.append({'mode':mode,'rows':r,'k':k,'changed_graph':True,'all_bytes_equal':True})
        # Every finite FP16 bit pattern appears in a group, including subnormals.
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
        bits[~torch.isfinite(bits)]=0
        x=bits.reshape(-1,128).cuda();q,sf=native.pack(x);eq,es=torch.empty_like(q),torch.empty_like(sf)
        assert old.hg_pack(x.data_ptr(),eq.data_ptr(),es.data_ptr(),len(x),128,0,torch.cuda.current_stream().cuda_stream)==0
        assert torch.equal(q,eq) and torch.equal(sf,es)
    native.PACK_MODE=0
    for out,k,high in [(257,896,26),(65,128,0),(65,256,65),(129,3584,1)]:
        w=(torch.randn(out,k,device='cuda')*.04).half()
        for t in [1,3,8,17]:
            for policy in range(7):
                r=native.DualLinear(w,torch.randperm(out,device='cuda')[:high],28,28,policy)
                x=torch.randn(t,k,device='cuda',dtype=torch.float16);y=r(x)
                torch.testing.assert_close(y,reference(r,x,'hygemm'),atol=.01,rtol=.003)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=r(x)
                for scale in [0,.25,2]:
                    x.copy_(torch.randn_like(x)*scale);graph.replay()
                    torch.testing.assert_close(y,reference(r,x,'hygemm'),atol=.01,rtol=.003)
                x2=torch.randn_like(x)
                torch.testing.assert_close(r(x2),reference(r,x2,'hygemm'),atol=.01,rtol=.003)
                r.close();policies.append({'out':out,'k':k,'high':high,'tokens':t,'policy':policy,'passed':True})
    args.out.write_text(json.dumps({'passed':True,'pack':pack_records,'policies':policies,'exhaustive_finite_half_patterns':True},indent=2))


if __name__=='__main__':main()
