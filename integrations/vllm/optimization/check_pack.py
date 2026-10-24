"""Byte-for-byte parity with the original packer, including all padding."""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from hygemm_vllm.native import pack, library

ap = argparse.ArgumentParser()
ap.add_argument('--old-library', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
torch.manual_seed(20261025)
old = C.CDLL(str(args.old_library))
old.hg_pack.argtypes = [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_void_p]
records = []
for rows in [1, 3, 8, 17, 31, 32, 33, 64, 127, 128, 129]:
    for k in [128, 896, 3584, 18944]:
        for fp4 in [False, True]:
            x = torch.randn(rows, k, device='cuda', dtype=torch.float16)
            q, sf = pack(x, fp4)
            expected_q, expected_sf = torch.empty_like(q), torch.empty_like(sf)
            assert old.hg_pack(x.data_ptr(), expected_q.data_ptr(), expected_sf.data_ptr(), rows, k, int(fp4), torch.cuda.current_stream().cuda_stream) == 0
            assert torch.equal(q, expected_q) and torch.equal(sf, expected_sf)
            if not fp4 and rows <= 32:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out_q, out_sf = pack(x)
                for scale in [0., .125, 2.]:
                    x.copy_(torch.randn_like(x) * scale)
                    graph.replay()
                    assert old.hg_pack(x.data_ptr(), expected_q.data_ptr(), expected_sf.data_ptr(), rows, k, 0, torch.cuda.current_stream().cuda_stream) == 0
                    assert torch.equal(out_q, expected_q) and torch.equal(out_sf, expected_sf)
            records.append({'rows': rows, 'k': k, 'fp4': fp4, 'all_bytes_equal': True})
            print('Equal', rows, k, fp4, flush=True)
args.out.write_text(json.dumps({'passed': True, 'records': records}, indent=2))
