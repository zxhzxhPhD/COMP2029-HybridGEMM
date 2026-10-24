"""Full-output lightweight-variant correctness, tails, rebinding and Graphs."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import torch
from hygemm_vllm.native import DualLinear, library, check

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_components import reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ids', default=','.join(map(str, range(8, 28))))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sanitizer', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    ids = list(map(int, args.ids.split(',')))
    lib = library()
    lib.hg_info.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
    records, inventory = [], {}
    shapes = [(257, 256, 26)] if args.smoke else [(257, k, h) for k in [128, 896, 3584, 18944] for h in [0, 26, 257]]
    if args.sanitizer:
        shapes = [(257, 896, 26), (65, 128, 0), (65, 256, 65)]
    for variant in ids:
        for outputs, k, high in shapes:
            w = (torch.randn(outputs, k, device='cuda') * .04).half()
            selected = torch.randperm(outputs, device='cuda')[:high]
            r = DualLinear(w, selected, variant, variant)
            for tokens in ([3] if args.smoke else [1, 3, 17] if args.sanitizer else [1, 3, 8, 17, 32, 65]):
                x = torch.randn(tokens, k, device='cuda', dtype=torch.float16)
                y = r(x)
                expected = reference(r, x, 'hygemm')
                torch.testing.assert_close(y, expected, atol=.002, rtol=.002)
                error = (y.float() - expected.float()).abs().max().item()
                values = (C.c_int * 20)()
                handle = r.handles[(tokens, variant, variant, r.lo.numel(), r.hi.numel())]
                check(lib.hg_info(handle, values))
                for part in range(2):
                    item = list(values)[part * 10:(part + 1) * 10]
                    if item[0] >= 0:
                        inventory[f'{variant}_{part}'] = item
                if tokens == 3:
                    x2 = torch.randn_like(x) * .5
                    torch.testing.assert_close(r(x2), reference(r, x2, 'hygemm'), atol=.002, rtol=.002)
                records.append({'variant': variant, 'outputs': outputs, 'k': k, 'high': high, 'tokens': tokens,
                                'max_abs': error, 'passed': True})
            if high == 26:
                x = torch.randn(3, k, device='cuda', dtype=torch.float16)
                r(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = r(x)
                for scale in [0, .25, 2.0]:
                    x.copy_(torch.randn_like(x) * scale)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(y, reference(r, x, 'hygemm'), atol=.002, rtol=.002)
                records.append({'variant': variant, 'k': k, 'changed_input_graph': True, 'passed': True})
            r.close()
            print('Passed', variant, outputs, k, high, flush=True)
        args.out.write_text(json.dumps({'passed': True, 'records': records, 'inventory': inventory,
                                        'info_fields': ['id', 'tile_m', 'tile_n', 'tile_k', 'stages', 'registers', 'shared_bytes', 'local_bytes', 'fp8', 'pingpong']}, indent=2))


if __name__ == '__main__':
    main()
