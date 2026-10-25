"""Diagnostic graph topology only; existing PyTorch sleep/fill CUDA kernels."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--trace', action='store_true')
ap.add_argument('--topology', action='store_true')
args = ap.parse_args()
lib = C.CDLL('/usr/local/cuda/lib64/libcudart.so')
lib.cudaMemsetAsync.argtypes = [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p]
torch.set_num_threads(4)
buf = torch.empty(14336, dtype=torch.uint8, device='cuda')
nodes, cycles = 128, 10000
torch.cuda._sleep(cycles)
buf.fill_(127)
torch.cuda.synchronize()

def memset():
    status = lib.cudaMemsetAsync(buf.data_ptr(), 127, buf.numel(), torch.cuda.current_stream().cuda_stream)
    assert status == 0

graphs = {}
names = (['forkjoin2', 'forkjoin3', 'forkjoin2_memset', 'forkjoin3_memset'] if args.topology else
         ['kernel_chain', 'one_memset', 'each_memset', 'each_fill_kernel'])
for name in names:
    graph = torch.cuda.CUDAGraph()
    if args.topology:
        sides = [torch.cuda.Stream() for _ in range(2)]
        fork, join0, join1 = [torch.cuda.Event() for _ in range(3)]
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        if name == 'one_memset':
            memset()
        for i in range(nodes):
            if name == 'each_memset' or name.endswith('_memset'):
                memset()
            elif name == 'each_fill_kernel':
                buf.fill_(127)
            if name.startswith('forkjoin'):
                fork.record(main)
                three = name.startswith('forkjoin3')
                low = sides[0] if three else main
                if three:
                    low.wait_event(fork)
                with torch.cuda.stream(low):
                    torch.cuda._sleep(cycles)
                    join0.record()
                sides[1].wait_event(fork)
                with torch.cuda.stream(sides[1]):
                    torch.cuda._sleep(cycles)
                    join1.record()
                if three:
                    main.wait_event(join0)
                main.wait_event(join1)
                torch.cuda._sleep(1000)
            else:
                torch.cuda._sleep(cycles)
    graphs[name] = graph
    for _ in range(4):
        graph.replay()
torch.cuda.synchronize()
results = {'sleep_nodes': nodes, 'sleep_cycles': cycles, 'memset_bytes': buf.numel(), 'trace': args.trace, 'cases': {}}
if args.trace:
    torch.cuda.cudart().cudaProfilerStart()
for name, graph in graphs.items():
    samples = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        a.record()
        with torch.cuda.nvtx.range('probe|' + name):
            for _ in range(8):
                graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / 8)
    results['cases'][name] = {'samples_ms': samples, 'median_ms': statistics.median(samples),
                              'cv': statistics.pstdev(samples) / statistics.mean(samples)}
if args.trace:
    torch.cuda.cudart().cudaProfilerStop()
args.out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
