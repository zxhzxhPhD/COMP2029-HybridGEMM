"""Frozen native MX arithmetic and experiment IO; no serving code changes."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_components import decode_mx
from hygemm_vllm.native import pack


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def tensor_sha(x):
    return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()


def dq(x, fp4=False):
    q, sf = pack(x.contiguous(), fp4)
    return decode_mx(q, sf, x.shape[-1], fp4)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def aggregate(blocks):
    count = sum(x['tokens'] for x in blocks)
    nll = math.fsum(x['nll'] for x in blocks)
    return {'nll': nll, 'tokens': count, 'mean_nll': nll/count, 'ppl': math.exp(nll/count), 'blocks': blocks}


def fused_cases(selection, shapes, num_layers):
    layers = []
    for layer in range(num_layers):
        stem = f'model.layers.{layer}.'
        for path, parts in [('self_attn.qkv_proj', ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj']),
                            ('self_attn.o_proj', ['self_attn.o_proj']),
                            ('mlp.gate_up_proj', ['mlp.gate_proj','mlp.up_proj']), ('mlp.down_proj', ['mlp.down_proj'])]:
            offset, ids, ks = 0, [], set()
            for part in parts:
                name = stem + part
                o,k = shapes[name]
                chosen = selection[name]
                assert len(set(chosen)) == len(chosen) and all(0 <= i < o for i in chosen)
                ids.extend(offset+i for i in chosen)
                offset += o; ks.add(k)
            assert len(ks) == 1
            layers.append({'layer': stem+path, 'out': offset, 'k': ks.pop(), 'high': len(ids), 'high_ids': sorted(ids)})
    return layers


def analytic_budget(layers):
    payload = sum(4*(r['out']+r['high'])*r['k'] for r in layers)
    scales = sum(((r['out']-r['high']+127)//128+(r['high']+127)//128)*(r['k']//128)*512 for r in layers)
    indices = sum(4*r['out'] for r in layers)
    return {'payload_bits': payload, 'weight_bytes': payload//8, 'weight_scale_bytes': scales,
            'channel_index_bytes': indices, 'persistent_quantized_bytes': payload//8+scales+indices}
