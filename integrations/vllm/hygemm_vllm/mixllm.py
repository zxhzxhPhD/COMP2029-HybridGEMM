# Packing/RTN adapted from MixLLM and the local reproduction.
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: MIT
"""Modern serving adapter for the original INT4/INT8 MixLLM CUDA operators."""
import os
from pathlib import Path
import torch

_loaded = False


def load_library():
    global _loaded
    if not _loaded:
        torch.ops.load_library(os.environ['MIXLLM_LIBRARY'])
        _loaded = True


def quantize_weight(w, bits):
    n, k = w.shape
    v = w.float().reshape(n, k//128, 128)
    if bits == 4:
        lo, hi = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
        s = ((hi-lo).clamp_min(1e-5)/15).half()
        z = (-torch.round(lo/s.float())).clamp(0, 15).to(torch.uint8)
        q = (torch.round(v/s.float())+z).clamp(0, 15).to(torch.uint8)
    else:
        s = (v.abs().amax(-1, keepdim=True).clamp_min(1e-5)/127).half()
        z = torch.zeros_like(s, dtype=torch.uint8)
        q = torch.round(v/s.float()).clamp(-127, 127).to(torch.int8)
    return q.reshape(n, k), s.squeeze(-1).t().contiguous(), z.squeeze(-1).t().contiguous()


def interleave_order(k, device):
    i = torch.arange(k, device=device)
    t, delta = i % 32, torch.zeros_like(i)
    for start, end, d in [(4,8,12), (8,12,-4), (12,16,8), (16,20,-8), (20,24,4), (24,28,-12)]:
        delta[(t >= start) & (t < end)] = d
    second = (torch.arange(k//8, device=device)[:, None]*8 + torch.tensor([0,4,1,5,2,6,3,7], device=device)).flatten()
    return (i+delta)[second]


def pad_cols(t, multiple):
    out = torch.zeros((t.shape[0], (t.shape[1]+multiple-1)//multiple*multiple), dtype=t.dtype, device=t.device)
    out[:, :t.shape[1]] = t
    return out


def padded_ids(ids):
    # The original epilogue speculatively reads IDs in a tail tile before its
    # store predicate. Supply readable zero padding; keep the logical numel.
    storage = torch.zeros(ids.numel()+128, dtype=torch.int32, device=ids.device)
    storage[:ids.numel()] = ids
    return storage[:ids.numel()]


class MixLinear:
    def __init__(self, weight, high_ids,plans=None):
        load_library()
        self.outputs, self.k = weight.shape
        mask = torch.zeros(self.outputs, dtype=torch.bool, device=weight.device)
        mask[high_ids] = True
        hi, lo = torch.where(mask)[0].to(torch.int32), torch.where(~mask)[0].to(torch.int32)
        q8, s8, _ = quantize_weight(weight[hi.long()], 8)
        q4, s4, z4 = quantize_weight(weight[lo.long()], 4)
        self.initialize(q8, q4, s8, s4, z4, hi, lo)
        self.plans=plans

    def initialize(self, q8, q4, s8, s4, z4, hi, lo):
        self.k = q8.shape[1] if q8.shape[0] else q4.shape[1]
        self.outputs = hi.numel()+lo.numel()
        if self.outputs % 128:
            raise ValueError('Original MixLLM serving adapter currently validates O divisible by 128; arbitrary O tails are unsupported')
        self.q8 = q8.to(torch.int8).contiguous()
        v = q4.to(torch.uint8)[:, interleave_order(self.k, q4.device)]
        self.q4 = (v[:, ::2] | (v[:, 1::2] << 4)).contiguous()
        self.s8, self.s4, self.z4 = pad_cols(s8, 2), pad_cols(s4, 2), pad_cols(z4, 4)
        self.hi, self.lo = padded_ids(hi), padded_ids(lo)
        self.calls = {}
        self.plans=None
        self.forced_config=None
        self.plans_used={}
        self.config_dir = Path(os.environ['MIXLLM_LIBRARY']).resolve().parent.parent / 'source/best_config'
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(cls, qweight, scale, zero, indices, is_int8):
        load_library()
        obj = cls.__new__(cls)
        mask = is_int8.bool()
        obj.initialize(qweight[mask].contiguous(), qweight[~mask].contiguous(), scale[:, mask].contiguous(),
                       scale[:, ~mask].contiguous(), zero[:, ~mask].contiguous(),
                       indices[mask].int(), indices[~mask].int())
        return obj

    def seed_smoke_plan(self, tokens):
        prefix = 'RM_' if not self.hi.numel() or not self.lo.numel() else ''
        name = f'{prefix}MBin-{(tokens+7)//8}_N-{self.outputs}_K-{self.k}_8bit-{self.hi.numel()//1000}'
        path = self.config_dir / name
        if not path.exists():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('MixLLM shape must be warmed before graph capture')
            # Original dispatch ID 1, five stages: conservative SM120 smoke
            # plan, NOT a tuned performance result. Upstream autotuning tries
            # large-SMEM 11-stage candidates unsupported by this GPU.
            path.write_text('5 1\nsmoke plan: original dispatch ID 1\n')

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float16 or x.device != self.q8.device or not x.is_contiguous():
            raise ValueError('MixLLM requires contiguous CUDA FP16 input matching its weights')
        selected=self.forced_config
        if self.plans:
            plan=self.plans.get(self.outputs,self.k,self.hi.numel(),x.shape[0])
            selected=(plan['stage'],plan['id'])
            self.plans_used[x.shape[0]]=[plan['tokens'],*selected]
        if selected is None:self.seed_smoke_plan(x.shape[0])
        a, sa = torch.ops.kernels_mixllm.quantize(x)
        if selected is None:
            y=torch.ops.kernels_mixllm.gemm(a,sa,self.z4,self.s8,self.s4,self.hi,self.lo,self.q8,self.q4)
        else:
            y=torch.ops.kernels_mixllm.gemm_planned(a,sa,self.z4,self.s8,self.s4,self.hi,self.lo,self.q8,self.q4,*selected)
        if self.hi.numel() and self.lo.numel():
            # Preserve the original LinearMixLLM transpose dispatch.
            y = (torch.ops.kernels_mixllm.transpose(y) if self.outputs in [4096,14336,1024,6144,28672]
                 else y.t().contiguous())
        self.calls[x.shape[0]] = self.calls.get(x.shape[0], 0)+1
        return y

    def dequant_weight(self):
        w = torch.empty((self.outputs, self.k), device=self.q8.device, dtype=torch.float32)
        if self.hi.numel():
            w[self.hi.long()] = (self.q8.float().reshape(-1, self.k//128, 128)*self.s8[:, :self.hi.numel()].t()[..., None]).reshape(-1, self.k)
        if self.lo.numel():
            q = torch.stack((self.q4 & 15, self.q4 >> 4), dim=-1).flatten(1)
            q = q[:, torch.argsort(interleave_order(self.k, q.device))].float().reshape(-1, self.k//128, 128)
            w[self.lo.long()] = ((q-self.z4[:, :self.lo.numel()].t()[..., None])*self.s4[:, :self.lo.numel()].t()[..., None]).reshape(-1, self.k)
        return w
