"""Framework-independent GEMM ABI, with tensors and streams owned by PyTorch."""
import ctypes as C
import os
import torch


class Problem(C.Structure):
    _fields_ = [(n, C.c_int) for n in ['rows', 'tokens', 'k', 'outputs']] + [
        (n, C.c_void_p) for n in ['weight', 'activation', 'weight_scales', 'activation_scales', 'ids', 'output']]


_lib = None
PACK_MODE = int(os.environ.get('HYGEMM_PACK_MODE', '0'))
POLICIES = {'serial': 0, 'concurrent': 1, 'high_first': 2, 'high_main': 3,
            'gemv_pair1': 4, 'gemv_pair2': 5, 'gemv_pair4': 6}
FUSED_GATEUP = int(os.environ.get('HYGEMM_FUSED_GATEUP','0'))


def library():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.environ['HYGEMM_LIBRARY'])
        _lib.hg_last_error.restype = C.c_char_p
        _lib.hg_scale_size.argtypes = [C.c_int, C.c_int]
        _lib.hg_scale_size.restype = C.c_int64
        _lib.hg_pack.argtypes = [C.c_void_p]*3 + [C.c_int]*3 + [C.c_void_p]
        if hasattr(_lib, 'hg_pack_mode'):
            _lib.hg_pack_mode.argtypes = _lib.hg_pack.argtypes + [C.c_int]
        _lib.hg_create.argtypes = [C.c_int, C.c_int, C.POINTER(Problem), C.POINTER(Problem), C.c_void_p, C.POINTER(C.c_void_p)]
        _lib.hg_run.argtypes = [C.c_void_p, C.POINTER(Problem), C.POINTER(Problem), C.c_void_p, C.c_int]
        _lib.hg_destroy.argtypes = [C.c_void_p]
    return _lib


def check(status):
    if status:
        raise RuntimeError(library().hg_last_error().decode())


def pack(x, fp4=False):
    if not (x.is_cuda and x.dtype == torch.float16 and x.ndim == 2 and x.is_contiguous() and x.shape[1] % 128 == 0):
        raise ValueError('MX pack requires contiguous CUDA FP16 [rows,K], K divisible by 128')
    rows, k = x.shape
    q = torch.empty((rows, k//2 if fp4 else k), dtype=torch.uint8, device=x.device)
    sf = torch.empty(library().hg_scale_size(rows, k) if rows else 0, dtype=torch.uint8, device=x.device)
    if rows:
        params=(x.data_ptr(),q.data_ptr(),sf.data_ptr(),rows,k,int(fp4),torch.cuda.current_stream(x.device).cuda_stream)
        mode=(2 if rows>=16 else 0) if PACK_MODE==3 else PACK_MODE
        check(library().hg_pack_mode(*params,mode) if mode else library().hg_pack(*params))
    return q, sf


class DualLinear:
    def __init__(self, weight, high_ids, low_id=0, high_id=0, concurrent=True, plans=None):
        self.outputs, self.k = weight.shape
        mask = torch.zeros(self.outputs, dtype=torch.bool, device=weight.device)
        mask[high_ids] = True
        self.hi = torch.where(mask)[0].to(torch.int32)
        self.lo = torch.where(~mask)[0].to(torch.int32)
        self.w4, self.sf4 = pack(weight[self.lo.long()].contiguous(), True)
        self.w8, self.sf8 = pack(weight[self.hi.long()].contiguous(), False)
        self.low_id, self.high_id, self.concurrent = low_id, high_id, concurrent
        self.handles = {}
        self.fused_handles = {}
        self.calls = {}
        self.plans=plans
        self.plans_used={}

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.k or x.device != self.w4.device:
            raise ValueError('HyGEMM input shape/device does not match its weights')
        q, sf = pack(x)
        return self.run_packed(q,sf)

    def run_packed(self,q,sf):
        if q.dtype!=torch.uint8 or q.ndim!=2 or q.shape[1]!=self.k or q.device!=self.w4.device or not q.is_contiguous():
            raise ValueError('Invalid prepacked MXFP8 activation')
        tokens=q.shape[0]
        if sf.dtype!=torch.uint8 or sf.device!=q.device or not sf.is_contiguous() or sf.numel()!=library().hg_scale_size(tokens,self.k):
            raise ValueError('Invalid prepacked MXFP8 scale buffer')
        y = torch.empty((tokens, self.outputs), dtype=torch.float16, device=q.device)
        l = Problem(self.lo.numel(), tokens, self.k, self.outputs, self.w4.data_ptr(), q.data_ptr(),
                    self.sf4.data_ptr(), sf.data_ptr(), self.lo.data_ptr(), y.data_ptr())
        h = Problem(self.hi.numel(), tokens, self.k, self.outputs, self.w8.data_ptr(), q.data_ptr(),
                    self.sf8.data_ptr(), sf.data_ptr(), self.hi.data_ptr(), y.data_ptr())
        stream = torch.cuda.current_stream(q.device).cuda_stream
        if FUSED_GATEUP and tokens<=FUSED_GATEUP and getattr(self,'layer_name','').endswith('.gate_up_proj'):
            from .fused import run
            run(self,l,h,stream)
            self.plans_used[tokens]=[FUSED_GATEUP,0,0,'fused_tma']
            self.calls[tokens]=self.calls.get(tokens,0)+1
            return y
        li,hi,concurrent=self.low_id,self.high_id,self.concurrent
        if self.plans:
            plan=self.plans.get(self.outputs,self.k,self.hi.numel(),tokens)
            li,hi,concurrent=plan['low_id'],plan['high_id'],POLICIES[plan['policy']]
            self.plans_used[tokens]=[plan['tokens'],li,hi,concurrent]
        key = (tokens,li,hi,self.lo.numel(),self.hi.numel())
        if key not in self.handles:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'HyGEMM shape T={key} must be warmed before graph capture')
            handle = C.c_void_p()
            check(library().hg_create(li,hi,C.byref(l),C.byref(h),stream,C.byref(handle)))
            self.handles[key] = handle
        check(library().hg_run(self.handles[key], C.byref(l), C.byref(h), stream, concurrent))
        self.calls[tokens] = self.calls.get(tokens, 0) + 1
        return y

    def close(self):
        for h in self.handles.values():
            library().hg_destroy(h)
        self.handles.clear()
        if self.fused_handles:
            from .fused import library as fused_library
            for h in self.fused_handles.values():fused_library().hg_fused_destroy(h)
            self.fused_handles.clear()
