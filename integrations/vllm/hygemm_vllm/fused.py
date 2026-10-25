"""Opt-in, rebindable wrapper for the existing tile-boundary TMA switch kernel."""
import ctypes as C
import os
import torch
from .native import Problem
_lib=None


def library():
    global _lib
    if _lib is None:
        _lib=C.CDLL(os.environ['HYGEMM_FUSED_LIBRARY'])
        _lib.hg_fused_error.restype=C.c_char_p
        _lib.hg_fused_create.argtypes=[C.c_int,C.POINTER(Problem),C.POINTER(Problem),C.c_void_p,C.POINTER(C.c_void_p)]
        _lib.hg_fused_update.argtypes=[C.c_void_p,C.POINTER(Problem),C.POINTER(Problem)]
        _lib.hg_fused_run.argtypes=[C.c_void_p,C.c_void_p]
        _lib.hg_fused_destroy.argtypes=[C.c_void_p]
    return _lib


def check(code):
    if code:raise RuntimeError(library().hg_fused_error().decode())


def run(runtime,low,high,stream,variant=0):
    handles=runtime.fused_handles
    key=(low.tokens,variant,low.rows,high.rows)
    if key not in handles:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Fused shape must be warmed before capture')
        handle=C.c_void_p()
        check(library().hg_fused_create(variant,C.byref(low),C.byref(high),stream,C.byref(handle)))
        handles[key]=handle
    else:check(library().hg_fused_update(handles[key],C.byref(low),C.byref(high)))
    check(library().hg_fused_run(handles[key],stream))
