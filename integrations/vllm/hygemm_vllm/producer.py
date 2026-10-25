"""Opt-in Qwen2 SwiGLU+pack experiment; the FP16 intermediate has one consumer."""
import ctypes as C
import os
import torch
from .native import library
from .quantization import _runtimes

_producer=None


def silu_pack(x,math=0):
    global _producer
    if _producer is None:
        _producer=C.CDLL(os.environ['HYGEMM_PRODUCER_LIBRARY'])
        _producer.hg_silu_pack.argtypes=[C.c_void_p]*4+[C.c_int]*2+[C.c_void_p,C.c_int]
    assert x.is_contiguous() and x.dtype==torch.float16 and x.ndim==2
    rows,k=x.shape[0],x.shape[1]//2
    assert x.shape[1]==2*k and k%128==0
    q=torch.empty((rows,k),device=x.device,dtype=torch.uint8)
    sf=torch.empty(library().hg_scale_size(rows,k),device=x.device,dtype=torch.uint8)
    status=_producer.hg_silu_pack(x.data_ptr(),q.data_ptr(),sf.data_ptr(),None,rows,k,
        torch.cuda.current_stream(x.device).cuda_stream,math)
    if status:raise RuntimeError(f'SwiGLU MX producer CUDA status {status}')
    return q,sf


@torch.library.custom_op('hygemm_vllm::linear_silu',mutates_args=())
def linear_silu(x:torch.Tensor,handle:int,outputs:int,math:int)->torch.Tensor:
    runtime=_runtimes[handle]
    if hasattr(runtime,'profile_name') and torch.cuda.is_current_stream_capturing():
        label=f'projection|{runtime.profile_name}|O={runtime.outputs}|K={runtime.k}|H={runtime.hi.numel()}|T={x.shape[0]}'
        with torch.cuda.nvtx.range(label):
            q,sf=silu_pack(x,math)
            return runtime.run_packed(q,sf)
    q,sf=silu_pack(x,math)
    return runtime.run_packed(q,sf)


@linear_silu.register_fake
def linear_silu_fake(x,handle,outputs,math):return x.new_empty((x.shape[0],outputs))


def install():
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from .offline import OfflineLinearMethod
    if getattr(Qwen2MLP,'_hygemm_producer_installed',False):return
    original=Qwen2MLP.forward
    math=int(os.environ.get('HYGEMM_SILU_MATH','0'))
    def forward(self,x):
        method=getattr(self.down_proj,'quant_method',None)
        if not isinstance(method,OfflineLinearMethod) or x.numel()//x.shape[-1]>128:
            return original(self,x)
        assert self.down_proj.bias is None
        gate_up,_=self.gate_up_proj(x)
        result=linear_silu(gate_up.reshape(-1,2*method.k),method.handle,method.outputs,math)
        return result.reshape(*x.shape[:-1],method.outputs)
    Qwen2MLP.forward=forward
    Qwen2MLP._hygemm_producer_installed=True
