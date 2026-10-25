"""Load-time MXFP4 tensor replacement; native GEMM/activation pack are unchanged."""
import torch
from .native import library


def scale_indices(rows,k,device):
    r=torch.arange(rows,device=device)[:,None]
    g=torch.arange(k//32,device=device)[None,:]
    return ((r//128)*(k//128)+g//4)*512+(r%32)*16+((r%128)//32)*4+g%4


def logical_scales(sf,rows,k):
    return sf[scale_indices(rows,k,sf.device)].contiguous()


def physical_scales(values,k):
    rows=values.shape[0]
    assert values.shape==(rows,k//32) and values.dtype==torch.uint8
    result=torch.full((library().hg_scale_size(rows,k) if rows else 0,),127,dtype=torch.uint8,device=values.device)
    if rows:result[scale_indices(rows,k,values.device)]=values
    return result


def replace_low_weight(runtime,q4,scales,verify_native=False):
    assert not runtime.handles,'Offline weights must be installed before any GEMM/capture'
    assert q4.dtype==scales.dtype==torch.uint8
    assert q4.shape==(runtime.outputs,runtime.k//2)
    assert scales.shape==(runtime.outputs,runtime.k//32)
    ids=runtime.lo.to(device=q4.device,dtype=torch.long)
    packed=q4.index_select(0,ids).contiguous().to(runtime.w4.device)
    logical=scales.index_select(0,ids.to(scales.device)).contiguous().to(runtime.sf4.device)
    sf=physical_scales(logical,runtime.k)
    assert packed.shape==runtime.w4.shape and sf.shape==runtime.sf4.shape
    if verify_native:
        assert torch.equal(packed,runtime.w4),'Native-RTN bank payload mismatch'
        assert torch.equal(sf,runtime.sf4),'Native-RTN bank scale/padding mismatch'
    runtime.w4,runtime.sf4=packed,sf
