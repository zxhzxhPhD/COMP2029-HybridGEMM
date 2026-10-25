"""Load-time permutation inside fixed precision partitions, without a shadow copy."""
import torch
from .offline_weights import logical_scales,physical_scales


def reorder_runtime_rows(runtime,mode):
    assert not runtime.handles,'Row layout must be frozen before handle creation'
    if mode not in ['reverse','block_reverse']:raise ValueError(mode)
    for ids_name,weight_name,scale_name in [('lo','w4','sf4'),('hi','w8','sf8')]:
        ids=getattr(runtime,ids_name);n=ids.numel()
        if not n:continue
        if mode=='reverse':order=torch.arange(n-1,-1,-1,device=ids.device)
        else:order=torch.cat(list(torch.arange(n,device=ids.device).split(32))[::-1])
        weight=getattr(runtime,weight_name);scales=getattr(runtime,scale_name)
        new_scales=physical_scales(logical_scales(scales,n,runtime.k)[order].contiguous(),runtime.k)
        setattr(runtime,weight_name,weight[order].contiguous())
        setattr(runtime,scale_name,new_scales)
        setattr(runtime,ids_name,ids[order].contiguous())
    runtime.row_order=mode
