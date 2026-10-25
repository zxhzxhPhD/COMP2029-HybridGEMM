"""Offline E2M1/E8M0 exponent search. No online quantizer or GEMM changes."""
import torch
from hygemm_vllm.native import pack
from hygemm_vllm.offline_weights import logical_scales


def unpack(q):return torch.stack((q&15,q>>4),dim=-1).flatten(1)


def codes_rne(x):
    boundaries=x.new_tensor([.25,.75,1.25,1.75,2.5,3.5,5.])
    magnitude=x.abs().contiguous()
    code=torch.bucketize(magnitude,boundaries,right=False)
    tie=(code<7)&(magnitude==boundaries[code.clamp(max=6)])
    code=code+((code&1).bool()&tie).long()
    return (code.to(torch.uint8)|(torch.signbit(x).to(torch.uint8)<<3))


def pack_codes(code):return (code[:,0::2]|(code[:,1::2]<<4)).contiguous()


def decode_logical(q,scales,k):
    code=unpack(q).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=q.device,dtype=torch.float32)
    return (lut[code].reshape(q.shape[0],k//32,32)*torch.exp2(scales.float()-127)[:,:,None]).reshape(q.shape[0],k)


def quantize(w,importance=None,offsets=(0,-1,-2)):
    assert w.dtype==torch.float16 and w.is_cuda and torch.isfinite(w).all()
    rows,k=w.shape
    q,sf=pack(w.contiguous(),True);logical=logical_scales(sf,rows,k)
    source=w.float().reshape(rows,k//32,32)
    weight=(torch.ones(k,device=w.device) if importance is None else importance.float().to(w.device)).reshape(1,k//32,32)
    assert torch.isfinite(weight).all() and (weight>=0).all()
    decoded=decode_logical(q,logical,k).reshape_as(source)
    original_error=((decoded-source).square()*weight).sum(-1)
    best_error=original_error.clone();best_codes=unpack(q).reshape(rows,k//32,32);best_scale=logical.clone()
    for shift in offsets:
        if shift==0:continue
        exponent=(logical.to(torch.int16)+shift).clamp(0,254).to(torch.uint8)
        scales=torch.exp2(exponent.float()-127)
        codes=codes_rne(source/scales[:,:,None])
        values=decode_logical(pack_codes(codes.reshape(rows,k)),exponent,k).reshape_as(source)
        error=((values-source).square()*weight).sum(-1)
        better=error<best_error
        best_error=torch.where(better,error,best_error)
        best_codes=torch.where(better[:,:,None],codes,best_codes)
        best_scale=torch.where(better,exponent,best_scale)
    result=pack_codes(best_codes.reshape(rows,k))
    final=decode_logical(result,best_scale,k).reshape_as(source)
    check=((final-source).square()*weight).sum(-1)
    assert torch.all(check<=original_error*(1+1e-6)+1e-25)
    shifts=logical.int()-best_scale.int()
    stats={'groups':rows*(k//32),'shift_groups':{str(-s):int((shifts==-s).sum()) for s in offsets},
        'weighted_error_before':original_error.double().sum().item(),'weighted_error_after':check.double().sum().item(),
        'weight_error_before':(decoded-source).square().double().sum().item(),
        'weight_error_after':(final-source).square().double().sum().item()}
    return result,best_scale,stats
