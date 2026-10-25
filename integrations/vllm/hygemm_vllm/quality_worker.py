"""Opt-in, read-only budget audit for the fixed-count selection experiment."""
from .worker import AuditWorkerExtension


class QualityWorkerExtension(AuditWorkerExtension):
    def quality_budget(self):
        import hashlib
        import os
        from pathlib import Path
        import torch
        from .quantization import MixedLinearMethod
        from .native import library
        rows=[]; quant_storages={}
        for name,module in self.model_runner.model.named_modules():
            method=getattr(module,'quant_method',None)
            if not isinstance(method,MixedLinearMethod):continue
            r=method.runtime
            assert method.config.kind=='hygemm' and not hasattr(module,'weight')
            tensors={k:v for k,v in vars(r).items() if isinstance(v,torch.Tensor)}
            assert set(tensors)=={'w4','w8','sf4','sf8','lo','hi'},set(tensors)
            values={}
            for key,t in tensors.items():
                storage=t.untyped_storage()
                values[key]={'shape':list(t.shape),'dtype':str(t.dtype),'logical_bytes':t.numel()*t.element_size(),
                    'storage_bytes':storage.nbytes()}
                if storage.nbytes():quant_storages[storage.data_ptr()]=storage.nbytes()
            for ids,sf in [(r.lo,r.sf4),(r.hi,r.sf8)]:
                assert sf.numel()==(library().hg_scale_size(ids.numel(),r.k) if ids.numel() else 0)
                assert sf.numel()==((ids.numel()+127)//128)*(r.k//128)*512
            assert torch.equal(torch.sort(torch.cat([r.lo,r.hi])).values,torch.arange(r.outputs,device=r.lo.device,dtype=torch.int32))
            rows.append({'name':name,'out':r.outputs,'k':r.k,'high':r.hi.numel(),'low':r.lo.numel(),
                         'no_fp16_weight_parameter':True,'tensors':values})
        other={}
        for _,t in list(self.model_runner.model.named_parameters())+list(self.model_runner.model.named_buffers()):
            storage=t.untyped_storage()
            if storage.nbytes() and storage.data_ptr() not in quant_storages:other[storage.data_ptr()]=storage.nbytes()
        path=Path(os.environ['HYGEMM_LIBRARY'])
        return {'layers':rows,'persistent_quantized_storage_bytes':sum(quant_storages.values()),
                'remaining_registered_parameter_buffer_bytes':sum(other.values()),
                'model_representation_bytes':sum(quant_storages.values())+sum(other.values()),
                'scope':'Unique tensor storages for quantized projections plus remaining registered model parameters/buffers; excludes KV, Graph pools, allocator slack, and transient workspace.',
                'library_path':str(path),'library_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
