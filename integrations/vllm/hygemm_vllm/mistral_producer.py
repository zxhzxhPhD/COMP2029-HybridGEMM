"""Explicit TP1 Mistral/Llama MLP adapter for the already verified SiLU producer."""
import os


def install():
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.model_executor.models.mistral import MistralMLP
    for mlp_class in [LlamaMLP,MistralMLP]:
        _install_class(mlp_class)


def _install_class(mlp_class):
    from .offline import OfflineLinearMethod
    from .producer import linear_silu
    if getattr(mlp_class,'_hygemm_producer_installed',False):return
    original=mlp_class.forward
    math=int(os.environ.get('HYGEMM_SILU_MATH','0'))
    def forward(self,x):
        method=getattr(self.down_proj,'quant_method',None)
        if not isinstance(method,OfflineLinearMethod) or x.numel()//x.shape[-1]>128:
            return original(self,x)
        assert self.down_proj.bias is None
        gate_up,_=self.gate_up_proj(x)
        result=linear_silu(gate_up.reshape(-1,2*method.k),method.handle,method.outputs,math)
        return result.reshape(*x.shape[:-1],method.outputs)
    mlp_class.forward=forward
    mlp_class._hygemm_producer_installed=True
