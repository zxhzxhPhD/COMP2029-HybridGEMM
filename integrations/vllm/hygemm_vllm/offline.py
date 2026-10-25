"""Opt-in Q2/Q3 loader. All forward computation remains the existing DualLinear."""
import hashlib
import json
from pathlib import Path
import torch
from vllm.model_executor.layers.linear import LinearBase,UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from .quantization import HyGEMMConfig,MixedLinearMethod
from .plans import PlanTable
from .offline_weights import replace_low_weight


class InheritedLayerPlans:
    def __init__(self,base,reference,actual,override=None):
        self.base,self.reference,self.actual,self.override=base,reference,actual,override

    def get(self,outputs,k,high,tokens):
        assert (outputs,k,high)==self.actual
        if self.override is not None and tokens<=128:
            return self.override.get(outputs,k,high,tokens)
        # Retain the same per-layer arithmetic/tile configuration as Q1, while
        # permitting the new high partition width. No missing-shape fallback.
        return self.base.get(outputs,k,self.reference['high'],tokens)


@register_quantization_config('hygemm_offline')
class OfflineConfig(HyGEMMConfig):
    kind='hygemm'

    def __init__(self,config):
        super().__init__({k:v for k,v in config.items() if k!='tuning_plan'})
        self.reference_plans=PlanTable(config['reference_plans'],'hygemm')
        self.reference_layers={r['layer']:r for r in json.loads(Path(config['reference_cases']).read_text())['layers']}
        self.bank_path=Path(config['q4_bank']) if config.get('q4_bank') else None
        self.bank=json.loads(self.bank_path.read_text()) if self.bank_path else None
        self.decode_plans=PlanTable(config['decode_plans'],'hygemm') if config.get('decode_plans') else None

    @classmethod
    def get_name(cls):return 'hygemm_offline'

    def get_quant_method(self,layer,prefix):
        if isinstance(layer,LinearBase):
            return UnquantizedLinearMethod() if prefix.endswith('lm_head') else OfflineLinearMethod(self,prefix)
        return None


class OfflineLinearMethod(MixedLinearMethod):
    def process_weights_after_loading(self,layer):
        super().process_weights_after_loading(layer)
        r=self.runtime
        reference=self.config.reference_layers[self.prefix]
        assert (reference['out'],reference['k'])==(r.outputs,r.k)
        r.plans=InheritedLayerPlans(self.config.reference_plans,reference,(r.outputs,r.k,r.hi.numel()),self.config.decode_plans)
        if self.config.bank is None or not r.lo.numel():return
        if self.prefix.endswith('.qkv_proj'):
            names=[self.prefix.replace('.qkv_proj','.'+p+'_proj') for p in ['q','k','v']]
        elif self.prefix.endswith('.gate_up_proj'):
            names=[self.prefix.replace('.gate_up_proj','.'+p+'_proj') for p in ['gate','up']]
        else:names=[self.prefix]
        qs=[];scales=[]
        for name,rows in zip(names,self.parts):
            spec=self.config.bank['layers'][name]
            path=self.config.bank_path.parent/spec['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            value=torch.load(path,map_location='cpu',weights_only=True)
            assert value['q4'].shape==(rows,r.k//2)
            assert value['scales'].shape==(rows,r.k//32)
            qs.append(value['q4']);scales.append(value['scales'])
        replace_low_weight(r,torch.cat(qs),torch.cat(scales),verify_native=self.config.bank['mode']=='native_rtn')
