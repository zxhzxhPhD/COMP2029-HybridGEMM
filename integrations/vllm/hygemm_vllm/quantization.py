# Portions adapted from Microsoft's MixLLM vLLM patch.
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: MIT
"""Out-of-tree port of the MixLLM v0.9 adapter and matching HyGEMM integration.

Single-GPU dense linear layers only. Both paths retain the same vLLM attention,
KV cache, embeddings, LM head, fused projection loader and bias handling.
"""
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

_runtimes = {}


@torch.library.custom_op('hygemm_vllm::linear', mutates_args=())
def linear(x: torch.Tensor, handle: int, outputs: int) -> torch.Tensor:
    return _runtimes[handle](x)


@linear.register_fake
def linear_fake(x, handle, outputs):
    return x.new_empty((x.shape[0], outputs))


class CommonConfig(QuantizationConfig):
    kind = ''

    def __init__(self, config):
        super().__init__()
        self.selection = config.get('selection', {})
        self.format = config.get('checkpoint_format', 'mixllm_v1' if self.kind == 'mixllm' else 'fp16_rtn')
        self.low_id = int(config.get('low_id', 0))
        self.high_id = int(config.get('high_id', 0))
        self.concurrent = bool(config.get('concurrent', True))
        if self.format not in ['fp16_rtn', 'mixllm_v1'] or (self.format == 'mixllm_v1' and self.kind != 'mixllm'):
            raise ValueError('Unsupported checkpoint format')

    @classmethod
    def get_name(cls): return cls.kind
    @classmethod
    def get_supported_act_dtypes(cls): return [torch.float16]
    @classmethod
    def get_min_capability(cls): return 120 if cls.kind == 'hygemm' else 80
    @staticmethod
    def get_config_filenames(): return ['quant_config.json', 'quantize_config.json']
    @classmethod
    def from_config(cls, config): return cls(config)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if prefix.endswith('lm_head'):
                return UnquantizedLinearMethod()
            return MixedLinearMethod(self, prefix)
        return None


@register_quantization_config('mixllm')
class MixLLMConfig(CommonConfig):
    kind = 'mixllm'


@register_quantization_config('hygemm')
class HyGEMMConfig(CommonConfig):
    kind = 'hygemm'


class MixedLinearMethod(LinearMethodBase):
    def __init__(self, config, prefix):
        self.config, self.prefix = config, prefix
        self.runtime = None

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        from vllm.distributed import get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError('This integration currently validates tensor_parallel_size=1 only')
        if input_size_per_partition % 128 or params_dtype != torch.float16:
            raise ValueError('Requires FP16 activations and K divisible by 128')
        self.parts = list(output_partition_sizes)
        self.outputs, self.k = sum(self.parts), input_size_per_partition
        if self.config.format == 'fp16_rtn':
            UnquantizedLinearMethod().create_weights(layer, input_size_per_partition, output_partition_sizes,
                input_size, output_size, params_dtype, **attrs)
        else:
            # Original five-tensor checkpoint schema; default weight_loader
            # supports merged QKV/gate-up with the declared output dimensions.
            specs = [('qweight', (self.outputs, self.k), torch.int8, {'input_dim':1, 'output_dim':0}),
                     ('scale', (self.k//128, self.outputs), torch.float16, {'input_dim':0, 'output_dim':1}),
                     ('zero', (self.k//128, self.outputs), torch.uint8, {'input_dim':0, 'output_dim':1}),
                     ('indices', (self.outputs,), torch.int32, {'output_dim':0}),
                     ('is_int8_channel', (self.outputs,), torch.bool, {'output_dim':0})]
            for name, shape, dtype, dims in specs:
                p = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
                layer.register_parameter(name, p)
                set_weight_attrs(p, {**attrs, **dims})

    def selected_rows(self, device):
        if self.prefix.endswith('.qkv_proj'):
            names = [self.prefix.replace('.qkv_proj', '.'+p+'_proj') for p in ['q','k','v']]
        elif self.prefix.endswith('.gate_up_proj'):
            names = [self.prefix.replace('.gate_up_proj', '.'+p+'_proj') for p in ['gate','up']]
        else:
            names = [self.prefix]
        if len(names) != len(self.parts):
            raise ValueError(f'Projection partition mismatch: {self.prefix}, {self.parts}')
        selected, offset = [], 0
        for name, size in zip(names, self.parts):
            if name not in self.config.selection:
                raise ValueError(f'Missing channel selection for {name}')
            ids = self.config.selection[name]
            if len(set(ids)) != len(ids) or any(i < 0 or i >= size for i in ids):
                raise ValueError(f'Invalid channel map for {name}')
            selected.extend(i+offset for i in ids)
            offset += size
        return torch.tensor(sorted(selected), dtype=torch.long, device=device)

    def process_weights_after_loading(self, layer):
        if self.config.format == 'mixllm_v1':
            from .mixllm import MixLinear
            offset, start = 0, 0
            for size in self.parts:
                layer.indices.data[start:start+size] += offset
                offset += size
                start += size
            ids = layer.indices.data
            if not torch.equal(torch.sort(ids).values, torch.arange(self.outputs, device=ids.device)):
                raise ValueError('Checkpoint channel maps must form a permutation')
            self.runtime = MixLinear.from_checkpoint(layer.qweight, layer.scale, layer.zero,
                                                       layer.indices, layer.is_int8_channel)
            for name in ['qweight', 'scale', 'zero', 'indices', 'is_int8_channel']:
                delattr(layer, name)
        else:
            w = layer.weight.data.contiguous()
            ids = self.selected_rows(w.device)
            if self.config.kind == 'mixllm':
                from .mixllm import MixLinear
                self.runtime = MixLinear(w, ids)
            else:
                from .native import DualLinear
                self.runtime = DualLinear(w, ids, self.config.low_id, self.config.high_id, self.config.concurrent)
            del layer.weight
        self.handle = id(self.runtime)
        _runtimes[self.handle] = self.runtime

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        y = linear(x.reshape(-1, self.k).contiguous(), self.handle, self.outputs)
        if bias is not None:
            y = y+bias
        return y.reshape(*shape, self.outputs)


def model_audit(model):
    layers = []
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        if isinstance(method, MixedLinearMethod) and method.runtime is not None:
            r = method.runtime
            layers.append({'name':name, 'kind':method.config.kind, 'k':method.k, 'outputs':method.outputs,
                           'high':r.hi.numel(), 'low':r.lo.numel(), 'calls_by_tokens':dict(r.calls)})
    return layers
