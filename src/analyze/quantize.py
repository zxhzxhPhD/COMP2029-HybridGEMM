"""Numerical simulation of MXFP4 and MXFP8 block quantization.

MX formats use one power-of-two UE8M0 scale for every 32 values.  This
module returns dequantized float tensors; it models the numerical result,
not the packed byte layout used by a kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


MX_BLOCK_SIZE = 32
E2M1_MAX = 6.0
E4M3_MAX = 448.0

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_e2m1_unscaled(x: torch.Tensor) -> torch.Tensor:
    """Round values already normalized by their MX scale to E2M1."""
    levels = x.new_tensor(_E2M1_LEVELS)
    magnitude = x.abs().clamp(max=E2M1_MAX)
    indices = (magnitude.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return torch.sign(x) * levels[indices]


def _quantize_e4m3_unscaled(x: torch.Tensor) -> torch.Tensor:
    """Round normalized values to NVIDIA's finite E4M3 encoding."""
    x = x.clamp(-E4M3_MAX, E4M3_MAX)
    # PyTorch implements the exact E4M3FN conversion on CPU and CUDA.
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def _ue8m0_scale(amax: torch.Tensor, element_max: float) -> torch.Tensor:
    """Choose the smallest power-of-two scale that avoids element overflow.

    UE8M0 conversion uses round-toward-positive-infinity, hence ceil(log2).
    The representable exponent range is clamped to [-127, 127].
    """
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(safe_amax / element_max)).clamp(-127, 127)
    scale = torch.pow(2.0, exponent)
    return torch.where(amax > 0, scale, torch.ones_like(scale))


def quantize_mx(
    x: torch.Tensor,
    element_format: str,
    *,
    block_dim: int,
    block_size: int = MX_BLOCK_SIZE,
) -> torch.Tensor:
    """Quantize ``x`` with MX block scaling along ``block_dim``.

    Each slice along all other dimensions gets an independent scale for each
    consecutive K32 block. Partial trailing blocks are zero padded.
    """
    if x.ndim == 0:
        raise ValueError("MX quantization requires a tensor with at least one dimension")
    block_dim %= x.ndim
    moved = x.movedim(block_dim, -1).contiguous()
    original_k = moved.shape[-1]
    pad_k = (-original_k) % block_size
    if pad_k:
        moved = F.pad(moved, (0, pad_k))

    blocked = moved.reshape(*moved.shape[:-1], -1, block_size)
    amax = blocked.abs().amax(dim=-1, keepdim=True)

    if element_format == "e2m1":
        element_max = E2M1_MAX
        quantizer = _quantize_e2m1_unscaled
    elif element_format == "e4m3":
        element_max = E4M3_MAX
        quantizer = _quantize_e4m3_unscaled
    else:
        raise ValueError(f"Unsupported MX element format: {element_format}")

    scale = _ue8m0_scale(amax, element_max)
    dequantized = quantizer(blocked / scale) * scale
    dequantized = dequantized.reshape(*moved.shape)[..., :original_k]
    return dequantized.movedim(-1, block_dim)


def quantize_mxfp4(x: torch.Tensor, *, block_dim: int) -> torch.Tensor:
    return quantize_mx(x, "e2m1", block_dim=block_dim)


def quantize_mxfp8_e4m3(x: torch.Tensor, *, block_dim: int) -> torch.Tensor:
    return quantize_mx(x, "e4m3", block_dim=block_dim)


def quantize_w4a4(w: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 weights and activations, both block-scaled along GEMM K."""
    return quantize_mxfp4(w, block_dim=1), quantize_mxfp4(x, block_dim=0)


def quantize_w4a8(w: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 weights and MXFP8-E4M3 activations, block-scaled along K."""
    return quantize_mxfp4(w, block_dim=1), quantize_mxfp8_e4m3(x, block_dim=0)


def quantize_w8a4(w: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-E4M3 weights and MXFP4 activations, block-scaled along K."""
    return quantize_mxfp8_e4m3(w, block_dim=1), quantize_mxfp4(x, block_dim=0)
