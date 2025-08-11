from tinygrad import Tensor
from einops import rearrange

from comfy.ldm.modules.attention_stub import optimized_attention
import comfy.model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        q = q.cast(pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k = k.cast(pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)
        q = (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1]).reshape(*q_shape).cast(v.dtype)
        k = (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1]).reshape(*k_shape).cast(v.dtype)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu() or comfy.model_management.is_directml_enabled():
        device = "CPU"
    else:
        device = pos.device

    scale = Tensor.arange(0, dim//2).cast(Tensor.default_type) * (2.0 / dim)
    omega = 1.0 / (theta**scale)
    out = pos.cast(Tensor.default_type).unsqueeze(-1) * omega
    out = Tensor.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.cast(pos.dtype)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.cast(freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.cast(freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)

