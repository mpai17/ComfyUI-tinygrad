import math
from dataclasses import dataclass

from tinygrad import Tensor

from .math import attention, rope
import comfy.ops
import comfy.ldm.common_dit


class EmbedND:
    def __init__(self, dim: int, theta: int, axes_dim: list):
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = Tensor.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = (-math.log(max_period) * Tensor.arange(start=0, end=half, dtype=Tensor.default_type) / half).exp()

    args = t[:, None].float() * freqs[None]
    embedding = Tensor.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        embedding = Tensor.cat([embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1)
    # tinygrad tensors are always floating point
    embedding = embedding.cast(t.dtype)
    return embedding

class MLPEmbedder:
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        if operations is None:
            operations = comfy.ops
        self.in_layer = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        return self.out_layer(self.in_layer(x).silu())


class RMSNorm:
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        self.scale = Tensor.empty((dim,), dtype=dtype or Tensor.default_type)

    def __call__(self, x: Tensor):
        return comfy.ldm.common_dit.rms_norm(x, self.scale, 1e-6)


class QKNorm:
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.cast(v.dtype), k.cast(v.dtype)


class SelfAttention:
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        if operations is None:
            operations = comfy.ops
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
    
    def __call__(self, x: Tensor, pe: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, heads=self.num_heads, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation:
    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None):
        if operations is None:
            operations = comfy.ops
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def __call__(self, vec: Tensor) -> tuple:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(vec.silu()).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        if m_add is not None:
            return m_add + tensor * m_mult  # equivalent to torch.addcmul
        else:
            return tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] = tensor[:, d[0]:d[1]] * m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] = tensor[:, d[0]:d[1]] + m_add[:, d[2]]
        return tensor


class DoubleStreamBlock:
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, flipped_img_txt=False, dtype=None, device=None, operations=None):
        if operations is None:
            operations = comfy.ops

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp_1 = operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device)
        self.img_mlp_2 = operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device)

        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp_1 = operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device)
        self.txt_mlp_2 = operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.flipped_img_txt = flipped_img_txt
    
    def img_mlp(self, x):
        return self.img_mlp_2(self.img_mlp_1(x).gelu())
    
    def txt_mlp(self, x):
        return self.txt_mlp_2(self.txt_mlp_1(x).gelu())

    def __call__(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
            # run actual attention
            attn = attention(Tensor.cat((img_q, txt_q), dim=2),
                             Tensor.cat((img_k, txt_k), dim=2),
                             Tensor.cat((img_v, txt_v), dim=2),
                             pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            # run actual attention
            attn = attention(Tensor.cat((txt_q, img_q), dim=2),
                             Tensor.cat((txt_k, img_k), dim=2),
                             Tensor.cat((txt_v, img_v), dim=2),
                             pe=pe, mask=attn_mask)

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # calculate the img bloks
        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

        # calculate the txt bloks
        txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

        if 'float16' in str(txt.dtype):
            # tinygrad handles NaN/inf automatically, but we can clamp if needed
            txt = txt.clip(-65504, 65504)

        return img, txt


class SingleStreamBlock:
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        # GELU activation will be applied directly to tensors
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)

    def __call__(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None) -> Tensor:
        mod, _ = self.modulation(vec)
        linear1_out = self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims))
        qkv = linear1_out[..., :3 * self.hidden_size]
        mlp = linear1_out[..., 3 * self.hidden_size:]

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(Tensor.cat((attn, mlp.gelu()), 2))
        x += apply_mod(output, mod.gate, None, modulation_dims)
        if 'float16' in str(x.dtype):
            # tinygrad handles NaN/inf automatically, but we can clamp if needed
            x = x.clip(-65504, 65504)
        return x


class LastLayer:
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.ada_linear = operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device)

    def __call__(self, x: Tensor, vec: Tensor, modulation_dims=None) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.ada_linear(vec.silu()).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        x = self.linear(x)
        return x
