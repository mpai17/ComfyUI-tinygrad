#original code from https://github.com/genmoai/models under apache 2.0 license

# import functools
import math

from tinygrad import Tensor


def centers(start: float, stop, num, dtype=None, device=None):
    """linspace through bin centers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        num (int): Number of points.
        dtype: Data type of the points.
        device: Device of the points (ignored in tinygrad).

    Returns:
        centers (Tensor): Centers of the bins. Shape: (num,).
    """
    edges = Tensor.linspace(start, stop, num + 1, dtype=dtype)
    return (edges[:-1] + edges[1:]) / 2


# @functools.lru_cache(maxsize=1)
def create_position_matrix(
    T: int,
    pH: int,
    pW: int,
    device,  # ignored in tinygrad
    dtype,
    *,
    target_area: float = 36864,
):
    """
    Args:
        T: int - Temporal dimension
        pH: int - Height dimension after patchify
        pW: int - Width dimension after patchify

    Returns:
        pos: [T * pH * pW, 3] - position matrix
    """
    # Create 1D tensors for each dimension
    t = Tensor.arange(T, dtype=dtype)

    # Positionally interpolate to area 36864.
    # (3072x3072 frame with 16x16 patches = 192x192 latents).
    # This automatically scales rope positions when the resolution changes.
    # We use a large target area so the model is more sensitive
    # to changes in the learned pos_frequencies matrix.
    scale = math.sqrt(target_area / (pW * pH))
    w = centers(-pW * scale / 2, pW * scale / 2, pW)
    h = centers(-pH * scale / 2, pH * scale / 2, pH)

    # Use meshgrid to create 3D grids
    # tinygrad meshgrid equivalent
    grid_t = t.reshape(T, 1, 1).expand(T, pH, pW)
    grid_h = h.reshape(1, pH, 1).expand(T, pH, pW)
    grid_w = w.reshape(1, 1, pW).expand(T, pH, pW)

    # Stack and reshape the grids.
    pos = Tensor.stack([grid_t, grid_h, grid_w], dim=-1)  # [T, pH, pW, 3]
    pos = pos.view(-1, 3)  # [T * pH * pW, 3]
    pos = pos.cast(dtype)

    return pos


def compute_mixed_rotation(
    freqs: Tensor,
    pos: Tensor,
):
    """
    Project each 3-dim position into per-head, per-head-dim 1D frequencies.

    Args:
        freqs: [3, num_heads, num_freqs] - learned rotation frequency (for t, row, col) for each head position
        pos: [N, 3] - position of each token
        num_heads: int

    Returns:
        freqs_cos: [N, num_heads, num_freqs] - cosine components
        freqs_sin: [N, num_heads, num_freqs] - sine components
    """
    assert freqs.ndim == 3
    freqs_sum = pos.cast(freqs.dtype).unsqueeze(-1) * freqs.unsqueeze(0)
    freqs_cos = freqs_sum.cos()
    freqs_sin = freqs_sum.sin()
    return freqs_cos, freqs_sin
