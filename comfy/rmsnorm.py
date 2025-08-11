from tinygrad import Tensor, dtypes
import comfy.model_management
import numbers
import logging

RMSNorm = None

def rms_norm(x, weight=None, eps=1e-6):
    r = x * (x*x).mean(axis=-1, keepdim=True).rsqrt()
    if weight is None:
        return r
    else:
        return r * comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)


class RMSNorm:
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Tensor.empty(self.normalized_shape, device=device, dtype=dtype, requires_grad=True)
        else:
            self.weight = None
        self.bias = None

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)
