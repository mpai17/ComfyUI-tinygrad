import logging
from typing import Optional

from tinygrad import Tensor, dtypes
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


def hada_weight_forward(w1u, w1d, w2u, w2d, scale=1.0):
    """Forward pass for Hada weight computation"""
    if isinstance(scale, (int, float)):
        scale = Tensor([scale])
    diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
    return diff_weight


def hada_weight_tucker_forward(t1, w1u, w1d, t2, w2u, w2d, scale=1.0):
    """Forward pass for Tucker decomposed Hada weight computation"""
    if isinstance(scale, (int, float)):
        scale = Tensor([scale])
    
    rebuild1 = Tensor.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
    rebuild2 = Tensor.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)
    
    return rebuild1 * rebuild2 * scale


class LohaDiff:
    def __init__(self, weights):
        # Unpack weights tuple from LoHaAdapter
        w1a, w1b, alpha, w2a, w2b, t1, t2, _ = weights

        # Store weights as tensors
        self.hada_w1_a = w1a
        self.hada_w1_b = w1b
        self.hada_w2_a = w2a
        self.hada_w2_b = w2b

        self.use_tucker = False
        if t1 is not None and t2 is not None:
            self.use_tucker = True
            self.hada_t1 = t1
            self.hada_t2 = t2
        else:
            self.hada_t1 = None
            self.hada_t2 = None

        # Store rank and alpha
        self.rank = w1b.shape[0]
        self.alpha = alpha if isinstance(alpha, (int, float)) else alpha.item()

    def __call__(self, w):
        org_dtype = w.dtype

        scale = self.alpha / self.rank
        if self.use_tucker:
            diff_weight = hada_weight_tucker_forward(self.hada_t1, self.hada_w1_a, self.hada_w1_b, 
                                                    self.hada_t2, self.hada_w2_a, self.hada_w2_b, scale)
        else:
            diff_weight = hada_weight_forward(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, scale)

        # Add the scaled difference to the original weight
        weight = w + diff_weight.reshape(w.shape)

        return weight.cast(org_dtype)

    def passive_memory_usage(self):
        """Calculates memory usage of the stored tensors."""
        total_size = 0
        for tensor in [self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b]:
            total_size += tensor.numel() * tensor.dtype.itemsize
        if self.use_tucker:
            for tensor in [self.hada_t1, self.hada_t2]:
                total_size += tensor.numel() * tensor.dtype.itemsize
        return total_size


class LoHaAdapter(WeightAdapterBase):
    name = "loha"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        in_dim = int(Tensor(weight.shape[1:]).prod().item())
        mat1 = Tensor.randn(out_dim, rank) * 0.1
        mat2 = Tensor.zeros(rank, in_dim)
        mat3 = Tensor.randn(out_dim, rank) * 0.1
        mat4 = Tensor.randn(rank, in_dim) * 0.01
        return LohaDiff(
            (mat1, mat2, alpha, mat3, mat4, None, None, None)
        )

    def to_train(self):
        return LohaDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, Tensor],
        alpha: float,
        dora_scale: Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoHaAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            weights = (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2, dora_scale)
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)
            return cls(loaded_keys, weights)
        else:
            return None

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=dtypes.float32,
        original_weight=None,
    ):
        v = self.weights
        w1a = v[0]
        w1b = v[1]
        if v[2] is not None:
            alpha = v[2] / w1b.shape[0]
        else:
            alpha = 1.0

        w2a = v[3]
        w2b = v[4]
        dora_scale = v[7]
        if v[5] is not None: #cp decomposition
            t1 = v[5]
            t2 = v[6]
            m1 = Tensor.einsum('i j k l, j r, i p -> p r k l',
                                comfy.model_management.cast_to_device(t1, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype))

            m2 = Tensor.einsum('i j k l, j r, i p -> p r k l',
                                comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype))
        else:
            m1 = comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype) @ \
                 comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype)
            m2 = comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype) @ \
                 comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype)

        try:
            lora_diff = (m1 * m2).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function(((strength * alpha) * lora_diff).cast(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
