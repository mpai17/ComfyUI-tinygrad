import logging
from typing import Optional

from tinygrad import Tensor, dtypes
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose, factorization


class OFTDiff:
    def __init__(self, weights):
        # Unpack weights tuple from adapter
        blocks, rescale, alpha, _ = weights

        # Store tensors
        self.oft_blocks = blocks
        if rescale is not None:
            self.rescale = rescale
            self.rescaled = True
        else:
            self.rescaled = False
        self.block_num, self.block_size, _ = blocks.shape
        self.constraint = float(alpha) if isinstance(alpha, (int, float)) else alpha.item()
        self.alpha = alpha if isinstance(alpha, (int, float)) else alpha.item()

    def __call__(self, w):
        org_dtype = w.dtype
        I = Tensor.eye(self.block_size)

        ## generate r
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        normed_q = q
        if self.constraint:
            q_norm = q.norm() + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).cast(dtypes.float32).inverse()

        ## Apply chunked matmul on weight
        _, *shape = w.shape
        org_weight = w.cast(r.dtype)
        org_weight = org_weight.unflatten(0, (self.block_num, self.block_size))
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = Tensor.einsum(
            "k n m, k n ... -> k m ...",
            r,
            org_weight,
        ).flatten(0, 1)
        if self.rescaled:
            weight = self.rescale * weight
        return weight.cast(org_dtype)

    def passive_memory_usage(self):
        """Calculates memory usage of the trainable parameters."""
        total_size = self.oft_blocks.numel() * self.oft_blocks.dtype.itemsize
        if self.rescaled:
            total_size += self.rescale.numel() * self.rescale.dtype.itemsize
        return total_size


class OFTAdapter(WeightAdapterBase):
    name = "oft"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        block_size, block_num = factorization(out_dim, rank)
        block = Tensor.zeros(block_num, block_size, block_size)
        return OFTDiff(
            (block, None, alpha, None)
        )

    def to_train(self):
        return OFTDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, Tensor],
        alpha: float,
        dora_scale: Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["OFTAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        blocks_name = "{}.oft_blocks".format(x)
        rescale_name = "{}.rescale".format(x)

        blocks = None
        if blocks_name in lora.keys():
            blocks = lora[blocks_name]
            if blocks.ndim == 3:
                loaded_keys.add(blocks_name)
            else:
                blocks = None
        if blocks is None:
            return None

        rescale = None
        if rescale_name in lora.keys():
            rescale = lora[rescale_name]
            loaded_keys.add(rescale_name)

        weights = (blocks, rescale, alpha, dora_scale)
        return cls(loaded_keys, weights)

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
        blocks = v[0]
        rescale = v[1]
        alpha = v[2]
        if alpha is None:
            alpha = 0
        dora_scale = v[3]

        blocks = comfy.model_management.cast_to_device(blocks, weight.device, intermediate_dtype)
        if rescale is not None:
            rescale = comfy.model_management.cast_to_device(rescale, weight.device, intermediate_dtype)

        block_num, block_size, *_ = blocks.shape

        try:
            # Get r
            I = Tensor.eye(block_size).cast(blocks.dtype)
            # for Q = -Q^T
            q = blocks - blocks.transpose(1, 2)
            normed_q = q
            if alpha > 0: # alpha in oft/boft is for constraint
                q_norm = q.norm() + 1e-8
                if q_norm > alpha:
                    normed_q = q * alpha / q_norm
            # use float() to prevent unsupported type in .inverse()
            r = (I + normed_q) @ (I - normed_q).cast(dtypes.float32).inverse()
            r = r.cast(weight.dtype)
            _, *shape = weight.shape
            lora_diff = Tensor.einsum(
                "k n m, k n ... -> k m ...",
                (r * strength) - strength * I,
                weight.reshape(block_num, block_size, *shape),
            ).reshape(-1, *shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function((strength * lora_diff).cast(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
