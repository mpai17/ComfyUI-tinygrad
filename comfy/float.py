from tinygrad import Tensor, dtypes

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS):
    mantissa_scaled = Tensor.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += Tensor.rand(*mantissa_scaled.shape, dtype=mantissa_scaled.dtype, device=mantissa_scaled.device)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

#Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype):
    if dtype == dtypes.fp8e4m3:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == dtypes.fp8e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = Tensor.sign(x)
    abs_x = x.abs()
    sign = Tensor.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = Tensor.clamp(
        Tensor.floor(Tensor.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS)

    sign *= Tensor.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    # tinygrad clamp doesn't have out parameter
    sign = sign.clamp(min=-float('inf'), max=float('inf'))
    return sign



def stochastic_rounding(value, dtype, seed=0):
    if dtype == dtypes.float32:
        return value.cast(dtypes.float32)
    if dtype == dtypes.float16:
        return value.cast(dtypes.float16)
    if dtype == dtypes.bfloat16:
        return value.cast(dtypes.bfloat16)
    if dtype == dtypes.fp8e4m3 or dtype == dtypes.fp8e5m2:
        Tensor.manual_seed(seed)
        output = Tensor.zeros_like(value).cast(dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size] = manual_stochastic_round_to_float8(value[i:i+slice_size], dtype)
        return output

    return value.cast(dtype)
