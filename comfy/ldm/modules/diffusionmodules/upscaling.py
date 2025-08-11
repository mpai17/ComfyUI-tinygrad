from tinygrad import Tensor
import numpy as np
from functools import partial

from .util import extract_into_tensor, make_beta_schedule


class AbstractLowScaleModel:
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_tensor = partial(Tensor, dtype=Tensor.default_type)

        self.betas = to_tensor(betas)
        self.alphas_cumprod = to_tensor(alphas_cumprod)
        self.alphas_cumprod_prev = to_tensor(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_tensor(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_tensor(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_tensor(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_tensor(np.sqrt(1. / alphas_cumprod - 1))

    def q_sample(self, x_start, t, noise=None, seed=None):
        if noise is None:
            if seed is None:
                noise = Tensor.randn(*x_start.shape, dtype=x_start.dtype)
            else:
                # Note: tinygrad seed handling different from torch
                noise = Tensor.randn(*x_start.shape, dtype=x_start.dtype)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x


class SimpleImageConcat(AbstractLowScaleModel):
    # no noise level conditioning
    def __init__(self):
        super(SimpleImageConcat, self).__init__(noise_schedule_config=None)
        self.max_noise_level = 0

    def forward(self, x):
        # fix to constant noise level
        return x, Tensor.zeros(x.shape[0]).cast(Tensor.int64)


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = Tensor.randint(0, self.max_noise_level, (x.shape[0],)).cast(Tensor.int64)
        else:
            assert isinstance(noise_level, Tensor)
        z = self.q_sample(x, noise_level, seed=seed)
        return z, noise_level



