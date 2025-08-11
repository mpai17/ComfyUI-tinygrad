from tinygrad import Tensor
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clip(-30.0, 20.0)
        self.deterministic = deterministic
        self.std = (0.5 * self.logvar).exp()
        self.var = self.logvar.exp()
        if self.deterministic:
            self.var = self.std = Tensor.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * Tensor.randn(*self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return Tensor([0.])
        else:
            if other is None:
                return 0.5 * ((self.mean ** 2) + self.var - 1.0 - self.logvar).sum(axis=(1, 2, 3))
            else:
                return 0.5 * (((self.mean - other.mean) ** 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar).sum(axis=(1, 2, 3))

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * (logtwopi + self.logvar + ((sample - self.mean) ** 2) / self.var).sum(axis=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, Tensor) else Tensor([x])
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + (logvar1 - logvar2).exp()
        + ((mean1 - mean2) ** 2) * (-logvar2).exp()
    )
