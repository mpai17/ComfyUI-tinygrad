from tinygrad import Tensor


class PixelNorm:
    def __init__(self, dim=1, eps=1e-8):
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / (x**2).mean(axis=self.dim, keepdim=True).add(self.eps).sqrt()
    
    def __call__(self, x):
        return self.forward(x)
