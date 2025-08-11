import importlib
import logging

from tinygrad import Tensor, dtypes
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            logging.warning("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = Tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=tuple(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logging.info(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings:
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    # Tinygrad implementation using built-in AdamW optimizer
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        
        # Import tinygrad's AdamW optimizer
        from tinygrad.nn.optim import AdamW
        
        # Create the underlying AdamW optimizer
        self.optimizer = AdamW(params, lr=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
        
        # Store additional EMA parameters
        self.ema_decay = ema_decay
        self.ema_power = ema_power
        self.param_names = param_names
        self.amsgrad = amsgrad
        
        # Initialize EMA state for each parameter
        self.ema_state = {}
        for p in self.optimizer.params:
            self.ema_state[p] = {
                'step': 0,
                'param_exp_avg': p.detach().cast(dtypes.float)
            }

    def __setstate__(self, state):
        # Restore optimizer and EMA state
        if 'optimizer' in state:
            self.optimizer = state['optimizer']
        if 'ema_state' in state:
            self.ema_state = state['ema_state']

    def step(self, closure=None):
        """Performs a single optimization step using tinygrad's built-in AdamW optimizer.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Enable training mode for tinygrad optimizer
        training_state = Tensor.training
        Tensor.training = True
        
        try:
            # Use tinygrad's built-in AdamW optimizer for the main optimization step
            # This ensures GPU compatibility and optimal performance
            self.optimizer.step()
        finally:
            # Restore original training state
            Tensor.training = training_state

        # Update EMA parameters manually
        for param in self.optimizer.params:
            if param.grad is not None:
                ema_state = self.ema_state[param]
                ema_state['step'] += 1
                
                # Calculate current EMA decay based on step and power
                cur_ema_decay = min(self.ema_decay, 1 - ema_state['step'] ** -self.ema_power)
                
                # Update EMA parameter: ema = ema * decay + param * (1 - decay)
                ema_param = ema_state['param_exp_avg']
                new_ema_value = ema_param * cur_ema_decay + param.detach().cast(dtypes.float) * (1 - cur_ema_decay)
                ema_param.assign(new_ema_value)

        return loss

    def zero_grad(self):
        """Zero the gradients of all parameters."""
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        """Compatibility property for PyTorch-style optimizers."""
        return [{'params': self.optimizer.params, 'lr': self.optimizer.lr, 
                'betas': (self.optimizer.b1, self.optimizer.b2), 'eps': self.optimizer.eps,
                'weight_decay': self.optimizer.wd, 'amsgrad': self.amsgrad, 
                'ema_decay': self.ema_decay, 'ema_power': self.ema_power, 
                'param_names': self.param_names}]

    @property 
    def state(self):
        """Access to optimizer state (for compatibility)."""
        return {'optimizer_state': self.optimizer, 'ema_state': self.ema_state}
