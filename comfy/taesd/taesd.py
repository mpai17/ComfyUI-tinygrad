#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
from tinygrad import Tensor
import comfy.ops

# Local utils function to avoid importing full comfy.utils
def load_torch_file(path, safe_load=True):
    """Simplified load function - placeholder for now"""
    # TODO: Implement proper checkpoint loading when needed
    return {}

def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp:
    def forward(self, x):
        return (x / 3).tanh() * 3

class Block:
    def __init__(self, n_in, n_out):
        # Sequential layers stored as list
        self.conv1 = conv(n_in, n_out)
        self.conv2 = conv(n_out, n_out)  
        self.conv3 = conv(n_out, n_out)
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else None
    def forward(self, x):
        conv_out = self.conv3(self.conv2(self.conv1(x).relu()).relu())
        skip_out = self.skip(x) if self.skip is not None else x
        return (conv_out + skip_out).relu()

class Encoder:
    def __init__(self, latent_channels=4):
        # Layer 1
        self.conv1 = conv(3, 64)
        self.block1 = Block(64, 64)
        # Layer 2  
        self.conv2 = conv(64, 64, stride=2, bias=False)
        self.block2_1 = Block(64, 64)
        self.block2_2 = Block(64, 64)
        self.block2_3 = Block(64, 64)
        # Layer 3
        self.conv3 = conv(64, 64, stride=2, bias=False)
        self.block3_1 = Block(64, 64)
        self.block3_2 = Block(64, 64)
        self.block3_3 = Block(64, 64)
        # Layer 4
        self.conv4 = conv(64, 64, stride=2, bias=False)
        self.block4_1 = Block(64, 64)
        self.block4_2 = Block(64, 64)
        self.block4_3 = Block(64, 64)
        # Final layer
        self.conv_final = conv(64, latent_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.conv3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.conv4(x)
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.conv_final(x)
        return x


class Decoder:
    def __init__(self, latent_channels=4):
        self.clamp = Clamp()
        self.conv1 = conv(latent_channels, 64)
        # Stage 1
        self.block1_1 = Block(64, 64)
        self.block1_2 = Block(64, 64)
        self.block1_3 = Block(64, 64)
        self.conv1_up = conv(64, 64, bias=False)
        # Stage 2
        self.block2_1 = Block(64, 64)
        self.block2_2 = Block(64, 64)
        self.block2_3 = Block(64, 64)
        self.conv2_up = conv(64, 64, bias=False)
        # Stage 3
        self.block3_1 = Block(64, 64)
        self.block3_2 = Block(64, 64)
        self.block3_3 = Block(64, 64)
        self.conv3_up = conv(64, 64, bias=False)
        # Final stage
        self.block_final = Block(64, 64)
        self.conv_final = conv(64, 3)
    
    def forward(self, x):
        x = self.clamp(x)
        x = self.conv1(x).relu()
        # Stage 1
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = x.interpolate(scale_factor=2)  # Upsample replacement
        x = self.conv1_up(x)
        # Stage 2 
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = x.interpolate(scale_factor=2)
        x = self.conv2_up(x)
        # Stage 3
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = x.interpolate(scale_factor=2)
        x = self.conv3_up(x)
        # Final
        x = self.block_final(x)
        x = self.conv_final(x)
        return x

class TAESD:
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        self.taesd_encoder = Encoder(latent_channels)
        self.taesd_decoder = Decoder(latent_channels)
        self.vae_scale = Tensor([1.0])
        self.vae_shift = Tensor([0.0])
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(load_torch_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift
