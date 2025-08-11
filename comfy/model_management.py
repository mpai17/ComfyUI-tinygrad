"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import psutil
import logging
from enum import Enum
from comfy.cli_args import args, PerformanceFeature
from tinygrad import Tensor, dtypes, Device
import sys
import platform
import weakref
import gc

class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

def get_supported_float8_types():
    """Return supported float8 types in tinygrad"""
    float8_types = []
    # tinygrad supports fp8e4m3 and fp8e5m2
    if hasattr(dtypes, 'fp8e4m3'):
        float8_types.append(dtypes.fp8e4m3)
    if hasattr(dtypes, 'fp8e5m2'):
        float8_types.append(dtypes.fp8e5m2)
    return float8_types

FLOAT8_TYPES = get_supported_float8_types()

# tinygrad device detection
try:
    available_devices = Device._enum()
    default_device = Device.DEFAULT
except:
    available_devices = ["CPU"]
    default_device = "CPU"

lowvram_available = True
if hasattr(args, 'deterministic') and args.deterministic:
    logging.info("Using deterministic algorithms for tinygrad")
    # tinygrad is deterministic by default, no special setting needed

directml_enabled = False
# DirectML not supported in tinygrad

total_vram_available_mb = -1
total_ram_available_mb = -1

def is_intel_xpu():
    """Check if Intel XPU is available"""
    return "INTEL" in str(available_devices).upper()

def get_torch_device():
    """Get the main device for tinygrad"""
    global directml_enabled
    
    if directml_enabled:
        return "CPU"  # DirectML not supported, fallback to CPU
    
    if hasattr(args, 'cpu') and args.cpu:
        return "CPU"
    
    # Try to get GPU device if available
    if "GPU" in available_devices:
        return "GPU"
    elif "CUDA" in available_devices:
        return "CUDA"
    elif "METAL" in available_devices:
        return "METAL"
    elif "OPENCL" in available_devices:
        return "OPENCL"
    else:
        return "CPU"

def get_total_memory(dev=None, tinygrad_dev=None):
    """Get total memory for device"""
    if dev is not None:
        dev = str(dev)
    
    global total_vram_available_mb
    global total_ram_available_mb
    
    if total_ram_available_mb < 0:
        try:
            total_ram_available_mb = psutil.virtual_memory().total // (1024 * 1024)
        except:
            total_ram_available_mb = 8192  # Default 8GB
    
    if dev == "CPU" or dev == "cpu":
        return total_ram_available_mb * 1024 * 1024
    
    # For GPU devices, try to get VRAM info
    if total_vram_available_mb < 0:
        try:
            # Try to get GPU memory info (implementation would be device-specific)
            total_vram_available_mb = 8192  # Default 8GB VRAM
        except:
            total_vram_available_mb = 8192
    
    return total_vram_available_mb * 1024 * 1024

def get_free_memory(dev=None, tinygrad_dev=None):
    """Get free memory for device"""
    if dev == "CPU" or dev == "cpu":
        try:
            return psutil.virtual_memory().available
        except:
            return get_total_memory(dev) // 2  # Estimate 50% free
    
    # For GPU, estimate based on total
    return get_total_memory(dev) // 2

def maximum_batch_area(device=None):
    """Calculate maximum batch area for device"""
    global vram_state
    if vram_state == VRAMState.NO_VRAM:
        return 0
    
    memory_free = get_free_memory(device) / (1024 * 1024)
    
    # Conservative estimation
    if memory_free > 14000:
        return 4096 * 4096
    elif memory_free > 8000:
        return 2048 * 2048
    elif memory_free > 4000:
        return 1024 * 1024
    else:
        return 512 * 512

def cpu_mode():
    """Check if in CPU mode"""
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    """Check if in MPS mode"""
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_type(device, dtype):
    """Check if device matches type"""
    if device is None:
        return False
    device_str = str(device).upper()
    dtype_str = str(dtype).upper()
    return dtype_str in device_str

def is_device_cpu(device):
    """Check if device is CPU"""
    return is_device_type(device, "CPU")

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if FP16 should be used"""
    if hasattr(args, 'force_fp32') and args.force_fp32:
        return False
    
    if hasattr(args, 'force_fp16') and args.force_fp16:
        return True
    
    if device is None:
        device = get_torch_device()
    
    if is_device_cpu(device):
        return False  # CPU typically doesn't benefit from FP16
    
    # For GPU devices, use FP16 if we have enough memory and it's beneficial
    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if BF16 should be used"""
    if hasattr(args, 'force_fp32') and args.force_fp32:
        return False
        
    if hasattr(args, 'force_fp16') and args.force_fp16:
        return False
    
    # BF16 support in tinygrad would depend on device capabilities
    return False

def supports_fp8_compute(device=None):
    """Check if device supports FP8 compute"""
    if device is None:
        device = get_torch_device()
    
    # Limited FP8 support in current tinygrad
    return len(FLOAT8_TYPES) > 0 and not is_device_cpu(device)

def cast_to_device(tensor, device, dtype, non_blocking=False, copy=False, stream=None):
    """Cast tensor to device and dtype"""
    if tensor is None:
        return tensor
    
    if not isinstance(tensor, Tensor):
        # Convert to Tensor if needed
        tensor = Tensor(tensor)
    
    # Move to device
    if device != tensor.device:
        tensor = tensor.to(device)
    
    # Cast dtype
    if dtype is not None and dtype != tensor.dtype:
        tensor = tensor.cast(dtype)
    
    return tensor

def cast_to(tensor, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    """Cast tensor to dtype and/or device"""
    if tensor is None:
        return tensor
    
    result = tensor
    
    # Cast dtype first
    if dtype is not None and dtype != tensor.dtype:
        result = result.cast(dtype)
    
    # Move to device
    if device is not None and device != tensor.device:
        result = result.to(device)
    
    return result

def device_supports_non_blocking(device):
    """Check if device supports non-blocking operations"""
    # tinygrad operations are generally non-blocking by default
    return True

def get_autocast_device(dev):
    """Get autocast device for mixed precision"""
    if dev is None:
        dev = get_torch_device()
    return dev

def supports_dtype(device, dtype):
    """Check if device supports dtype"""
    if device is None:
        return True
    
    device_str = str(device).upper()
    
    # CPU supports most dtypes
    if "CPU" in device_str:
        return dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.int32, dtypes.int64]
    
    # GPU devices have broader dtype support
    return True

def pick_weight_dtype(device, dtype=None, manual_cast=False):
    """Pick appropriate weight dtype for device"""
    if dtype is not None:
        return dtype
    
    if device is None:
        device = get_torch_device()
    
    if is_device_cpu(device):
        return dtypes.float32
    
    # For GPU, prefer FP16 for memory efficiency
    if should_use_fp16(device):
        return dtypes.float16
    
    return dtypes.float32

def pick_compute_dtype(device, dtype=None, manual_cast=False):
    """Pick appropriate compute dtype for device"""
    if dtype is not None:
        return dtype
    
    if device is None:
        device = get_torch_device()
    
    # Compute typically done in FP32 for precision
    return dtypes.float32

def get_offload_device():
    """Get device for offloading"""
    return "CPU"

def get_offload_stream(device):
    """Get offload stream for device"""
    # tinygrad doesn't use explicit streams like CUDA
    return None

def sync_stream(device, stream=None):
    """Sync stream for device"""
    # tinygrad handles synchronization automatically
    pass

def soft_empty_cache(force=False):
    """Soft empty cache"""
    if force:
        gc.collect()
    # tinygrad doesn't have explicit cache management like PyTorch

def cleanup_models(keep_clone_weights_loaded=False):
    """Cleanup models from memory"""
    soft_empty_cache(True)

def dtype_size(dtype):
    """Get size in bytes of dtype"""
    if dtype == dtypes.float32:
        return 4
    elif dtype == dtypes.float16:
        return 2
    elif dtype == dtypes.bfloat16:
        return 2
    elif dtype == dtypes.int32:
        return 4
    elif dtype == dtypes.int64:
        return 8
    elif dtype == dtypes.int8:
        return 1
    elif dtype in FLOAT8_TYPES:
        return 1
    else:
        return 4  # Default to 4 bytes

def module_size(module):
    """Calculate module size in bytes"""
    if hasattr(module, 'parameters'):
        total_size = 0
        for param in module.parameters():
            if hasattr(param, 'numel') and hasattr(param, 'dtype'):
                total_size += param.numel() * dtype_size(param.dtype)
        return total_size
    return 0

# Model loading and management
current_loaded_models = []

class ModelPatcher:
    """Simplified model patcher for tinygrad compatibility"""
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.current_device = current_device or offload_device
        self.weight_inplace_update = weight_inplace_update
        
    def clone(self):
        """Clone the model patcher"""
        return ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            self.weight_inplace_update
        )
    
    def is_clone(self, other):
        """Check if this is a clone of another patcher"""
        return (self.model is other.model and
                self.load_device == other.load_device and
                self.offload_device == other.offload_device)

def load_models_gpu(models, memory_required=0, force_patch_weights=False, force_full_load=False):
    """Load models to GPU"""
    global current_loaded_models
    
    for model in models:
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model = model.model.to(model.load_device)
        current_loaded_models.append(model)
    
    return True

def loaded_models(only_currently_used=False):
    """Get currently loaded models"""
    global current_loaded_models
    return current_loaded_models.copy()

def free_memory(memory_required, device, keep_loaded=[]):
    """Free memory on device"""
    global current_loaded_models
    
    models_to_unload = []
    for model in current_loaded_models:
        if model not in keep_loaded:
            models_to_unload.append(model)
    
    for model in models_to_unload:
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model = model.model.to(model.offload_device)
        current_loaded_models.remove(model)
    
    soft_empty_cache(True)

def unload_all_models(keep_clone_weights_loaded=False):
    """Unload all models"""
    global current_loaded_models
    current_loaded_models.clear()
    soft_empty_cache(True)

def resolve_lowvram_weight(weight, model, key):
    """Resolve low VRAM weight"""
    return weight

def accelerate_enabled(device, model_input_dtype, unet_dtype, manual_cast_dtype, device_supports_non_blocking):
    """Check if acceleration is enabled"""
    return False  # Simplified for tinygrad

# Initialize device and memory settings
get_torch_device_name = get_torch_device
torch_device = get_torch_device()

def get_torch_device_name(device=None):
    """Get device name"""
    if device is None:
        device = torch_device
    return str(device)

# Set up VRAM detection
try:
    total_vram = get_total_memory(torch_device) // (1024 * 1024)  # Convert to MB
    if total_vram < 4000:  # Less than 4GB
        set_vram_to = VRAMState.LOW_VRAM
    elif total_vram < 2000:  # Less than 2GB
        set_vram_to = VRAMState.NO_VRAM
    elif total_vram > 12000:  # More than 12GB
        set_vram_to = VRAMState.HIGH_VRAM
    else:
        set_vram_to = VRAMState.NORMAL_VRAM
except:
    set_vram_to = VRAMState.NORMAL_VRAM

# Apply user overrides
if hasattr(args, 'always_normal_vram') and args.always_normal_vram:
    set_vram_to = VRAMState.NORMAL_VRAM
elif hasattr(args, 'always_low_vram') and args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
elif hasattr(args, 'always_no_vram') and args.always_no_vram:
    set_vram_to = VRAMState.NO_VRAM
elif hasattr(args, 'always_high_vram') and (args.always_high_vram or getattr(args, 'always_gpu', False)):
    set_vram_to = VRAMState.HIGH_VRAM

vram_state = set_vram_to

# Set CPU state
if hasattr(args, 'always_cpu') and args.always_cpu:
    cpu_state = CPUState.CPU
elif "METAL" in str(torch_device).upper():
    cpu_state = CPUState.MPS
else:
    cpu_state = CPUState.GPU

logging.info(f"Set tinygrad device to: {torch_device}")
logging.info(f"VRAM State: {vram_state.name}")
logging.info(f"CPU State: {cpu_state.name}")

# Additional utility functions for compatibility
def attention_dtype():
    """Get attention computation dtype"""
    return dtypes.float32

def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    """Cast bias and weight - moved from ops.py for compatibility"""
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    if hasattr(s, 'bias') and s.bias is not None:
        bias = cast_to(s.bias, bias_dtype, device)

    weight = None
    if hasattr(s, 'weight') and s.weight is not None:
        weight = cast_to(s.weight, dtype, device)

    return weight, bias