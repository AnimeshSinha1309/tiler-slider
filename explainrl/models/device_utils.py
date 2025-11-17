"""Device detection and management utilities."""

import torch
import platform


def get_device(prefer_gpu: bool = True) -> str:
    """
    Automatically detect the best available device.

    Detects in this order:
    1. CUDA (NVIDIA GPU) on Linux/Windows
    2. MPS (Metal Performance Shaders) on macOS
    3. CPU as fallback

    Args:
        prefer_gpu: If True, use GPU if available. If False, always use CPU.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if not prefer_gpu:
        return 'cpu'

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device

    # Check for MPS (Apple Silicon GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using Metal Performance Shaders (Apple Silicon GPU)")
        return device

    # Fallback to CPU
    print(f"Using CPU ({platform.processor() or 'Unknown processor'})")
    return 'cpu'


def move_to_device(obj, device: str):
    """
    Move a tensor, model, or optimizer to the specified device.

    Args:
        obj: Object to move (tensor, model, optimizer, etc.)
        device: Target device ('cuda', 'mps', or 'cpu')

    Returns:
        Object moved to device
    """
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.optim.Optimizer):
        # Move optimizer state to device
        for state in obj.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return obj
    else:
        return obj


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cpu': True,
        'cpu_count': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': [],
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'platform': platform.system(),
        'processor': platform.processor(),
    }

    if info['cuda_available']:
        for i in range(torch.cuda.device_count()):
            info['cuda_devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'total_memory': torch.cuda.get_device_properties(i).total_memory / 1e9,  # GB
            })

    return info


def print_device_info():
    """Print information about available devices."""
    info = get_device_info()

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"CPU Threads: {info['cpu_count']}")
    print()

    if info['cuda_available']:
        print("CUDA (NVIDIA GPU): Available ✓")
        for dev in info['cuda_devices']:
            print(f"  - {dev['name']}")
            print(f"    Compute Capability: {dev['capability']}")
            print(f"    Total Memory: {dev['total_memory']:.1f} GB")
    else:
        print("CUDA (NVIDIA GPU): Not available")

    print()

    if info['mps_available']:
        print("MPS (Apple Silicon GPU): Available ✓")
    else:
        print("MPS (Apple Silicon GPU): Not available")

    print("=" * 60)
