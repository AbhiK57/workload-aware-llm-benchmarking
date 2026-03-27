"""
Core utilities for LLM inference benchmarking.
Includes GPU detection, dtype selection, and common helper functions.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU capabilities and return a dictionary with device info.
    
    Returns:
        Dictionary containing:
        - name: GPU name (or "CPU" if no GPU)
        - cc: Compute capability as float
        - memory_gb: Total GPU memory in GB
        - is_t4: Boolean indicating if GPU is T4
        - is_ampere_or_newer: Boolean for Ampere+ architecture (CC >= 8.0)
    """
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "cc": 0.0,
            "memory_gb": 0.0,
            "is_t4": False,
            "is_ampere_or_newer": False,
        }
    
    name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = float(f"{cc_major}.{cc_minor}")
    
    # Get memory info
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / (1024**3)
    
    is_t4 = "T4" in name
    is_ampere_or_newer = cc_major >= 8
    
    return {
        "name": name,
        "cc": cc,
        "memory_gb": round(memory_gb, 2),
        "is_t4": is_t4,
        "is_ampere_or_newer": is_ampere_or_newer,
    }


def pick_dtype(gpu_info: Dict[str, Any]) -> str:
    """
    Select optimal dtype based on GPU capabilities.
    
    Args:
        gpu_info: Dictionary from detect_gpu()
        
    Returns:
        "bfloat16" for Ampere+, "float16" for older GPUs
    """
    return "bfloat16" if gpu_info["is_ampere_or_newer"] else "float16"


def percentiles_ms(arr_s: List[float], ps: tuple = (50, 90, 95, 99)) -> Dict[str, float]:
    """
    Calculate percentiles from an array of values in seconds, returning ms.
    
    Args:
        arr_s: List of values in seconds
        ps: Tuple of percentiles to calculate
        
    Returns:
        Dictionary with keys like "p50_ms", "p90_ms", etc.
    """
    if not arr_s:
        return {f"p{p}_ms": float("nan") for p in ps}
    a = np.array(arr_s) * 1000.0
    return {f"p{p}_ms": float(np.percentile(a, p)) for p in ps}


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


def format_tokens(count: int) -> str:
    """Format token count with K/M suffix."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count/1000:.1f}K"
    else:
        return f"{count/1_000_000:.2f}M"


@dataclass
class BenchmarkConfig:
    """Global configuration for benchmark runs."""
    
    # Target model
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_mem_util: float = 0.90
    max_model_len: int = 4096
    
    # Benchmark settings
    warmup_requests: int = 3
    cooldown_seconds: float = 2.0
    
    # Default batch sizes
    default_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    
    # Timeout settings
    server_startup_timeout: int = 600
    request_timeout: int = 600
    
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"
