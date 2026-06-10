"""
Diagnostics for memory / latency / token measurement on M1 8GB.
"""

import psutil
import os
from loguru import logger


def memory_snapshot(label: str) -> dict:
    """
    Log RSS, VMS, and available system memory.

    Returns:
        dict with rss_mb, vms_mb, available_gb
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    sys_mem = psutil.virtual_memory()

    snapshot = {
        "label": label,
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "available_gb": sys_mem.available / (1024 ** 3),
    }

    logger.debug(
        f"[{label}] RSS: {snapshot['rss_mb']:.1f}MB | "
        f"VMS: {snapshot['vms_mb']:.1f}MB | "
        f"Available: {snapshot['available_gb']:.1f}GB"
    )

    return snapshot


def detect_swap_pressure() -> bool:
    """
    Returns True if system swap usage exceeds 100MB.

    If swap is detected during a benchmark run, latency measurements
    on M1 are invalidated.
    """
    swap = psutil.swap_memory()
    swap_mb = swap.used / (1024 * 1024)

    if swap_mb > 100:
        logger.critical(
            f"Swap pressure detected: {swap_mb:.1f}MB used. "
            f"Benchmark run aborted — swap invalidates latency measurements."
        )
        return True

    return False