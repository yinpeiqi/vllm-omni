# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cleanup of AsyncOmni model resources.
"""

import asyncio
import gc
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm.third_party.pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

# Simplified GPU memory check based on vllm-omni/tests/utils.py
def get_gpu_memory_usage(devices=None):
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    usage = {}

    try:
        nvmlInit()
        for device in devices:
            handle = nvmlDeviceGetHandleByIndex(device)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            usage[device] = mem_info.used # bytes
    except Exception as e:
            print(f"Error getting CUDA memory: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass

    return usage


def get_gpu_pids(devices=None):
    """Get set of PIDs running on GPU."""
    pids = set()
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    try:
        nvmlInit()
        for device in devices:
            handle = nvmlDeviceGetHandleByIndex(device)
            try:
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    pids.add(p.pid)
            except Exception:
                # Some devices might not support this or fail
                pass
    except Exception as e:
        print(f"Error getting CUDA processes: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return pids


def test_diffusion_model_passive_close_cleanup():
    """Test that diffusion model cleans up spawned resources when garbage collected."""
    
    model = "riverclouds/qwen_image_random"
    
    # Check initial GPU memory
    initial_gpu_usage = get_gpu_memory_usage()
    initial_gpu_pids = get_gpu_pids()

    omni = Omni(model=model)

    # Verify we can see processes on GPU
    running_gpu_pids = get_gpu_pids()
    new_gpu_pids = running_gpu_pids - initial_gpu_pids
    print(f"New GPU PIDs during execution: {new_gpu_pids}")
    
    del omni
    gc.collect()
    
    # Wait for cleanup
    time.sleep(10)

    # Check GPU processes after cleanup
    final_gpu_pids = get_gpu_pids()
    leaked_gpu_pids = final_gpu_pids - initial_gpu_pids

    # Check GPU memory after cleanup
    final_gpu_usage = get_gpu_memory_usage()
    
    if leaked_gpu_pids:
        print(f"Killing leaked GPU processes: {leaked_gpu_pids}")
        for pid in leaked_gpu_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    assert not leaked_gpu_pids, f"Leaked GPU processes: {leaked_gpu_pids}"

    for dev, usage in final_gpu_usage.items():
        initial = initial_gpu_usage.get(dev, 0)
        # detect if the memory is released correctly
        assert usage < initial + 500 * 1024**2


def test_async_diffusion_model_passive_close_cleanup():
    """Test that AsyncOmni diffusion model cleans up spawned resources when garbage collected."""
    
    model = "riverclouds/qwen_image_random"

    # We create a new loop for this test to avoid interfering with any existing loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Check initial GPU memory
    initial_gpu_usage = get_gpu_memory_usage()
    initial_gpu_pids = get_gpu_pids()

    omni = AsyncOmni(model=model)

    # Verify we can see processes on GPU
    running_gpu_pids = get_gpu_pids()
    new_gpu_pids = running_gpu_pids - initial_gpu_pids
    print(f"New GPU PIDs during execution: {new_gpu_pids}")
    
    del omni
    gc.collect()
    
    # Wait for cleanup
    time.sleep(10)

    # Check GPU processes after cleanup
    final_gpu_pids = get_gpu_pids()
    leaked_gpu_pids = final_gpu_pids - initial_gpu_pids

    # Check GPU memory after cleanup
    final_gpu_usage = get_gpu_memory_usage()
    
    if leaked_gpu_pids:
        print(f"Killing leaked GPU processes: {leaked_gpu_pids}")
        for pid in leaked_gpu_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    assert not leaked_gpu_pids, f"Leaked GPU processes: {leaked_gpu_pids}"

    for dev, usage in final_gpu_usage.items():
        initial = initial_gpu_usage.get(dev, 0)
        # detect if the memory is released correctly
        assert usage < initial + 500 * 1024**2

    loop.close()
