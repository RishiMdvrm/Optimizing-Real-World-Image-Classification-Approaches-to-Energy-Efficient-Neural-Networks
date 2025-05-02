from __future__ import annotations
import time
from contextlib import suppress
from typing import Tuple

import torch
from torch import nn


# ----------------------------------------------------------------------
# 1. Accuracy
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_model_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    model.eval().to(device)
    correct = total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ----------------------------------------------------------------------
# 2. Latency / throughput / power / energy / EDP
# ----------------------------------------------------------------------
def _nvml_handle():
    with suppress(ImportError, Exception):
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    return None


@torch.no_grad()
def measure_inference_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    warmup: int = 10,
) -> Tuple[float, float, float | None, float | None, float | None]:
    """
    Returns:
        latency (s/img), throughput (img/s),
        avg_power (W | None), energy_per_img (J | None), edp (J·s | None)
    """
    model.eval().to(device)

    # quick warm‑up for GPU kernels / caches
    it = iter(data_loader)
    for _ in range(min(warmup, len(data_loader))):
        x, _ = next(it)
        _ = model(x.to(device))

    handle = _nvml_handle()
    power_samples = []

    start = time.perf_counter()
    total_imgs = 0

    for x, _ in data_loader:
        if handle:
            import pynvml

            power_samples.append(
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            )  # mW → W
        _ = model(x.to(device))
        total_imgs += x.size(0)

    elapsed = time.perf_counter() - start
    latency = elapsed / total_imgs
    throughput = total_imgs / elapsed

    if handle and power_samples:
        avg_pwr = sum(power_samples) / len(power_samples)
        energy = avg_pwr * elapsed / total_imgs
        edp = energy * latency
    else:
        avg_pwr = energy = edp = None

    return latency, throughput, avg_pwr, energy, edp


# ----------------------------------------------------------------------
# 3. Parameter count & FLOPs
# ----------------------------------------------------------------------
def measure_model_size_and_flops(
    model: nn.Module, input_res: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Tuple[float | None, int]:
    """
    Returns (giga‑FLOPs | None, parameter_count)
    """
    params = sum(p.numel() for p in model.parameters())
    with suppress(ImportError):
        from ptflops import get_model_complexity_info

        flops, _ = get_model_complexity_info(
            model,
            input_res[1:],  # drop batch dim
            as_strings=False,
            print_per_layer_stat=False,
        )
        return flops / 1e9, params
    return None, params