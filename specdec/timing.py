"""Timing helpers: GPU synchronisation and GPU clock and temperature readings."""

import subprocess
import time
from contextlib import contextmanager

import torch


def sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(device: torch.device | str, into: list[float]):
    """Append the wall-clock seconds of the block to `into`, synchronising the GPU on both sides."""
    sync(device)
    start = time.perf_counter()
    yield
    sync(device)
    into.append(time.perf_counter() - start)


def gpu_state(index: int = 0) -> dict | None:
    """SM clock (MHz), temperature (C) and power (W) of one GPU, or None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        clock, temperature, power = (float(x) for x in out.split(","))
        return {"sm_clock_mhz": clock, "temperature_c": temperature, "power_w": power}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
