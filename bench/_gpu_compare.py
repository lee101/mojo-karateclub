"""Interleaved GPU-vs-CPU comparison for the one GPU candidate, `pinv_spectrum`.

`bench/_gpu_pinv.py` (CPU serial) and `bench/_gpu_pinv_model.mojo` (device, full
per-call cost including the copies) are timed in alternating passes, so a load
spike on this shared host lands on both sides rather than on one. Reports the
best of the passes for each side, and the minimum across passes is reported for
both.

    pixi run python bench/_gpu_compare.py [passes]
"""
import re
import subprocess
import sys

PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DS = [34, 60, 200, 300]


def cpu():
    out = subprocess.run(
        ["pixi", "run", "python", "bench/_gpu_pinv.py", "5"], capture_output=True,
        text=True, check=True).stdout
    return {int(d): float(ms) for d, ms in re.findall(r"^\s*(\d+)\s+([\d.]+)\s", out, re.M)}


def gpu():
    out = subprocess.run(
        ["pixi", "run", "mojo", "run", "bench/_gpu_pinv_model.mojo"], capture_output=True,
        text=True, check=True).stdout
    return {int(d): float(ms) for d, ms in re.findall(r"d= (\d+)  per call ms= ([\d.e-]+)", out)}


cpu_best = {}
gpu_best = {}
for p in range(PASSES):
    for d, ms in cpu().items():
        cpu_best[d] = min(cpu_best.get(d, 1e9), ms)
    for d, ms in gpu().items():
        gpu_best[d] = min(gpu_best.get(d, 1e9), ms)
    print(f"pass {p + 1}/{PASSES} done", flush=True)

print(f"\n{'d':>6}{'cpu serial ms':>16}{'gpu per call ms':>18}{'cpu/gpu':>10}")
for d in DS if d in cpu_best and d in gpu_best:
    c, g = cpu_best[d], gpu_best[d]
    print(f"{d:>6}{c:>16.3f}{g:>18.3f}{c / g:>9.2f}x")
