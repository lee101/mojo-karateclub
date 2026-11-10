"""Whole-model `FGSD` on `ws300`, plus the `pinv_spectrum` share of it.

`_calculate_fgsd` allocates five `n x n` float64 buffers plus a small
histogram, and the device has to hold a copy of each one it touches. This prints
the resident size those buffers imply at every fixture size, the device bytes a
GPU `pinv_spectrum` would need, and - at `ws300` only, because the sweep is
seconds on this shared host - the reconstruction's share of the whole call.

    pixi run python bench/_gpu_footprint.py
"""
import importlib.util
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from mojokarateclub import _ffi, FGSD  # noqa: E402

spec = importlib.util.spec_from_file_location("kc_fixtures", ROOT / "tests" / "_graphs.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def main():
    print(f"{'n':>6}{'host FGSD buffers MiB':>24}{'device set MiB':>18}")
    for name in ("karate", "gnp60", "gnp200", "ws300", "ba2000"):
        _, n = fx.build_graph(name, integrity=True)
        host = 5 * n * n * 8 / 2**20
        # A GPU pinv_spectrum needs `vectors` in, `inv` in, `dst` out only; the
        # Jacobi sweep stays on the host, so these three are the whole device set.
        device = 3 * n * n * 8 / 2**20
        print(f"{n:>6}{host:>24.1f}{device:>18.1f}")

    graph, n = fx.build_graph("ws300", integrity=True)
    model = FGSD()
    t0 = time.perf_counter()
    model.fit([graph])
    whole = (time.perf_counter() - t0) * 1e3
    print(f"\nFGSD ws300 whole fit: {whole:.1f} ms")
    print(f"pinv_spectrum share at d=300: 5.5-11.8 ms (bench/_gpu_pinv.py), "
          f"i.e. {5.5 / whole:.1%} to {11.8 / whole:.1%} of the call; the rest is the Jacobi sweep")


if __name__ == "__main__":
    main()
