"""Time `kc_symmetric_pinv`'s reconstruction-only case at the sizes callers pass.

The input is the identity rotated by an orthogonal basis, which Jacobi converges
on immediately, so this is the `U diag(1/s) U.T` reconstruction alone - the one
kernel in the port above 2 flops/byte. Serial path (workers=1), best of N.

    pixi run python bench/_gpu_pinv.py
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojokarateclub import _ffi  # noqa: E402

_lib = _ffi._lib


def pinv(a, workers=1, sweeps=400):
    d = a.shape[0]
    out = np.array(a, order="C", copy=True).reshape(-1)
    tmp = np.zeros(d * d)
    vectors = np.zeros(d * d)
    vt = np.zeros(d * d)
    assert _lib.kc_symmetric_pinv(
        _ffi._data(out), _ffi._data(tmp), _ffi._data(vectors), _ffi._data(vt),
        d, 1e-15, sweeps, 1e-24, workers,
    )
    return out


def main():
    rng = np.random.default_rng(20261110)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"{'d':>6}{'best ms':>12}{'flops/byte':>14}")
    for d in (34, 60, 200, 300, 512, 700, 768):
        basis, _ = np.linalg.qr(rng.normal(size=(d, d)))
        a = np.ascontiguousarray(basis @ np.eye(d) @ basis.T)
        pinv(a)  # warm
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            pinv(a)
            best = min(best, (time.perf_counter() - t0) * 1e3)
        # flops/byte for the reconstruction at this d (2 d^3 / 24 d^2 = d / 12)
        print(f"{d:>6}{best:>12.3f}{d / 12:>14.2f}")


if __name__ == "__main__":
    main()
