"""Time the dense eigensolver on the fixture graphs the slow rows use.

`bench/bench.py` times whole estimators, and four of the rows that lose to
upstream are the same dense Jacobi sweep reached through a different estimator.
This isolates that one kernel so an edit to it can be measured without paying
for the upstream half of the table. Run it under the machine lock:

    flock /tmp/mojo-bench.lock pixi run python bench/bench_eig.py
"""
import importlib.util
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "tests" / "_graphs.py"
sys.path.insert(0, str(ROOT / "python"))

from mojokarateclub import _ffi  # noqa: E402
from mojokarateclub.estimator import csr_from_graph  # noqa: E402

spec = importlib.util.spec_from_file_location("kc_fixtures", GRAPHS)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

_lib = _ffi._lib
REPEATS = 5


def adjacency(name):
    """`D^-1/2 A D^-1/2` densified: the same input the estimators diagonalise."""
    graph, n = fixtures.build_graph(name, integrity=True)
    indptr, indices, values = csr_from_graph(graph)
    out = np.zeros(n * n)
    _lib.kc_normalized_adjacency(
        _ffi._index(indptr, "i"),
        _ffi._index(indices, "i"),
        _ffi._data(values),
        n,
        _ffi._data(out),
    )
    return out.reshape(n, n), n


def main():
    print(f"{'graph':>8} {'d':>5} {'sweeps':>7} {'best':>10}")
    for name in ("karate", "ws300", "gnp200", "lollipop"):
        a, d = adjacency(name)
        best = None
        sweeps = None
        for _ in range(REPEATS):
            work = a.copy()
            vectors = np.zeros((d, d))
            vt = np.zeros((d, d))
            start = time.perf_counter()
            used = _lib.kc_jacobi_eigh(
                _ffi._data(work),
                _ffi._data(vectors),
                _ffi._data(vt),
                d,
                400,
                1e-24,
            )
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
            sweeps = used
        print(f"{name:>8} {d:>5} {sweeps:>7} {best * 1e3:>9.3f}ms")


if __name__ == "__main__":
    main()
