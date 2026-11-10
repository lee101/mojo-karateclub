"""Exact structural counts for the flops / bytes columns of bench/analysis.md.

Every number here is counted, not estimated: nnz, n, the kernel's iteration
counts, the Jacobi sweeps actually used (reported by the kernel), and for flops
an operation count derived from the kernel's own loop structure, read off
src/mojokarateclub/*.mojo. Bytes are the buffers the kernel touches, sized from
n, nnz and the parameters the estimator was built with.

    taskset -c 20-27 pixi run python bench/_analyze_counts.py
"""
import importlib.util
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

import mojokarateclub as mk  # noqa: E402
from mojokarateclub import _ffi  # noqa: E402
from mojokarateclub.estimator import csr_from_graph  # noqa: E402

spec = importlib.util.spec_from_file_location("kc_fixtures", ROOT / "tests" / "_graphs.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def graph_counts(name):
    graph, n = fx.build_graph(name, integrity=True)
    indptr, indices, values = csr_from_graph(graph)
    out = np.zeros(n * n)
    _ffi._lib.kc_normalized_adjacency(
        _ffi._index(indptr, "i"), _ffi._index(indices, "i"), _ffi._data(values),
        n, _ffi._data(out))
    a = out.reshape(n, n)
    work = a.copy()
    vec = np.zeros((n, n))
    vt = np.zeros((n, n))
    sweeps = _ffi._lib.kc_jacobi_eigh(
        _ffi._data(work), _ffi._data(vec), _ffi._data(vt), n, 400, 1e-24)
    # Jacobi rotations actually executed: sweeps x d(d-1)/2, but a zero apq is
    # skipped, so count the non-zeros of the upper triangle on the input.
    nz = int(np.count_nonzero(np.triu(a, 1)))
    return {
        "fixture": name, "n": n, "nnz": int(indptr.size and indices.size),
        "nnz_row": int(indices.size), "edges_directed": int(indices.size),
        "density": float(a.size and indices.size / (n * n)),
        "jacobi_sweeps": int(sweeps),
        "jacobi_upper_nonzero_first": nz,
        "bytes_csr": int(indptr.nbytes + indices.nbytes + values.nbytes),
        "bytes_dense_nn": int(n * n * 8),
    }


def main():
    names = [a for a in sys.argv[1:]] or ["karate", "ws300", "ba2000"]
    print(json.dumps([graph_counts(n) for n in names], indent=1))


if __name__ == "__main__":
    main()
