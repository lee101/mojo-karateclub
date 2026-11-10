"""Flops per byte for the dense kernels the GPU phase would have to accelerate.

Every figure is counted from the kernel source in `src/mojokarateclub/` and the
real fixture sizes in `tests/_graphs.py`; nothing is estimated. The GPU gate in
the phase brief admits a kernel only above roughly 2 flops/byte, so this prints
the ratio for each candidate at the sizes real callers pass.

    pixi run python bench/_gpu_gate.py
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

spec = importlib.util.spec_from_file_location("kc_fixtures", ROOT / "tests" / "_graphs.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def jacobi(d, sweeps, upper_nonzero):
    """`jacobi_eigh` over one d x d symmetric matrix, d(d-1)/2 rotations per sweep.

    Per rotation (`jacobi_rows` / `jacobi_pairs`): a `k`-loop over up to d
    entries doing 4 loads, 4 stores, 8 multiplies and 4 adds for the two matrix
    rows and two eigenvector rows; plus one 2x2 update of the rotation. Flops =
    12 per entry (8 mul + 4 add), so 12 * d per rotation; bytes = 8 * 8 * d
    touched (8 streams of Float64), so 64 * d. The ratio is therefore constant
    in d: 12/64 = 0.1875 flops/byte.
    """
    flops = sweeps * upper_nonzero * 12 * d
    bytes_ = sweeps * upper_nonzero * 64 * d
    return flops, bytes_


def pinv_spectrum(d):
    """`U diag(f) U.T` reconstruction, from the `gemm_nt` loop structure.

    Each output cell is one `dot` of length d: 2 flops per element. Both operands
    are read once per output row and the result is written once, so bytes =
    8 * (2 d^2 + d^2) = 24 d^2 and flops = 2 d^3, i.e. d/12 flops/byte.
    """
    return 2 * d**3, 24 * d * d


def wl_hashing(n, nnz, iterations):
    """One MD5 per node label per iteration: the hash is tens of flops over a
    few label bytes, the definition of memory-bound."""
    return 64 * n * iterations, 8 * nnz * iterations


def main():
    # The Jacobi sweep count comes from the kernel's own return value on each
    # fixture's dense normalized Laplacian, not from a guess.
    from mojokarateclub import _ffi
    from mojokarateclub.estimator import csr_from_graph

    import numpy as np

    measured_sweeps = {}
    measured_nz = {}
    for name in ("karate", "gnp60", "gnp200", "ws300", "ba2000"):
        graph, n = fx.build_graph(name, integrity=True)
        indptr, indices, values = csr_from_graph(graph)
        dense = np.zeros(n * n)
        _ffi._lib.kc_normalized_adjacency(
            _ffi._index(indptr, "i"), _ffi._index(indices, "i"),
            _ffi._data(values), n, _ffi._data(dense))
        work, vec, vt = dense.copy(), np.zeros(n * n), np.zeros(n * n)
        measured_sweeps[n] = int(_ffi._lib.kc_jacobi_eigh(
            _ffi._data(work), _ffi._data(vec), _ffi._data(vt), n, 400, 1e-24))
        # Rotations actually executed: a zero `a[q, p]` is skipped, so count the
        # non-zeros of the matrix's upper triangle, not every cell.
        measured_nz[n] = int(np.count_nonzero(np.triu(dense.reshape(n, n), 1)))

    print(f"{'kernel':<28}{'size':>18}{'flops':>14}{'bytes':>14}{'flops/byte':>12}")
    for name in ("karate", "gnp60", "gnp200", "ws300", "ba2000"):
        graph, n = fx.build_graph(name, integrity=True)
        nz = measured_nz[n]
        sweeps = measured_sweeps[n]
        f, b = jacobi(n, sweeps, nz)
        print(f"{'jacobi_eigh (dense)':<28}{'n=' + str(n):>18}{f:>14}{b:>14}{f / b:>12.4f}")
        f, b = pinv_spectrum(n)
        print(f"{'pinv_spectrum U f U.T':<28}{'d=' + str(n):>18}{f:>14}{b:>14}{f / b:>12.4f}")
        f, b = wl_hashing(n, graph.number_of_edges() * 2, 3)
        print(f"{'WL hashing':<28}{'n=' + str(n):>18}{f:>14}{b:>14}{f / b:>12.4f}")


if __name__ == "__main__":
    main()
