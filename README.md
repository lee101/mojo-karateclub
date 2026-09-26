# mojo-karateclub

Mojo kernels for the numerical primitives that
[karateclub](https://github.com/benedekrozemberczki/karateclub) delegates to
numpy, scipy.sparse, networkx and LAPACK.

## What this is, and what it is not

**This is not a port of karateclub's embedders.** None of `DeepWalk`,
`Node2Vec`, `LaplacianEigenmaps`, `Graph2Vec`, `NetMF`, `GraphWave`,
`FeatherNode`, `GEMSEC` or the rest of the karateclub model zoo is
reimplemented here. Those models are built out of the primitives below, and
this repository provides those primitives in Mojo behind a ctypes API.

karateclub itself does almost no arithmetic. It calls into:

| upstream call | used for | this repo |
| --- | --- | --- |
| `numpy` elementwise ops, reductions | feature scaling, normalisation | `fill`, `copy`, `axpy`, `scale`, `dot`, `total`, `sum_squares`, `frobenius` |
| `numpy` / BLAS products | propagation, embedding construction | `gemm`, `gemm_nt`, `matvec` |
| `numpy.linalg.eigh` (LAPACK) | Laplacian and proximity decompositions | `eigh` (cyclic Jacobi) |
| `numpy.linalg.inv` (LAPACK) | matrix inversions | `invert` (Gauss-Jordan) |
| `scipy.sparse` CSR algebra | adjacency products, densification | `csr_matvec`, `csr_matmul`, `csr_dense`, `csr_scatter`, `csr_row_scale` |
| `scipy.sparse` COO algebra | the same, from coordinate form | `coo_matmul`, `coo_matmul_t`, `coo_scatter` |
| `networkx` graph metrics | clustering, eccentricity | `clustering`, `eccentricity` |
| `numpy` / `sklearn` utilities | sampling, binning, sparsity conversion | `linspace`, `histogram`, `l1_normalize_rows`, `dense_to_coo` |

**Not covered:** every karateclub estimator, the `fit` / `get_embedding`
estimator API, graph construction and preprocessing, random-walk sampling,
the Gensim and PyGSP dependencies, and anything GPU. There is no GPU path
here; the pinned Mojo toolchain does not ship the host API needed to write
one (see `MOJO_NOTES.md`, section 5). Everything is single-threaded CPU code:
`parallelize` does not exist in this toolchain either.

Two kernels diverge from their upstream counterpart in a way that is
observable, and both are documented in place:

- `eigh` uses cyclic Jacobi, not LAPACK's divide and conquer. Eigenvalues are
  returned ascending with matching eigenvector columns, and the eigenvector
  *signs* are arbitrary. Every consumer of a basis-invariant
  `U @ diag(f) @ U.T` is unaffected; anything that pins a sign must do so
  itself.
- `eccentricity` returns `-1.0` for a node that cannot reach the whole graph,
  where `networkx.eccentricity` raises. This is a deliberate choice so a
  disconnected graph yields data rather than an exception.

## Install

Requires Linux x86-64 and the pinned Mojo toolchain from `pixi.toml`.

```bash
pixi install
pixi run build
```

`build` compiles `src/mojokarateclub/capi.mojo` into
`dist/libmojokarateclub.so` with `mojo build --emit shared-lib`. The Python
package lives in `python/` and is put on `PYTHONPATH` by the pixi activation
environment, so `pixi run python ...` and `pixi run test` see it without an
install step. To use the library from elsewhere, either keep using `pixi run`
or point `PYTHONPATH` at `python/` yourself.

## Usage

```python
import numpy as np
import mojokarateclub as mk

adjacency = np.array(
    [[0.0, 1.0, 1.0, 0.0],
     [1.0, 0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0]]
)
indptr = np.array([0, 2, 5, 7, 8], dtype=np.int32)
indices = np.array([1, 2, 0, 2, 3, 0, 1, 2], dtype=np.int32)

print(mk.clustering(indptr, indices))    # [1.  0.5 1.  0. ]
print(mk.eccentricity(indptr, indices))  # [2. 1. 2. 2.]

laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
values, vectors = mk.eigh(laplacian)
print(np.round(values, 6))               # [-0.  1.  3.  4.]
```

That example is the one in this README, run verbatim; its output is shown
above.

Routines documented as in place (`fill`, `scale`, `axpy`, `copy`,
`l1_normalize_rows`, `csr_row_scale`) mutate the array you pass and return it.
Everything else returns a fresh array.

## Contract

These kernels write through caller-owned pointers and never allocate, so a
wrong buffer is memory corruption rather than an exception. The wrapper
therefore validates before every call:

- float buffers must be `float64`, index buffers `int32`; there is no silent
  narrowing. Anything else raises `TypeError`.
- every buffer must be C-contiguous; a strided view raises `ValueError`
  rather than being read with the wrong stride.
- in-place routines require a writeable array.
- a CSR triple must start at `indptr[0] == 0`, be non-decreasing, end at
  `indptr[-1] == len(indices)`, and carry `values` of the same length as
  `indices`. A COO triple must have `rows`, `cols` and `values` of equal
  length and non-negative indices.
- column and row indices are range-checked against the destination before the
  call, so a dangling index raises instead of writing out of bounds.
- `eigh` raises if Jacobi runs out of sweeps rather than returning an
  unconverged decomposition, and `invert` raises `numpy.linalg.LinAlgError` on
  an exactly singular pivot.

## How it works

`src/mojokarateclub/kclinalg.mojo` holds the kernels. They take
`Pointer[Float64, AnyOrigin[mut=True]]` and write into caller-owned buffers;
nothing in that file allocates, so buffer lifetime stays the caller's problem.

`src/mojokarateclub/capi.mojo` is the C ABI shim. Mojo pointers are
non-nullable and so cannot be parameters of an `abi("C")` function, so every
buffer crosses the boundary as a 64-bit `Int` address and is rebuilt inside
the wrapper with `Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=)`.
The address is the load-bearing detail: Mojo `Int` is 64 bits, so the ctypes
declarations use `c_int64`. A narrower `c_int` would truncate a heap address
into a wild pointer and kill the process with no diagnostic from either side.
A test asserts that no argument type in the signature table is narrower than
`c_int64`.

`python/mojokarateclub/__init__.py` holds the ctypes glue. Buffers are always
row-major, contiguous `float64` or `int32` numpy arrays, so a kernel index is
a flat offset. Address extraction, dtype and contiguity checks, and the
validation above all live on the Python side, which is the only place that can
raise.

The elementwise and reduction kernels are hand-vectorised with a compile-time
width of 4 `float64` lanes (`comptime W = 4`), running the SIMD body while
`i + W <= n` and a scalar tail for the remainder. The CSR and COO kernels
accumulate through the same `axpy` primitive, so the SIMD path is shared by
every sparse product.

## Tests

```bash
pixi run test
```

Every exported symbol is covered. Dense kernels are checked against numpy;
`eigh` and `invert` against LAPACK through `numpy.linalg`; the two graph
kernels against reference implementations written from the algorithm in the
test file, and separately against real `networkx` in the benchmark. The
validation rules in the section above each have a test that asserts the
rejection.

## Benchmarks

Measured on this machine, Linux x86-64, Intel Xeon E5-2697 v4 at 2.30 GHz,
CPython 3.13, numpy 2.5.1, scipy 1.18.1, networkx 3.7. Best of N with the
repeat count auto-calibrated to run each side for at least 50 ms; the machine
is shared, so best-of rather than mean, because the only samples a scheduler
preempts are the slow ones.

| kernel | mojo (ms) | upstream (ms) | speedup |
| --- | ---: | ---: | ---: |
| `fill` n=1048576 | 0.8127 | 0.7903 | 0.97x |
| `copy` n=1048576 | 1.7514 | 0.8458 | 0.48x |
| `axpy` n=1048576 | 2.2386 | 9.0369 | 4.04x |
| `scale` n=1048576 | 1.3693 | 0.8712 | 0.64x |
| `dot` n=4194304 | 10.1024 | 64.4664 | 6.38x |
| `total` n=4194304 | 4.7812 | 5.0608 | 1.06x |
| `frobenius` n=4194304 | 6.5842 | 89.7891 | 13.64x |
| `gemm` 512x512x512 | 107.7012 | 224.0043 | 2.08x |
| `gemm_nt` 512x512x512 | 495.2961 | 151.5116 | 0.31x |
| `matvec` 4096x4096 | 19.6266 | 25.9771 | 1.32x |
| `eigh` d=256 | 3175.0275 | 5031.5580 | 1.58x |
| `invert` d=160 | 12.0828 | 1713.7971 | 141.84x |
| `linspace` n=1048576 | 3.2723 | 2.7603 | 0.84x |
| `l1_normalize_rows` 8192x128 | 3.0638 | 6.1682 | 2.01x |
| `histogram` n=4194304 | 21.2345 | 95.7781 | 4.51x |
| `dense_to_coo` 256x256 | 0.2594 | 1.3374 | 5.16x |
| `clustering` n=400 | 0.7759 | 100.2256 | 129.18x |
| `eccentricity` n=400 | 16.0177 | 188.4877 | 11.77x |
| `csr_matvec` n=20000 | 2.0418 | 1.8299 | 0.90x |
| `csr_scatter` n=20000 | 2393.3274 | 2831.0433 | 1.18x |
| `csr_matmul` n=20000x64 | 84.1731 | 232.5347 | 2.76x |
| `csr_dense` n=4000 | 78.3058 | 238.9753 | 3.05x |
| `csr_row_scale` n=4000 | 0.2326 | 1.5346 | 6.60x |
| `coo_matmul` nnz=399748x64 | 66.9755 | 231.4704 | 3.46x |
| `coo_matmul_t` nnz=399748x64 | 58.7927 | 67.2285 | 1.14x |
| `coo_scatter` nnz=79880 | 62.2226 | 168.6535 | 2.71x |

Geometric mean across all 26 cases: **2.87x**. Reproduce with
`pixi run bench`, which holds a machine-wide lock.

The pattern is the expected one for hand-written Mojo against a mature
native stack, and it is worth stating plainly rather than hiding:

- The port wins where the upstream call is interpreted Python looping over
  networkx objects (`clustering` 129x, `eccentricity` 12x), where the Mojo
  code avoids an allocation or a temporary (`invert` 142x, `dense_to_coo`),
  and on the sparse products (`csr_matmul` 2.76x, `coo_matmul` 3.46x), where
  scipy's general machinery is more than a CSR triple needs.
- The port loses where the upstream side is multithreaded BLAS. `gemm_nt` at
  0.31x is the clearest case: the hand-written strided inner loop is single
  threaded, while `b @ b.T` goes to a threaded BLAS. `copy` (0.48x), `scale`
  (0.64x) and `csr_matvec` (0.90x) are memory-bandwidth-bound streaming
  loops where numpy's already-tuned vectorised code is at parity or better,
  and a serial Mojo loop cannot win there.
- The Mojo kernels are single-threaded because this toolchain has no
  `parallelize` (see `MOJO_NOTES.md`, section 4). That is the whole reason for
  the rows below 1.0x.

## License

MIT, Lee Penkman. See `LICENSE`.
