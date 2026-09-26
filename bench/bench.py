"""Benchmark the Mojo kernels against the libraries upstream karateclub calls.

karateclub itself has no arithmetic: it delegates to numpy for dense work,
scipy.sparse for the adjacency algebra, networkx for the graph metrics, and
LAPACK (through numpy) for the eigendecompositions. Those calls are therefore
the parity bar for this port, and every row below pairs one Mojo kernel with
the upstream call a caller would otherwise have made.

Elementwise rows compare kernel against kernel: the numpy side writes through
`out=` or a plain assignment so neither side is charged for an allocation the
other does not make. Products and factorisations allocate on both sides,
because there the output buffer is part of the work.

Timing: best of `repeats`, with the repeat count auto-calibrated so each side
runs for at least `TARGET` seconds. Best-of rather than mean, because the
machine is shared and the only samples a scheduler preempts are the slow ones;
a preempted sample should not be charged to the kernel. Rows marked
single-shot are baselines too slow to repeat at all.

Run it through `pixi run bench`, which holds a machine-wide flock.
"""

import platform
import statistics
import time

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp

import mojokarateclub as mk

TARGET = 0.05
MAX_REPEATS = 200
WARMUP = 2

# Each entry: (group, label, mojo call, upstream call, repeat both).
# The callables take no arguments and close over their inputs through default
# arguments, so building the next case cannot rebind an earlier one.
CASES = []


def timed(fn, repeat):
    """Best-of-N wall time in seconds; `repeat` False means a single call."""
    for _ in range(WARMUP if repeat else 1):
        fn()
    if not repeat:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start
    count = 1
    while True:
        start = time.perf_counter()
        for _ in range(count):
            fn()
        elapsed = (time.perf_counter() - start) / count
        if elapsed * count >= TARGET or count >= MAX_REPEATS:
            return elapsed
        count = min(count * 4, MAX_REPEATS)


def rand(n, rng):
    return np.ascontiguousarray(rng.normal(size=n))


def dense_csr(a):
    """Row-major CSR `(indptr, indices, values)` of a dense square block."""
    rows, cols = np.nonzero(a)
    indptr = np.zeros(a.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=a.shape[0]), out=indptr[1:])
    return indptr, cols.astype(np.int32), np.ascontiguousarray(a[rows, cols])


def to_coo(a):
    rows, cols = np.nonzero(a)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        np.ascontiguousarray(a[rows, cols]),
    )


def random_graph(n, avg_degree, rng):
    """Symmetric zero-diagonal 0/1 dense block; the shape karateclub embeds."""
    a = (rng.random((n, n)) < avg_degree / n).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    return np.maximum(a, a.T)


def sparse(n, avg_degree, rng):
    """`(dense, scipy csr, mojo csr triple)` for one undirected graph."""
    a = random_graph(n, avg_degree, rng)
    indptr, indices, values = dense_csr(a)
    return a, sp.csr_matrix((values, indices, indptr), shape=(n, n)), (
        indptr,
        indices,
        values,
    )


def add(group, label, mojo, upstream, repeat=True):
    CASES.append((group, label, mojo, upstream, repeat))


def build_cases():
    rng = np.random.default_rng(20260926)

    # ---------------------------------------------------------- elementwise
    n = 1 << 20
    x, y = rand(n, rng), rand(n, rng)
    add("elementwise", f"fill  n={n}", lambda y=y: mk.fill(y, 0.5),
        lambda y=y: y.fill(0.5))
    add("elementwise", f"copy  n={n}", lambda x=x, y=y: mk.copy(x, y),
        lambda x=x, y=y: np.copyto(y, x))
    add("elementwise", f"axpy  n={n}", lambda x=x, y=y: mk.axpy(0.5, x, y),
        lambda x=x, y=y: np.add(y, 0.5 * x, out=y))
    add("elementwise", f"scale n={n}", lambda y=y: mk.scale(y, 0.5),
        lambda y=y: np.multiply(y, 0.5, out=y))

    # ---------------------------------------------------------- reductions
    n = 1 << 22
    x, y = rand(n, rng), rand(n, rng)
    add("reduction", f"dot  n={n}", lambda x=x, y=y: mk.dot(x, y),
        lambda x=x, y=y: float(np.dot(x, y)))
    add("reduction", f"sum  n={n}", lambda x=x: mk.total(x),
        lambda x=x: float(x.sum()))
    add("reduction", f"norm n={n}", lambda x=x: mk.frobenius(x),
        lambda x=x: float(np.linalg.norm(x)))

    # ---------------------------------------------------------- products
    m = k = p = 512
    a = rand(m * k, rng).reshape(m, k)
    b = rand(k * p, rng).reshape(k, p)
    add("product", f"gemm   {m}x{k}x{p}", lambda a=a, b=b: mk.gemm(a, b),
        lambda a=a, b=b: a @ b)
    add("product", f"gemm.T {m}x{k}x{p}", lambda a=a, b=b: mk.gemm_nt(a, b),
        lambda a=a, b=b: a @ b.T)
    rows = cols = 4096
    a = rand(rows * cols, rng).reshape(rows, cols)
    v = rand(cols, rng)
    add("product", f"matvec {rows}x{cols}", lambda a=a, v=v: mk.matvec(a, v),
        lambda a=a, v=v: a @ v)

    # ------------------------------------------------------ factorisations
    d = 256
    s = rand(d * d, rng).reshape(d, d)
    sym = np.ascontiguousarray((s + s.T) / 2)
    add("factor", f"eigh  d={d}", lambda a=sym: mk.eigh(a, sweeps=60),
        lambda a=sym: np.linalg.eigh(a), repeat=False)
    d = 160
    s = np.ascontiguousarray(rand(d * d, rng).reshape(d, d) * 0.1 + d * np.eye(d))
    add("factor", f"inv   d={d}", lambda a=s: mk.invert(a),
        lambda a=s: np.linalg.inv(a), repeat=False)

    # ---------------------------------------------------------- utilities
    n = 1 << 20
    add("util", f"linspace n={n}", lambda n=n: mk.linspace(0.0, 1.0, n),
        lambda n=n: np.linspace(0.0, 1.0, n))
    rows, cols = 8192, 128
    block = rand(rows * cols, rng).reshape(rows, cols)
    block[::7] = 0.0
    add("util", f"l1 rows {rows}x{cols}", lambda b=block: mk.l1_normalize_rows(b),
        lambda b=block: _l1_numpy(b))
    n = 1 << 22
    samples = rand(n, rng)
    add("util", f"hist  n={n}", lambda s=samples: mk.histogram(s, 256, -4.0, 4.0),
        lambda s=samples: np.histogram(s, bins=256, range=(-4.0, 4.0))[0])
    d = 256
    pattern = (rng.random((d, d)) < 0.05) * rand(d * d, rng).reshape(d, d)
    add("util", f"dense_to_coo {d}x{d}", lambda a=pattern: mk.dense_to_coo(a),
        lambda a=pattern: to_coo(a))

    # -------------------------------------------------------- graph metrics
    n = 400
    _, _, (indptr, indices, _) = sparse(n, 12, rng)
    graph = nx.from_scipy_sparse_array(
        sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=(n, n))
    )
    add("graph", f"clustering n={n}",
        lambda p=indptr, i=indices: mk.clustering(p, i),
        lambda g=graph: _nx_clustering(g), repeat=False)
    add("graph", f"eccentricity n={n}",
        lambda p=indptr, i=indices: mk.eccentricity(p, i),
        lambda g=graph: nx.eccentricity(g), repeat=False)

    # ------------------------------------------------------------------ CSR
    n = 20000
    _, mat, (indptr, indices, values) = sparse(n, 10, rng)
    v = rand(n, rng)
    add("csr", f"csr_matvec n={n}",
        lambda p=indptr, i=indices, w=values, v=v: mk.csr_matvec(p, i, w, v),
        lambda m=mat, v=v: m @ v)
    add("csr", f"csr_scatter n={n}",
        lambda p=indptr, i=indices, w=values: mk.csr_scatter(p, i, w),
        lambda m=mat: m.toarray())
    width = 64
    block = rand(n * width, rng).reshape(n, width)
    add("csr", f"csr_matmul n={n}x{width}",
        lambda p=indptr, i=indices, w=values, b=block: mk.csr_matmul(p, i, w, b),
        lambda m=mat, b=block: m @ b)
    n = 4000
    _, mat, (indptr, indices, values) = sparse(n, 10, rng)
    add("csr", f"csr_dense n={n}",
        lambda p=indptr, i=indices, w=values: mk.csr_dense(p, i, w),
        lambda m=mat: (m @ m).toarray(), repeat=False)
    scale = rand(n, rng)
    add("csr", f"csr_row_scale n={n}",
        lambda w=values, s=scale, p=indptr: mk.csr_row_scale(w, s, p),
        lambda m=mat, s=scale: sp.diags(s) @ m, repeat=False)

    # ------------------------------------------------------------------ COO
    n = 20000
    rows_a, cols_a, values_a = to_coo(random_graph(n, 10, rng))
    width = 64
    block = rand(n * width, rng).reshape(n, width)
    coo = sp.coo_matrix((values_a, (rows_a, cols_a)), shape=(n, n))
    add("coo", f"coo_matmul nnz={values_a.size}x{width}",
        lambda r=rows_a, c=cols_a, w=values_a, b=block: mk.coo_matmul(r, c, w, b),
        lambda a=coo, b=block: a @ b)
    add("coo", f"coo_matmul_T nnz={values_a.size}x{width}",
        lambda r=rows_a, c=cols_a, w=values_a, b=block: mk.coo_matmul_t(r, c, w, b),
        lambda a=coo, b=block: a.T @ b)
    n = 4000
    rows_a, cols_a, values_a = to_coo(random_graph(n, 10, rng))
    coo = sp.coo_matrix((values_a, (rows_a, cols_a)), shape=(n, n))
    add("coo", f"coo_scatter nnz={values_a.size}",
        lambda r=rows_a, c=cols_a, w=values_a: mk.coo_scatter(r, c, w),
        lambda a=coo: a.toarray())


def _l1_numpy(block):
    totals = np.abs(block).sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return block / totals


def _nx_clustering(g):
    return np.array([nx.clustering(g, v) for v in g], dtype=np.float64)


def main():
    build_cases()
    print(f"{platform.python_implementation()} {platform.machine()}   "
          f"numpy {np.__version__}  scipy {scipy.__version__}  "
          f"networkx {nx.__version__}")
    print(f"cases: {len(CASES)}   (best-of, target {TARGET}s per side)\n")
    header = f"{'kernel':<30}{'mojo (ms)':>12}{'upstream (ms)':>16}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for group, label, mojo, upstream, repeat in CASES:
        mojo_s, up_s = timed(mojo, repeat), timed(upstream, repeat)
        if rows and rows[-1][0] != group:
            print()
        rows.append((group, label, mojo_s, up_s))
        print(f"{label:<30}{mojo_s * 1e3:>12.4f}{up_s * 1e3:>16.4f}"
              f"{up_s / mojo_s:>9.2f}x", flush=True)
    print()
    print("geomean speedup vs upstream: "
          f"{statistics.geometric_mean([u / m for _, _, m, u in rows]):.2f}x")
    print("worst rows (speedup, kernel):")
    for ratio, label in sorted((u / m, lab) for _, lab, m, u in rows)[:8]:
        print(f"  {ratio:>7.2f}x  {label}")


if __name__ == "__main__":
    main()
