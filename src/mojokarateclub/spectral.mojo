"""Graph spectral primitives shared by the spectral estimators.

Not upstream code: karateclub reaches for `nx.normalized_laplacian_matrix` and
`scipy.sparse.linalg.eigsh` for these. They are here so the estimators read the
way the upstream files do, and so the dense `n x n` work has a Mojo kernel at
all. The eigensolver is the symmetric Jacobi sweep from `kclinalg`, not ARPACK;
`README.md` names that divergence for every model that depends on it.
"""

from std.math import max, sqrt

import kclinalg as linalg

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def degrees(indptr: IPtr, indices: IPtr, values: Ptr, n: Int, dst: Ptr):
    """`nx.degrees` of a CSR: the weighted row sums, self-loops counted once.

    An unweighted graph arrives with a `values` array of ones, which makes this
    the plain degree networkx reports.
    """
    for r in range(n):
        var total = 0.0
        for k in range(Int(indptr.unsafe_load(r)), Int(indptr.unsafe_load(r + 1))):
            total += values.unsafe_load(k)
        dst.unsafe_offset(r)[] = total


def normalized_adjacency(
    indptr: IPtr, indices: IPtr, values: Ptr, n: Int, dst: Ptr
):
    """`D^-1/2 A D^-1/2` densified: the `A` behind `nx.normalized_laplacian_matrix`.

    An isolated node gets a zero row: its degree is zero, so networkx's
    `D^-1/2` is a zero on that row and column.
    """
    for r in range(n):
        for c in range(n):
            dst.unsafe_offset(r * n + c)[] = 0.0
    for r in range(n):
        var start = Int(indptr.unsafe_load(r))
        var end = Int(indptr.unsafe_load(r + 1))
        var dr = 0.0
        for k in range(start, end):
            dr += values.unsafe_load(k)
        if dr == 0.0:
            continue
        for k in range(start, end):
            var c = Int(indices.unsafe_load(k))
            var dc = 0.0
            for j in range(Int(indptr.unsafe_load(c)), Int(indptr.unsafe_load(c + 1))):
                dc += values.unsafe_load(j)
            if dc == 0.0:
                continue
            dst.unsafe_offset(r * n + c)[] = values.unsafe_load(k) / (
                sqrt(dr) * sqrt(dc)
            )


def normalized_laplacian(
    indptr: IPtr, indices: IPtr, values: Ptr, n: Int, dst: Ptr
):
    """`nx.normalized_laplacian_matrix`, densified: `D^-1/2 (D - A) D^-1/2`.

    This is `I - D^-1/2 A D^-1/2` everywhere the degree is positive, but not
    on an isolated node: networkx replaces the infinite `1 / sqrt(0)` with
    zero before the product, so a degree-zero node gets an all-zero row rather
    than the lone 1 on the diagonal that `I - A_norm` would leave. Getting
    that wrong moves the spectrum by one per isolated node, which is visible
    in NetLSD's heat kernel trace and in SF's eigenvalue list.
    """
    normalized_adjacency(indptr, indices, values, n, dst)
    for r in range(n):
        var total = 0.0
        for k in range(Int(indptr.unsafe_load(r)), Int(indptr.unsafe_load(r + 1))):
            total += values.unsafe_load(k)
        var diagonal = 0.0 if total == 0.0 else 1.0 - dst.unsafe_load(r * n + r)
        for c in range(n):
            var identity = 1.0 if r == c else 0.0
            dst.unsafe_offset(r * n + c)[] = (
                identity - dst.unsafe_load(r * n + c)
            )
        dst.unsafe_offset(r * n + r)[] = diagonal


def symmetric_pinv(
    a: Ptr,
    tmp: Ptr,
    vectors: Ptr,
    vt: Ptr,
    d: Int,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """Moore-Penrose pseudo-inverse of a symmetric `a[d, d]`, written into `a`.

    `numpy.linalg.pinv` cuts at `rcond * sigma_max`; the same relative cut is
    applied to the eigenvalues of the symmetric factorisation, which is the SVD
    for a symmetric argument. `tmp`, `vectors` and `vt` are `d x d` scratch.
    Returns False when the sweep does not converge within `sweeps` passes.
    """
    if d == 0:
        return True
    linalg.copy(a, tmp, d * d)
    if linalg.jacobi_eigh(tmp, vectors, vt, d, sweeps, tol) >= sweeps:
        return False
    var largest = 0.0
    for i in range(d):
        largest = max(largest, abs(tmp.unsafe_load(i * d + i)))
    var cut = rcond * largest
    linalg.pinv_spectrum(vectors, tmp, a, tmp, vt, d, cut)
    return True


def eig_select(
    values: Ptr,
    vectors: Ptr,
    d: Int,
    k: Int,
    ascending: Bool,
    out_values: Ptr,
    out_vectors: Ptr,
    picked: IPtr,
):
    """The `k` extreme eigenvalues of a diagonalised matrix, and their vectors.

    `values` is the diagonal `jacobi_eigh` leaves behind and `vectors` its
    column block. `picked` is `Int32[d]` scratch. Selection is a repeated
    argmin/argmax, which costs `O(k d)` and is not worth sorting for the
    `k << d` the callers ask for.

    `out_vectors` must not overlap `vectors`: selection interleaves reads at
    `vectors[r * d + best]` with writes at `out_vectors[r * k + i]`, and with
    `k < d` the write front runs ahead of the read front.
    """
    for j in range(d):
        picked.unsafe_offset(j)[] = Int32(0)
    for i in range(k):
        var best = -1
        var best_value = 0.0
        for j in range(d):
            if picked.unsafe_load(j) != 0:
                continue
            var v = values.unsafe_load(j * d + j)
            if best < 0:
                best = j
                best_value = v
            elif (ascending and v < best_value) or (
                not ascending and v > best_value
            ):
                best = j
                best_value = v
        picked.unsafe_offset(best)[] = Int32(1)
        out_values.unsafe_offset(i)[] = best_value
        for r in range(d):
            out_vectors.unsafe_offset(r * k + i)[] = vectors.unsafe_load(r * d + best)
