"""FGSD: the histogram of the Moore-Penrose spectrum of the normalized Laplacian.

Port of `FGSD._calculate_fgsd` from `karateclub/graph_embedding/fgsd.py`. The
estimator is five lines of dense linear algebra with no walk state and no seed
in the arithmetic, so this file is the whole model: `fgsd_similarity` builds the
feature matrix `S`, `fgsd_embedding` histograms it.

Like the upstream, it is `O(n^2)` memory and `O(n^3)` time: the dense `n x n`
pseudo-inverse comes from a Jacobi sweep, not LAPACK's SVD.
"""

import kclinalg as linalg
import spectral

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def fgsd_similarity(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    n: Int,
    laplacian: Ptr,
    fL: Ptr,
    scratch_a: Ptr,
    scratch_b: Ptr,
    scratch_c: Ptr,
    picked: IPtr,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """`_calculate_fgsd` up to the histogram: the feature matrix `S` on return.

    `laplacian` is an `n x n` row-major buffer, and is where `S` lands; `fL` is
    the `n x n` pseudo-inverse, `scratch_a`, `scratch_b` and `scratch_c` the
    `n x n` work areas the pseudo-inverse needs, and `picked` the `Int32[n]`
    scratch the shared spectral signature carries. `rcond` is numpy's `pinv`
    cutoff (`1e-15` there), `sweeps` and `tol` the Jacobi limits. Returns False
    when the sweep does not converge.
    """
    # L = nx.normalized_laplacian_matrix(graph).todense()
    spectral.normalized_laplacian(indptr, indices, values, n, laplacian)

    # fL = np.linalg.pinv(L)
    linalg.copy(laplacian, fL, n * n)
    if not spectral.symmetric_pinv(
        fL, scratch_a, scratch_b, scratch_c, n, rcond, sweeps, tol
    ):
        return False

    # S = np.outer(np.diag(fL), ones) + np.outer(ones, np.diag(fL)) - 2 * fL
    # The outer products against `ones` are the two diagonal reads, so this is
    # the same sum with the products never materialized. `laplacian` is dead
    # once the pseudo-inverse is in `fL`, and `fL` must survive the loop: the
    # diagonal is read again for every column.
    for r in range(n):
        var dr = fL.unsafe_load(r * n + r)
        for c in range(n):
            laplacian.unsafe_offset(r * n + c)[] = (
                dr + fL.unsafe_load(c * n + c) - 2.0 * fL.unsafe_load(r * n + c)
            )
    return True


def fgsd_embedding(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    n: Int,
    laplacian: Ptr,
    fL: Ptr,
    scratch_a: Ptr,
    scratch_b: Ptr,
    scratch_c: Ptr,
    picked: IPtr,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
    hist_bins: Int,
    hist_lo: Float64,
    hist_hi: Float64,
    histogram_out: Ptr,
) -> Bool:
    """`_calculate_fgsd` whole: the `hist_bins` wide feature vector.

    `histogram_out` is `Float64[hist_bins]`, and the buffer roles are those of
    `fgsd_similarity`. `hist_lo` and `hist_hi` are upstream's `hist_range`,
    which the constructor pins to `(0, 20)`. Returns False when the
    pseudo-inverse does not converge, leaving `histogram_out` untouched.

    `capi.mojo` exports this fused form as `kc_fgsd_embedding`; the Python
    layer calls it directly.
    """
    if not fgsd_similarity(
        indptr,
        indices,
        values,
        n,
        laplacian,
        fL,
        scratch_a,
        scratch_b,
        scratch_c,
        picked,
        rcond,
        sweeps,
        tol,
    ):
        return False
    # np.histogram(S.flatten(), bins=self.hist_bins, range=self.hist_range)
    # `laplacian` holds S row-major, which is C order already, so `flatten()`
    # is a no-op and the count runs straight over the buffer.
    linalg.histogram(
        laplacian, histogram_out, n * n, hist_bins, hist_lo, hist_hi
    )
    return True
