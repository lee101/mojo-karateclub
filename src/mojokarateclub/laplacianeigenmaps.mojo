"""Laplacian Eigenmaps: the `k` smoothest eigenvectors of the normalized Laplacian.

An implementation of "Laplacian Eigenmaps" from the NIPS '01 paper
"Laplacian Eigenmaps and Spectral Techniques for Embedding and
Clustering". The procedure extracts the eigenvectors corresponding to the
largest eigenvalues of the graph Laplacian. These vectors are used as the node
embedding.

`scipy.sparse.linalg.eigsh` is ARPACK, and ARPACK is not here, so
`eigsh(L_tilde, k, which="SM")` becomes a dense symmetric Jacobi
diagonalisation of the same `L_tilde` followed by an ascending selection of the
`k` smallest eigenpairs. Three divergences follow from that, and none of them
is a rounding detail:

- `which="SM"` is ARPACK's *smallest magnitude* target, not a sort key. The
  dense path sorts, so the two agree on the eigenpairs, not on any notion of
  closeness to a target.
- Eigenvector signs are arbitrary. ARPACK pins them to a random start vector
  seeded by `_set_seed`, the Jacobi sweep pins them to its own rotation order,
  and the two do not agree. Compare embeddings column by column up to sign.
- ARPACK needs `O(nnz)` memory and a `maxiter` budget; the dense path needs
  `O(n^2)` and has no cap, so `maximum_number_of_iterations` is accepted and
  unused. `seed` is unused for the same reason.

Nothing allocates. Every buffer is the caller's, passed in explicitly.
"""

import kclinalg as linalg
import spectral

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def fit(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    n: Int,
    dimensions: Int,
    maximum_number_of_iterations: Int,
    seed: Int,
    embedding: Ptr,
    eigenvalues: Ptr,
    laplacian: Ptr,
    vectors: Ptr,
    vt: Ptr,
    picked: IPtr,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """`fit`: the `k` eigenvectors of the `k` smallest eigenvalues of `L_tilde`.

    `embedding` is `n x dimensions` row-major, `eigenvalues` is `dimensions`,
    and both come back in ascending eigenvalue order, as `eigsh` returns them.
    `laplacian`, `vectors` and `vt` are `n x n` row-major scratch for the
    dense `L_tilde`, the eigenvector block and that block's transpose;
    `picked` is `Int32[n]`.

    `tol` is `jacobi_eigh`'s budget on the summed squared off-diagonals, not a
    per-eigenvector accuracy, and the eigenvectors are the answer here rather
    than a by-product: at `1e-12` the near-null eigenvector of Zachary's club
    is only good to about `1e-7`, at `1e-24` to `1e-14`. Pass what the
    embedding needs; `sweeps` of the order of `100` is already far more than
    the sweep count these graphs use.

    Returns False when the Jacobi sweep misses `tol` inside `sweeps` passes,
    leaving `embedding` and `eigenvalues` undefined, and when `dimensions`
    is outside `[1, n]`. ARPACK refuses `k >= n` outright; the dense path
    computes it happily, so the caller still has to police that.
    """
    # self._set_seed() and self._check_graph() have no Mojo counterpart here:
    # the seed only ever reached ARPACK's start vector, and the indexing check
    # and the self-loops are the Python layer's, as in every other kernel.
    _ = maximum_number_of_iterations
    _ = seed
    if n <= 0 or dimensions < 1 or dimensions > n:
        return False
    spectral.normalized_laplacian(indptr, indices, values, n, laplacian)
    if linalg.jacobi_eigh(laplacian, vectors, vt, n, sweeps, tol) >= sweeps:
        return False
    spectral.eig_select(
        laplacian, vectors, n, dimensions, True, eigenvalues, embedding, picked
    )
    return True


def get_embedding(embedding: Ptr) -> Ptr:
    """`get_embedding`: upstream returns `self._embedding`, which on this side
    is the buffer `fit` wrote into, so there is nothing to copy."""
    return embedding
