"""SF: the k lowest eigenvalues of the normalized Laplacian, from `sf.py`.

`self.dimensions` and the `number_of_nodes <= self.dimensions` branch are
upstream's; the graph arrives as the CSR karateclub's integrity pass produces,
self-loops included. `nx.normalized_laplacian_matrix` is `I - D^-1/2 A D^-1/2`,
built here from `spectral.normalized_adjacency`.

The `eigsh` calls are ARPACK; this is a full Jacobi sweep followed by an
argmin/argmax selection, so the `ncv=10 * self.dimensions` restart subspace has
no counterpart. With `return_eigenvectors=False` ARPACK sorts by absolute
value, so the `which="LM"` values arrive ascending; the port reproduces that
order, to roughly sweep accuracy rather than to the last bit.
"""

import kclinalg as linalg
import spectral

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def calculate_sf(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    number_of_nodes: Int,
    dimensions: Int,
    laplacian: Ptr,
    vectors: Ptr,
    vt: Ptr,
    picked: IPtr,
    selected: Ptr,
    embedding: Ptr,
    vectors_out: Ptr,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """`SF._calculate_sf`; `embedding` gets `dimensions` values.

    `laplacian`, `vectors` and `vt` are `n x n` scratch, `picked` is
    `Int32[n]`, `selected` holds the `k` picked eigenvalues, `embedding` is
    `dimensions` long and `vectors_out` is `n x k` scratch for the selected
    eigenvectors, which SF discards but `eig_select` has to write somewhere it
    is not reading from. Returns False when the Jacobi sweep runs out of passes
    before the off-diagonal norm reaches `tol`, leaving `embedding` unwritten;
    the caller is expected to raise there, the way the Python layer already
    does for `kc_jacobi_eigh`.
    """
    var n = number_of_nodes
    spectral.normalized_laplacian(indptr, indices, values, n, laplacian)
    if linalg.jacobi_eigh(laplacian, vectors, vt, n, sweeps, tol) >= sweeps:
        return False
    var k = dimensions
    if n <= dimensions:
        k = n - 1
    # which="LM": the k largest of the n, which ARPACK hands back ascending.
    # `eig_select` walks argmax, so it produces them descending and the
    # embedding reverses.
    spectral.eig_select(
        laplacian, vectors, n, k, False, selected, vectors_out, picked
    )
    if n <= dimensions:
        # np.pad(embedding, (1, shape_diff), "constant", constant_values=0)
        var shape_diff = dimensions - k - 1
        linalg.fill(embedding, 0.0, 1)
        for i in range(k):
            embedding.unsafe_offset(i + 1)[] = selected.unsafe_load(k - 1 - i)
        linalg.fill(embedding.unsafe_offset(k + 1), 0.0, shape_diff)
    else:
        for i in range(k):
            embedding.unsafe_offset(i)[] = selected.unsafe_load(k - 1 - i)
    return True
