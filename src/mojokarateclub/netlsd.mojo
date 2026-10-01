"""NetLSD: the heat kernel trace of the normalized Laplacian, from `netlsd.py`.

The four upstream methods in order, with the two places they lean on networkx
or scipy named in place: `nx.normalized_laplacian_matrix` is `I -` the matrix
`spectral.normalized_adjacency` builds, and `sps.linalg.eigsh` is a full Jacobi
sweep plus the `which` selection `spectral.eig_select` does. The
`2 * self.approximations < number_of_nodes` switch, the `[::-1]` on the `which="SM"`
result and the linear interpolation between the two blocks are upstream's.

`eigsh` with `return_eigenvectors=False` sorts by absolute value, `'LM'`
ascending and `'SM'` descending, and the port depends on that twice: the
`[::-1]` on the lower block leaves it ascending, and a `which="LM"` block
goes into the output as it comes back.

Upstream casts the Laplacian to float32 before handing it to ARPACK; the CSR
arrives as Float64 and the decomposition stays there, so the port's trace sits
closer to the exact one than the reference does.
"""

import kclinalg as linalg
import mathfn
import spectral

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def calculate_heat_kernel_trace(
    eigenvalues: Ptr,
    number_of_eigenvalues: Int,
    scale_min: Float64,
    scale_max: Float64,
    scale_steps: Int,
    timescales: Ptr,
    heat_kernel_trace: Ptr,
) -> None:
    """`NetLSD._calculate_heat_kernel_trace`; `heat_kernel_trace` gets the trace.

    `number_of_eigenvalues` is upstream's `nodes`: `eigenvalues.shape[0]`, which
    is `n - 2` on the branch that asks for `number_of_nodes - 2` eigenvalues and
    `n` on the approximated one, so the divisor differs with the branch.
    `timescales` is `scale_steps` scratch for the `np.logspace`.
    """
    linalg.linspace(timescales, scale_min, scale_max, scale_steps)
    # np.logspace is 10 ** linspace; `std.math` transcendentals are only good to
    # 1e-10 here, which the t up to 100 would amplify straight into the trace.
    var ln10 = mathfn.natural_log(10.0)
    for idx in range(scale_steps):
        var t = mathfn.natural_exp(timescales.unsafe_load(idx) * ln10)
        var total = 0.0
        for i in range(number_of_eigenvalues):
            total += mathfn.natural_exp(-t * eigenvalues.unsafe_load(i))
        heat_kernel_trace.unsafe_offset(idx)[] = total / Float64(number_of_eigenvalues)


def updown_linear_approx(
    eigenvalues_lower: Ptr,
    eigenvalues_upper: Ptr,
    nal: Int,
    nau: Int,
    number_of_nodes: Int,
    eigenvalues: Ptr,
) -> None:
    """`NetLSD._updown_linear_approx`; `eigenvalues` gets `number_of_nodes`.

    `nal` and `nau` are upstream's `len()` of the two blocks. The
    `eigenvalues[nal - 1 : -nau + 1]` slice is written in place: its endpoints
    are the very values the two block copies just put there, so it needs no
    scratch and the overlap is exact.
    """
    linalg.fill(eigenvalues, 0.0, number_of_nodes)
    linalg.copy(eigenvalues_lower, eigenvalues, nal)
    linalg.copy(
        eigenvalues_upper, eigenvalues.unsafe_offset(number_of_nodes - nau), nau
    )
    linalg.linspace(
        eigenvalues.unsafe_offset(nal - 1),
        eigenvalues_lower.unsafe_load(nal - 1),
        eigenvalues_upper.unsafe_load(0),
        number_of_nodes - nal - nau + 2,
    )


def calculate_eigenvalues(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    number_of_nodes: Int,
    approximations: Int,
    laplacian: Ptr,
    vectors: Ptr,
    vt: Ptr,
    picked: IPtr,
    eigenvalues_lower: Ptr,
    eigenvalues_upper: Ptr,
    eigenvalues: Ptr,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """`NetLSD._calculate_eigenvalues`, taking the Laplacian as a CSR.

    Upstream passes the `sps.coo_matrix` of the normalized Laplacian; here the
    graph is the CSR and the Laplacian is built inside, because
    `nx.normalized_laplacian_matrix` is what that COO matrix holds.
    `eigenvalues` gets `number_of_nodes` values when the approximation branch
    runs and `number_of_nodes - 2` when the exact one does; the two block
    buffers are `approximations` long and are only touched on the first branch.
    Returns False when the Jacobi sweep runs out of passes, as in
    `calculate_sf`.
    """
    var n = number_of_nodes
    spectral.normalized_laplacian(indptr, indices, values, n, laplacian)
    if linalg.jacobi_eigh(laplacian, vectors, vt, n, sweeps, tol) >= sweeps:
        return False
    # `eig_select` writes the selected columns into its own output buffer, which
    # must not be the eigenvector block it reads: the same slot is written on
    # an earlier iteration and re-read on a later one. `vt` is the transposed
    # block `jacobi_eigh` used and is dead from here on.
    if 2 * approximations < n:
        # which="SM", k=approximations, then [::-1]. `eigsh` with
        # return_eigenvectors=False sorts by absolute value -- 'SM' descending,
        # 'LM' ascending -- so the reversal upstream applies leaves the lower
        # block ascending, which is what an argmin walk already produces.
        spectral.eig_select(
            laplacian,
            vectors,
            n,
            approximations,
            True,
            eigenvalues_lower,
            vt,
            picked,
        )
        # which="LM", k=approximations: ascending as it comes back, so the
        # argmax walk's descending block is reversed.
        spectral.eig_select(
            laplacian,
            vectors,
            n,
            approximations,
            False,
            eigenvalues_upper,
            vt,
            picked,
        )
        for i in range(approximations // 2):
            var head = eigenvalues_upper.unsafe_load(i)
            eigenvalues_upper.unsafe_offset(i)[] = eigenvalues_upper.unsafe_load(
                approximations - 1 - i
            )
            eigenvalues_upper.unsafe_offset(approximations - 1 - i)[] = head
        updown_linear_approx(
            eigenvalues_lower,
            eigenvalues_upper,
            approximations,
            approximations,
            n,
            eigenvalues,
        )
    else:
        var k = n - 2
        spectral.eig_select(
            laplacian, vectors, n, k, False, eigenvalues, vt, picked
        )
        for i in range(k // 2):
            var head = eigenvalues.unsafe_load(i)
            eigenvalues.unsafe_offset(i)[] = eigenvalues.unsafe_load(k - 1 - i)
            eigenvalues.unsafe_offset(k - 1 - i)[] = head
    return True


def strip_self_loops(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    number_of_nodes: Int,
    out_indptr: IPtr,
    out_indices: IPtr,
    out_values: Ptr,
) -> Int:
    """`graph.remove_edges_from(nx.selfloop_edges(graph))` on a CSR.

    Upstream does this to a networkx graph, where it is a graph edit; here the
    caller's graph is read-only, so the loop-free copy is written to
    `out_indptr[n + 1]`, `out_indices[nnz]` and `out_values[nnz]`. Returns the
    new nonzero count. Dropping the diagonal entries also drops them from the
    degrees, which is what networkx recomputes.
    """
    out_indptr.unsafe_offset(0)[] = Int32(0)
    var count = 0
    for r in range(number_of_nodes):
        for k in range(
            Int(indptr.unsafe_load(r)), Int(indptr.unsafe_load(r + 1))
        ):
            if Int(indices.unsafe_load(k)) == r:
                continue
            out_indices.unsafe_offset(count)[] = indices.unsafe_load(k)
            out_values.unsafe_offset(count)[] = values.unsafe_load(k)
            count += 1
        out_indptr.unsafe_offset(r + 1)[] = Int32(count)
    return count


def calculate_netlsd(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    number_of_nodes: Int,
    scale_min: Float64,
    scale_max: Float64,
    scale_steps: Int,
    approximations: Int,
    loopless_indptr: IPtr,
    loopless_indices: IPtr,
    loopless_values: Ptr,
    laplacian: Ptr,
    vectors: Ptr,
    vt: Ptr,
    picked: IPtr,
    eigenvalues_lower: Ptr,
    eigenvalues_upper: Ptr,
    eigenvalues: Ptr,
    timescales: Ptr,
    dst: Ptr,
    sweeps: Int,
    tol: Float64,
) -> Bool:
    """`NetLSD._calculate_netlsd`; `dst` gets `scale_steps` values.

    Every buffer of `calculate_eigenvalues` and `calculate_heat_kernel_trace`
    in one call, plus the loop-free copy from `strip_self_loops`. Returns False
    when the Jacobi sweep runs out of passes, as in `calculate_sf`.
    """
    var n = number_of_nodes
    _ = strip_self_loops(
        indptr,
        indices,
        values,
        n,
        loopless_indptr,
        loopless_indices,
        loopless_values,
    )
    if not calculate_eigenvalues(
        loopless_indptr,
        loopless_indices,
        loopless_values,
        n,
        approximations,
        laplacian,
        vectors,
        vt,
        picked,
        eigenvalues_lower,
        eigenvalues_upper,
        eigenvalues,
        sweeps,
        tol,
    ):
        return False
    # `nodes` is `eigenvalues.shape[0]`, which the branch above decided.
    var nodes = n
    if 2 * approximations >= n:
        nodes = n - 2
    calculate_heat_kernel_trace(
        eigenvalues,
        nodes,
        scale_min,
        scale_max,
        scale_steps,
        timescales,
        dst,
    )
    return True
