"""C ABI surface over the kernels, loaded by ctypes from `python/mojokarateclub`.

Every buffer crosses the boundary as an `Int` address and is rebuilt inside the
wrapper: Mojo pointers are non-nullable, so they cannot be parameters of an
`abi("C")` function. Symbols are `kc_`-prefixed because the library is loaded
with `RTLD_LOCAL` into whatever process the tests happen to run in.

The estimators' kernels take their scratch as explicit trailing arguments, so
the Python layer allocates it and the sizes live with the estimators rather
than with the kernel.
"""

import diffuser
import fgsd
import hashing
import kclinalg as linalg
import laplacianeigenmaps
import ldp
import line
import netlsd
import nodesketch
import sf
import spectral
import treefeatures
import walker

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime UPtr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def uptr(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def bptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


# ------------------------------------------------------------------ dense

@export("kc_fill")
def kc_fill(x: Int, value: Float64, n: Int) abi("C") -> None:
    linalg.fill(fptr(x), value, n)


@export("kc_copy")
def kc_copy(src: Int, dst: Int, n: Int) abi("C") -> None:
    linalg.copy(fptr(src), fptr(dst), n)


@export("kc_axpy")
def kc_axpy(alpha: Float64, x: Int, y: Int, n: Int) abi("C") -> None:
    linalg.axpy(alpha, fptr(x), fptr(y), n)


@export("kc_scale")
def kc_scale(x: Int, alpha: Float64, n: Int) abi("C") -> None:
    linalg.scale(fptr(x), alpha, n)


@export("kc_dot")
def kc_dot(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return linalg.dot(fptr(a), fptr(b), n)


@export("kc_sum")
def kc_sum(a: Int, n: Int) abi("C") -> Float64:
    return linalg.sumf(fptr(a), n)


@export("kc_sum_squares")
def kc_sum_squares(a: Int, n: Int) abi("C") -> Float64:
    return linalg.sum_squares(fptr(a), n)


@export("kc_frobenius")
def kc_frobenius(a: Int, n: Int) abi("C") -> Float64:
    return linalg.frobenius(fptr(a), n)


@export("kc_gemm")
def kc_gemm(a: Int, b: Int, dst: Int, m: Int, k: Int, n: Int) abi("C") -> None:
    linalg.gemm(fptr(a), fptr(b), fptr(dst), m, k, n)


@export("kc_gemm_nt")
def kc_gemm_nt(a: Int, b: Int, dst: Int, m: Int, k: Int, n: Int) abi("C") -> None:
    linalg.gemm_nt(fptr(a), fptr(b), fptr(dst), m, k, n)


@export("kc_matvec")
def kc_matvec(a: Int, x: Int, dst: Int, rows: Int, cols: Int) abi("C") -> None:
    linalg.matvec(fptr(a), fptr(x), fptr(dst), rows, cols)


@export("kc_jacobi_eigh")
def kc_jacobi_eigh(
    a: Int, vectors: Int, vt: Int, d: Int, sweeps: Int, tol: Float64
) abi("C") -> Int:
    return linalg.jacobi_eigh(fptr(a), fptr(vectors), fptr(vt), d, sweeps, tol)


@export("kc_dense_to_coo")
def kc_dense_to_coo(
    a: Int, n: Int, rows: Int, cols: Int, values: Int
) abi("C") -> Int:
    return linalg.dense_to_coo(fptr(a), n, iptr(rows), iptr(cols), fptr(values))


@export("kc_invert")
def kc_invert(src: Int, work: Int, dst: Int, d: Int) abi("C") -> Bool:
    return linalg.invert(fptr(src), fptr(work), fptr(dst), d)

@export("kc_linspace")
def kc_linspace(dst: Int, lo: Float64, hi: Float64, n: Int) abi("C") -> None:
    linalg.linspace(fptr(dst), lo, hi, n)


@export("kc_l1_normalize_rows")
def kc_l1_normalize_rows(x: Int, n: Int, cols: Int) abi("C") -> None:
    linalg.l1_normalize_rows(fptr(x), n, cols)


@export("kc_histogram")
def kc_histogram(
    x: Int, dst: Int, n: Int, bins: Int, lo: Float64, hi: Float64
) abi("C") -> None:
    linalg.histogram(fptr(x), fptr(dst), n, bins, lo, hi)


# ------------------------------------------------------------------ graphs

@export("kc_clustering")
def kc_clustering(
    indptr: Int, indices: Int, dst: Int, n: Int, marker: Int
) abi("C") -> None:
    linalg.clustering(iptr(indptr), iptr(indices), fptr(dst), n, fptr(marker))


@export("kc_eccentricity")
def kc_eccentricity(
    indptr: Int, indices: Int, dst: Int, n: Int, queue: Int, seen: Int
) abi("C") -> None:
    linalg.eccentricity(
        iptr(indptr), iptr(indices), fptr(dst), n, iptr(queue), fptr(seen)
    )


@export("kc_csr_row_scale")
def kc_csr_row_scale(
    values: Int, scale: Int, indptr: Int, n: Int
) abi("C") -> None:
    linalg.csr_row_scale(fptr(values), fptr(scale), iptr(indptr), n)


@export("kc_csr_matvec")
def kc_csr_matvec(
    indptr: Int, indices: Int, values: Int, x: Int, dst: Int, n: Int, cols: Int
) abi("C") -> None:
    linalg.csr_matvec(
        iptr(indptr), iptr(indices), fptr(values), fptr(x), fptr(dst), n, cols
    )


@export("kc_csr_matmul")
def kc_csr_matmul(
    indptr: Int,
    indices: Int,
    values: Int,
    x: Int,
    dst: Int,
    n: Int,
    cols: Int,
    width: Int,
) abi("C") -> None:
    linalg.csr_matmul(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        fptr(x),
        fptr(dst),
        n,
        cols,
        width,
    )


@export("kc_csr_dense")
def kc_csr_dense(
    indptr: Int, indices: Int, values: Int, n: Int, dst: Int
) abi("C") -> None:
    linalg.csr_dense(iptr(indptr), iptr(indices), fptr(values), n, fptr(dst))


@export("kc_csr_scatter")
def kc_csr_scatter(
    indptr: Int, indices: Int, values: Int, n: Int, cols: Int, dst: Int
) abi("C") -> None:
    linalg.csr_scatter(
        iptr(indptr), iptr(indices), fptr(values), n, cols, fptr(dst)
    )


@export("kc_coo_matmul")
def kc_coo_matmul(
    rows: Int,
    cols: Int,
    values: Int,
    nnz: Int,
    x: Int,
    dst: Int,
    n: Int,
    width: Int,
) abi("C") -> None:
    linalg.coo_matmul(
        iptr(rows), iptr(cols), fptr(values), nnz, fptr(x), fptr(dst), n, width
    )


@export("kc_coo_matmul_t")
def kc_coo_matmul_t(
    rows: Int,
    cols: Int,
    values: Int,
    nnz: Int,
    x: Int,
    dst: Int,
    n: Int,
    width: Int,
) abi("C") -> None:
    linalg.coo_matmul_t(
        iptr(rows), iptr(cols), fptr(values), nnz, fptr(x), fptr(dst), n, width
    )


@export("kc_coo_scatter")
def kc_coo_scatter(
    rows: Int, cols: Int, values: Int, nnz: Int, n: Int, dst: Int
) abi("C") -> None:
    linalg.coo_scatter(iptr(rows), iptr(cols), fptr(values), nnz, n, fptr(dst))

# --------------------------------------------------------------- hashing


@export("kc_md5_digest")
def kc_md5_digest(
    data: Int, n: Int, digest: Int, m: Int, state: Int, tail: Int
) abi("C") -> None:
    hashing.md5(bptr(data), n, uptr(digest), uptr(m), uptr(state), bptr(tail))


# ------------------------------------------------------------------- ldp


@export("kc_ldp_log_degrees")
def kc_ldp_log_degrees(
    indptr: Int, indices: Int, n: Int, dst: Int
) abi("C") -> None:
    ldp.log_degrees(iptr(indptr), iptr(indices), n, fptr(dst))


@export("kc_ldp_features")
def kc_ldp_features(
    indptr: Int, indices: Int, degrees: Int, n: Int, dst: Int
) abi("C") -> None:
    ldp.ldp_features(iptr(indptr), iptr(indices), fptr(degrees), n, fptr(dst))


@export("kc_ldp_embedding")
def kc_ldp_embedding(
    features: Int, n: Int, bins: Int, dst: Int, column: Int
) abi("C") -> None:
    ldp.ldp_embedding(fptr(features), n, bins, fptr(dst), fptr(column))


# ------------------------------------------------------------------ fgsd


@export("kc_fgsd_similarity")
def kc_fgsd_similarity(
    indptr: Int,
    indices: Int,
    values: Int,
    n: Int,
    laplacian: Int,
    pinv: Int,
    scratch_a: Int,
    scratch_b: Int,
    scratch_c: Int,
    picked: Int,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
) abi("C") -> Bool:
    return fgsd.fgsd_similarity(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        n,
        fptr(laplacian),
        fptr(pinv),
        fptr(scratch_a),
        fptr(scratch_b),
        fptr(scratch_c),
        iptr(picked),
        rcond,
        sweeps,
        tol,
    )


@export("kc_fgsd_embedding")
def kc_fgsd_embedding(
    indptr: Int,
    indices: Int,
    values: Int,
    n: Int,
    laplacian: Int,
    pinv: Int,
    scratch_a: Int,
    scratch_b: Int,
    scratch_c: Int,
    picked: Int,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
    hist_bins: Int,
    hist_lo: Float64,
    hist_hi: Float64,
    histogram_out: Int,
) abi("C") -> Bool:
    return fgsd.fgsd_embedding(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        n,
        fptr(laplacian),
        fptr(pinv),
        fptr(scratch_a),
        fptr(scratch_b),
        fptr(scratch_c),
        iptr(picked),
        rcond,
        sweeps,
        tol,
        hist_bins,
        hist_lo,
        hist_hi,
        fptr(histogram_out),
    )


# -------------------------------------------------------------------- sf


@export("kc_sf_calculate")
def kc_sf_calculate(
    indptr: Int,
    indices: Int,
    values: Int,
    n: Int,
    dimensions: Int,
    laplacian: Int,
    vectors: Int,
    vt: Int,
    picked: Int,
    selected: Int,
    embedding: Int,
    vectors_out: Int,
    sweeps: Int,
    tol: Float64,
) abi("C") -> Bool:
    return sf.calculate_sf(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        n,
        dimensions,
        fptr(laplacian),
        fptr(vectors),
        fptr(vt),
        iptr(picked),
        fptr(selected),
        fptr(embedding),
        fptr(vectors_out),
        sweeps,
        tol,
    )


# ---------------------------------------------------------------- netlsd


@export("kc_netlsd_calculate")
def kc_netlsd_calculate(
    indptr: Int,
    indices: Int,
    values: Int,
    n: Int,
    scale_min: Float64,
    scale_max: Float64,
    scale_steps: Int,
    approximations: Int,
    loopless_indptr: Int,
    loopless_indices: Int,
    loopless_values: Int,
    laplacian: Int,
    vectors: Int,
    vt: Int,
    picked: Int,
    eigenvalues_lower: Int,
    eigenvalues_upper: Int,
    eigenvalues: Int,
    timescales: Int,
    dst: Int,
    sweeps: Int,
    tol: Float64,
) abi("C") -> Bool:
    return netlsd.calculate_netlsd(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        n,
        scale_min,
        scale_max,
        scale_steps,
        approximations,
        iptr(loopless_indptr),
        iptr(loopless_indices),
        fptr(loopless_values),
        fptr(laplacian),
        fptr(vectors),
        fptr(vt),
        iptr(picked),
        fptr(eigenvalues_lower),
        fptr(eigenvalues_upper),
        fptr(eigenvalues),
        fptr(timescales),
        fptr(dst),
        sweeps,
        tol,
    )


# ---------------------------------------------------- laplacian eigenmaps


@export("kc_laplacian_eigenmaps_fit")
def kc_laplacian_eigenmaps_fit(
    indptr: Int,
    indices: Int,
    values: Int,
    n: Int,
    dimensions: Int,
    maximum_number_of_iterations: Int,
    seed: Int,
    embedding: Int,
    eigenvalues: Int,
    laplacian: Int,
    vectors: Int,
    vt: Int,
    picked: Int,
    sweeps: Int,
    tol: Float64,
) abi("C") -> Bool:
    return laplacianeigenmaps.fit(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        n,
        dimensions,
        maximum_number_of_iterations,
        seed,
        fptr(embedding),
        fptr(eigenvalues),
        fptr(laplacian),
        fptr(vectors),
        fptr(vt),
        iptr(picked),
        sweeps,
        tol,
    )


# ------------------------------------------------------------- nodesketch


@export("kc_nodesketch_fit")
def kc_nodesketch_fit(
    n: Int,
    dimensions: Int,
    iterations: Int,
    decay: Float64,
    seed: Int,
    indptr: Int,
    indices: Int,
    orig_rows: Int,
    orig_cols: Int,
    orig_vals: Int,
    sla_rows: Int,
    sla_cols: Int,
    sla_vals: Int,
    sla_cap: Int,
    hashes: Int,
    sketch: Int,
    state: Int,
    stamp: Int,
    counts: Int,
    touched: Int,
    acc: Int,
    min_val: Int,
) abi("C") -> Int:
    return nodesketch.fit(
        n,
        dimensions,
        iterations,
        decay,
        UInt32(seed),
        iptr(indptr),
        iptr(indices),
        iptr(orig_rows),
        iptr(orig_cols),
        fptr(orig_vals),
        iptr(sla_rows),
        iptr(sla_cols),
        fptr(sla_vals),
        sla_cap,
        fptr(hashes),
        iptr(sketch),
        uptr(state),
        iptr(stamp),
        iptr(counts),
        iptr(touched),
        fptr(acc),
        fptr(min_val),
    )


@export("kc_nodesketch_get_embedding")
def kc_nodesketch_get_embedding(
    n: Int, dimensions: Int, sketch: Int, embedding: Int
) abi("C") -> None:
    nodesketch.get_embedding(n, dimensions, iptr(sketch), fptr(embedding))


# ------------------------------------------------------------------- line


@export("kc_line_fit_first_order")
def kc_line_fit_first_order(
    n: Int,
    dimensions: Int,
    epochs: Int,
    mini_batch_size: Int,
    learning_rate: Float64,
    learning_rate_decay: Float64,
    number_of_edges: Int,
    edges: Int,
    seed: Int,
    embedding: Int,
    gradient: Int,
    positive_index: Int,
    positive_batch: Int,
    negative_batch: Int,
    src_embedding: Int,
    dst_embedding: Int,
    activations: Int,
    base: Int,
    state: Int,
) abi("C") -> None:
    line.fit_first_order(
        n,
        dimensions,
        epochs,
        mini_batch_size,
        learning_rate,
        learning_rate_decay,
        number_of_edges,
        iptr(edges),
        seed,
        fptr(embedding),
        fptr(gradient),
        iptr(positive_index),
        iptr(positive_batch),
        iptr(negative_batch),
        fptr(src_embedding),
        fptr(dst_embedding),
        fptr(activations),
        fptr(base),
        uptr(state),
    )


@export("kc_line_fit_second_order")
def kc_line_fit_second_order(
    n: Int,
    dimensions: Int,
    epochs: Int,
    mini_batch_size: Int,
    learning_rate: Float64,
    learning_rate_decay: Float64,
    number_of_edges: Int,
    edges: Int,
    seed: Int,
    src_embedding: Int,
    dst_embedding: Int,
    src_gradient: Int,
    dst_gradient: Int,
    positive_index: Int,
    positive_batch: Int,
    negative_batch: Int,
    src_batch: Int,
    dst_batch: Int,
    activations: Int,
    base: Int,
    state: Int,
) abi("C") -> None:
    line.fit_second_order(
        n,
        dimensions,
        epochs,
        mini_batch_size,
        learning_rate,
        learning_rate_decay,
        number_of_edges,
        iptr(edges),
        seed,
        fptr(src_embedding),
        fptr(dst_embedding),
        fptr(src_gradient),
        fptr(dst_gradient),
        iptr(positive_index),
        iptr(positive_batch),
        iptr(negative_batch),
        fptr(src_batch),
        fptr(dst_batch),
        fptr(activations),
        fptr(base),
        uptr(state),
    )


# ----------------------------------------------------------------- walker


@export("kc_mt_init_genrand")
def kc_mt_init_genrand(state: Int, s: Int) abi("C") -> None:
    walker.mt_init_genrand(uptr(state), UInt32(s))


@export("kc_mt_genrand_uint32")
def kc_mt_genrand_uint32(state: Int) abi("C") -> Int:
    return Int(walker.mt_genrand_uint32(uptr(state)))


@export("kc_mt_genrand_res53")
def kc_mt_genrand_res53(state: Int) abi("C") -> Float64:
    return walker.mt_genrand_res53(uptr(state))


@export("kc_random_walker_do_walk")
def kc_random_walker_do_walk(
    indptr: Int, indices: Int, node: Int, walk_length: Int, walk: Int, state: Int
) abi("C") -> Int:
    return walker.random_walker_do_walk(
        iptr(indptr), iptr(indices), node, walk_length, iptr(walk), uptr(state)
    )


@export("kc_random_walker_do_walks")
def kc_random_walker_do_walks(
    indptr: Int,
    indices: Int,
    n: Int,
    walk_length: Int,
    walk_number: Int,
    walk: Int,
    state: Int,
) abi("C") -> Int:
    return walker.random_walker_do_walks(
        iptr(indptr),
        iptr(indices),
        n,
        walk_length,
        walk_number,
        iptr(walk),
        uptr(state),
    )


@export("kc_biased_random_walker_do_walk")
def kc_biased_random_walker_do_walk(
    indptr: Int,
    indices: Int,
    values: Int,
    weighted: Int,
    node: Int,
    walk_length: Int,
    p: Float64,
    q: Float64,
    walk: Int,
    state: Int,
    weights: Int,
    mark: Int,
) abi("C") -> Int:
    return walker.biased_random_walker_do_walk(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        weighted != 0,
        node,
        walk_length,
        p,
        q,
        iptr(walk),
        uptr(state),
        fptr(weights),
        iptr(mark),
    )


@export("kc_biased_random_walker_do_walks")
def kc_biased_random_walker_do_walks(
    indptr: Int,
    indices: Int,
    values: Int,
    weighted: Int,
    n: Int,
    walk_length: Int,
    walk_number: Int,
    p: Float64,
    q: Float64,
    walk: Int,
    state: Int,
    weights: Int,
    mark: Int,
) abi("C") -> Int:
    return walker.biased_random_walker_do_walks(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        weighted != 0,
        n,
        walk_length,
        walk_number,
        p,
        q,
        iptr(walk),
        uptr(state),
        fptr(weights),
        iptr(mark),
    )


# --------------------------------------------------------------- diffuser


@export("kc_eulerian_diffuser_run_diffusion_process")
def kc_eulerian_diffuser_run_diffusion_process(
    indptr: Int,
    indices: Int,
    node: Int,
    diffusion_cover: Int,
    generation: Int,
    infected: Int,
    marked: Int,
    head: Int,
    tail: Int,
    degree: Int,
    stamp: Int,
    next_edge: Int,
    edge_dst: Int,
    stack: Int,
    circuit: Int,
    state: Int,
) abi("C") -> Int:
    return diffuser.eulerian_diffuser_run_diffusion_process(
        iptr(indptr),
        iptr(indices),
        node,
        diffusion_cover,
        generation,
        iptr(infected),
        iptr(marked),
        iptr(head),
        iptr(tail),
        iptr(degree),
        iptr(stamp),
        iptr(next_edge),
        iptr(edge_dst),
        iptr(stack),
        iptr(circuit),
        uptr(state),
    )


@export("kc_eulerian_diffuser_do_diffusions")
def kc_eulerian_diffuser_do_diffusions(
    indptr: Int,
    indices: Int,
    n: Int,
    diffusion_number: Int,
    diffusion_cover: Int,
    diffusions: Int,
    offsets: Int,
    state: Int,
    infected: Int,
    marked: Int,
    head: Int,
    tail: Int,
    degree: Int,
    stamp: Int,
    next_edge: Int,
    edge_dst: Int,
    stack: Int,
) abi("C") -> Int:
    return diffuser.eulerian_diffuser_do_diffusions(
        iptr(indptr),
        iptr(indices),
        n,
        diffusion_number,
        diffusion_cover,
        iptr(diffusions),
        iptr(offsets),
        uptr(state),
        iptr(infected),
        iptr(marked),
        iptr(head),
        iptr(tail),
        iptr(degree),
        iptr(stamp),
        iptr(next_edge),
        iptr(edge_dst),
        iptr(stack),
    )


# --------------------------------------------------------- tree features


@export("kc_wl_hash")
def kc_wl_hash(
    n: Int,
    indptr: Int,
    indices: Int,
    base_bytes: Int,
    base_offsets: Int,
    wl_iterations: Int,
    erase_base: Int,
    extracted: Int,
    msg: Int,
    order: Int,
    order_tmp: Int,
    m: Int,
    state: Int,
    tail: Int,
) abi("C") -> None:
    treefeatures.wl_hash(
        n,
        iptr(indptr),
        iptr(indices),
        bptr(base_bytes),
        iptr(base_offsets),
        wl_iterations,
        erase_base,
        uptr(extracted),
        bptr(msg),
        iptr(order),
        iptr(order_tmp),
        uptr(m),
        uptr(state),
        bptr(tail),
    )


# --------------------------------------------------------------- spectral


@export("kc_normalized_adjacency")
def kc_normalized_adjacency(
    indptr: Int, indices: Int, values: Int, n: Int, dst: Int
) abi("C") -> None:
    spectral.normalized_adjacency(
        iptr(indptr), iptr(indices), fptr(values), n, fptr(dst)
    )


@export("kc_symmetric_pinv")
def kc_symmetric_pinv(
    a: Int,
    tmp: Int,
    vectors: Int,
    vt: Int,
    d: Int,
    rcond: Float64,
    sweeps: Int,
    tol: Float64,
) abi("C") -> Bool:
    return spectral.symmetric_pinv(
        fptr(a), fptr(tmp), fptr(vectors), fptr(vt), d, rcond, sweeps, tol
    )
