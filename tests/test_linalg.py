"""Contract tests for the Mojo kernels behind `mojokarateclub`.

Every routine is checked against numpy, or - for the two graph kernels whose
upstream is networkx, which is not a dependency here - against a reference
written from the algorithm. The references are deliberately naive; they exist to
be obviously correct, not fast.
"""


import numpy as np
import pytest

import mojokarateclub as mk


def adjacencies(indptr, indices):
    """Python sets of neighbours for an undirected CSR."""
    return [set(indices[indptr[v] : indptr[v + 1]].tolist()) - {v} for v in range(indptr.size - 1)]


def ref_clustering(indptr, indices):
    """`networkx.clustering(G, v)` for every node."""
    adj = adjacencies(indptr, indices)
    out = np.zeros(len(adj))
    for v, nbrs in enumerate(adj):
        k = len(nbrs)
        if k < 2:
            continue
        ordered = sum(1 for w in nbrs for u in adj[w] if u != w and u in nbrs)
        out[v] = ordered / (k * (k - 1))
    return out


def ref_eccentricity(indptr, indices):
    """`networkx.eccentricity(G, v)`; unreachable-from-all nodes score -1."""
    adj = adjacencies(indptr, indices)
    n = len(adj)
    out = np.zeros(n)
    for v in range(n):
        seen = {v}
        frontier = [v]
        depth = 0
        while frontier and len(seen) < n:
            depth += 1
            nxt = []
            for w in frontier:
                for u in adj[w]:
                    if u not in seen:
                        seen.add(u)
                        nxt.append(u)
            frontier = nxt
        out[v] = float(depth) if len(seen) == n else -1.0
    return out


def to_csr(dense):
    """CSR `(indptr, indices, values)` of a dense square matrix."""
    rows, cols = np.nonzero(dense)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    values = dense[rows, cols]
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=dense.shape[0]), out=indptr[1:])
    return indptr, cols.astype(np.int32), values.astype(np.float64)


def symmetric_csr(dense):
    """CSR of `dense + dense.T`, so the graph really is undirected."""
    return to_csr(dense + dense.T)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260926)


# ------------------------------------------------------------------ dense


@pytest.mark.parametrize("n", [0, 1, 4, 5, 13])
def test_fill_and_copy(n, rng):
    src = np.ascontiguousarray(rng.normal(size=n))
    dst = np.zeros(n)
    mk.fill(dst, 2.5)
    assert np.array_equal(dst, np.full(n, 2.5))
    mk.copy(src, dst)
    assert np.array_equal(dst, src)


@pytest.mark.parametrize("n", [1, 4, 5, 13])
def test_axpy_in_place(n, rng):
    x = np.ascontiguousarray(rng.normal(size=n))
    y = np.ascontiguousarray(rng.normal(size=n))
    expected = y + 3.0 * x
    mk.axpy(3.0, x, y)
    assert np.allclose(y, expected)


@pytest.mark.parametrize("n", [1, 4, 5, 13])
def test_scale_in_place(n, rng):
    x = np.ascontiguousarray(rng.normal(size=n))
    expected = x * 0.25
    mk.scale(x, 0.25)
    assert np.allclose(x, expected)


@pytest.mark.parametrize("n", [4, 5, 13])
def test_scale_dot_and_norms(n, rng):
    x = np.ascontiguousarray(rng.normal(size=n))
    y = np.ascontiguousarray(rng.normal(size=n))
    assert mk.dot(x, y) == pytest.approx(float(x @ y))
    assert mk.total(x) == pytest.approx(float(x.sum()))
    assert mk.sum_squares(x) == pytest.approx(float(np.linalg.norm(x) ** 2))
    assert mk.frobenius(x) == pytest.approx(float(np.linalg.norm(x)))


def test_in_place_helpers_reject_wrong_dtype():
    y32 = np.zeros(4, dtype=np.float32)
    with pytest.raises(TypeError):
        mk.fill(y32, 1.0)


def test_index_buffers_must_be_int32():
    indptr = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(TypeError):
        mk.clustering(indptr, np.array([0, 1], dtype=np.int32))


def test_in_place_helpers_reject_a_read_only_buffer():
    y = np.zeros(4)
    y.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        mk.scale(y, 2.0)


def test_indptr_must_be_non_decreasing():
    # A decreasing indptr would make the kernels walk `row_range` backwards
    # and read before the start of the values buffer.
    indptr = np.array([0, 2, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="non-decreasing"):
        mk.clustering(indptr, np.array([0, 1], dtype=np.int32))


def test_non_contiguous_buffers_are_rejected():
    strided = np.zeros(8)[::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError):
        mk.fill(strided, 1.0)


# ------------------------------------------------------------------ products

@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 4), (4, 3, 3), (5, 5, 9)])
def test_gemm_matches_numpy(shape, rng):
    m, k, n = shape
    a = np.ascontiguousarray(rng.normal(size=(m, k)))
    b = np.ascontiguousarray(rng.normal(size=(k, n)))
    assert np.allclose(mk.gemm(a, b), a @ b)


@pytest.mark.parametrize("shape", [(2, 3, 4), (5, 4, 5)])
def test_gemm_nt_matches_numpy(shape, rng):
    m, k, n = shape
    a = np.ascontiguousarray(rng.normal(size=(m, k)))
    b = np.ascontiguousarray(rng.normal(size=(n, k)))
    assert np.allclose(mk.gemm_nt(a, b), a @ b.T)


def test_matvec_matches_numpy(rng):
    a = np.ascontiguousarray(rng.normal(size=(6, 5)))
    x = np.ascontiguousarray(rng.normal(size=5))
    assert np.allclose(mk.matvec(a, x), a @ x)


def test_gemm_rejects_inner_dimension_mismatch(rng):
    a = np.ascontiguousarray(rng.normal(size=(2, 3)))
    b = np.ascontiguousarray(rng.normal(size=(4, 2)))
    with pytest.raises(ValueError):
        mk.gemm(a, b)


# ------------------------------------------------------------------ eigen


@pytest.mark.parametrize("d", [1, 2, 5, 9])
def test_eigh_matches_lapack(d, rng):
    a = rng.normal(size=(d, d))
    a = np.ascontiguousarray(a + a.T)
    values, vectors = mk.eigh(a)
    expected = np.linalg.eigvalsh(a)
    assert np.allclose(values, expected, atol=1e-9)
    assert np.allclose(a @ vectors, vectors * values, atol=1e-8)
    assert np.allclose(vectors.T @ vectors, np.eye(d), atol=1e-8)


def test_eigh_leaves_its_input_alone(rng):
    a = np.ascontiguousarray(rng.normal(size=(4, 4)))
    symmetric = a + a.T
    original = symmetric.copy()
    mk.eigh(symmetric)
    assert np.array_equal(symmetric, original)


def test_eigh_reports_non_convergence(rng):
    a = np.ascontiguousarray(rng.normal(size=(12, 12)))
    a = a + a.T
    with pytest.raises(RuntimeError):
        mk.eigh(a, sweeps=1, tol=0.0)


# ------------------------------------------------------------------ inverse


@pytest.mark.parametrize("d", [1, 2, 3, 7])
def test_invert_matches_lapack(d, rng):
    a = np.ascontiguousarray(rng.normal(size=(d, d)) * 10 + d * np.eye(d))
    assert np.allclose(mk.invert(a), np.linalg.inv(a))


def test_invert_leaves_its_input_alone(rng):
    a = np.ascontiguousarray(rng.normal(size=(4, 4)))
    original = a.copy()
    mk.invert(a)
    assert np.array_equal(a, original)


def test_invert_raises_on_singular():
    singular = np.zeros((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        mk.invert(singular)


def test_invert_raises_on_non_square():
    with pytest.raises(ValueError):
        mk.invert(np.zeros((2, 3)))


# ------------------------------------------------------------------ utilities


@pytest.mark.parametrize(
    "lo,hi,n", [(0.0, 1.0, 0), (0.0, 1.0, 1), (0.0, 1.0, 5), (-2.0, 3.5, 4)]
)
def test_linspace_matches_numpy(lo, hi, n):
    assert np.array_equal(mk.linspace(lo, hi, n), np.linspace(lo, hi, n))


def test_l1_normalize_rows_matches_sklearn():
    x = np.ascontiguousarray([[3.0, -1.0], [0.0, 0.0], [2.0, 2.0]])
    expected = np.array([[0.75, -0.25], [0.0, 0.0], [0.5, 0.5]])
    assert np.allclose(mk.l1_normalize_rows(x), expected)
    assert np.allclose(np.abs(x).sum(axis=1), [1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "values,bins,lo,hi",
    [
        ([0.0, 0.25, 0.5, 0.75, 1.0], 4, 0.0, 1.0),
        ([-1.0, 0.0, 2.0, 5.0], 5, -1.0, 5.0),
        ([0.5, 0.5, 0.5], 1, 0.0, 1.0),
    ],
)
def test_histogram_matches_numpy(values, bins, lo, hi):
    x = np.ascontiguousarray(values)
    assert np.array_equal(
        mk.histogram(x, bins, lo, hi),
        np.histogram(x, bins=bins, range=(lo, hi))[0].astype(np.float64),
    )


def test_histogram_widens_a_degenerate_range_like_numpy():
    x = np.ascontiguousarray([1.0, 1.0])
    assert np.array_equal(
        mk.histogram(x, 4, 1.0, 1.0),
        np.histogram(x, bins=4, range=(1.0, 1.0))[0].astype(np.float64),
    )


def test_histogram_rejects_zero_bins():
    with pytest.raises(ValueError):
        mk.histogram(np.zeros(3), 0, 0.0, 1.0)


def test_dense_to_coo_lists_nonzeros_in_row_major_order():
    dense = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    rows, cols, values = mk.dense_to_coo(dense)
    assert rows.tolist() == [0, 1, 2]
    assert cols.tolist() == [1, 0, 2]
    assert values.tolist() == [2.0, 3.0, 4.0]


# ------------------------------------------------------------------ graphs


def test_clustering_on_a_triangle():
    indptr = np.array([0, 2, 4, 6], dtype=np.int32)
    indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    assert np.allclose(mk.clustering(indptr, indices), [1.0, 1.0, 1.0])


def test_clustering_on_a_path_scores_zero():
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    indices = np.array([1, 0, 1], dtype=np.int32)
    assert np.allclose(mk.clustering(indptr, indices), [0.0, 0.0, 0.0])


def test_clustering_handles_an_isolated_node():
    indptr = np.array([0, 2, 4, 4], dtype=np.int32)
    indices = np.array([1, 2, 0, 2], dtype=np.int32)
    got = mk.clustering(indptr, indices)
    assert np.allclose(got, ref_clustering(indptr, indices))
    assert got[2] == 0.0


def test_clustering_matches_reference_on_a_random_graph(rng):
    n = 24
    adjacency = rng.random((n, n)) < 0.3
    adjacency = np.triu(adjacency, 1)
    indptr, indices, _ = symmetric_csr(adjacency.astype(np.float64))
    got = mk.clustering(indptr, indices)
    assert np.allclose(got, ref_clustering(indptr, indices))
    assert got.max() > 0.0


def test_eccentricity_on_a_path():
    indptr = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    indices = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    assert np.array_equal(mk.eccentricity(indptr, indices), [3.0, 2.0, 2.0, 3.0])


def test_eccentricity_marks_a_disconnected_graph():
    indptr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    indices = np.array([1, 0, 3, 2], dtype=np.int32)
    got = mk.eccentricity(indptr, indices)
    assert np.array_equal(got, ref_eccentricity(indptr, indices))
    assert np.array_equal(got, [-1.0, -1.0, -1.0, -1.0])


def test_eccentricity_of_a_single_node_is_zero():
    indptr = np.array([0, 0], dtype=np.int32)
    indices = np.zeros(0, dtype=np.int32)
    assert np.array_equal(mk.eccentricity(indptr, indices), [0.0])


def test_eccentricity_matches_reference_on_a_random_graph(rng):
    n = 20
    adjacency = (rng.random((n, n)) < 0.4) * np.triu(np.ones((n, n)), 1)
    indptr, indices, _ = symmetric_csr(adjacency)
    assert np.array_equal(
        mk.eccentricity(indptr, indices), ref_eccentricity(indptr, indices)
    )


# ------------------------------------------------------------------ CSR


def test_csr_scatter_densifies(rng):
    dense = (rng.random((6, 6)) < 0.5) * rng.normal(size=(6, 6))
    indptr, indices, values = to_csr(dense)
    assert np.allclose(mk.csr_scatter(indptr, indices, values), dense)


def test_csr_matvec_matches_dense(rng):
    dense = (rng.random((5, 5)) < 0.6) * rng.normal(size=(5, 5))
    indptr, indices, values = to_csr(dense)
    x = np.ascontiguousarray(rng.normal(size=5))
    assert np.allclose(mk.csr_matvec(indptr, indices, values, x), dense @ x)


def test_csr_matvec_rejects_a_short_vector():
    indptr, indices, values = to_csr(np.eye(3))
    with pytest.raises(ValueError):
        mk.csr_matvec(indptr, indices, values, np.zeros(2))


def test_csr_matmul_matches_dense(rng):
    dense = (rng.random((5, 5)) < 0.6) * rng.normal(size=(5, 5))
    indptr, indices, values = to_csr(dense)
    x = np.ascontiguousarray(rng.normal(size=(5, 3)))
    assert np.allclose(mk.csr_matmul(indptr, indices, values, x), dense @ x)


def test_csr_dense_is_a_squared(rng):
    dense = (rng.random((6, 6)) < 0.5) * rng.normal(size=(6, 6))
    indptr, indices, values = to_csr(dense)
    assert np.allclose(mk.csr_dense(indptr, indices, values), dense @ dense)


def test_csr_row_scale_multiplies_each_row(rng):
    dense = (rng.random((4, 4)) < 0.6) * rng.normal(size=(4, 4))
    indptr, indices, values = to_csr(dense)
    diag = np.ascontiguousarray(rng.normal(size=4))
    scaled = mk.csr_row_scale(values, diag, indptr)
    expected = diag[:, None] * dense
    got = np.zeros_like(dense)
    for v, i in enumerate(indices):
        got[mk_csr_row(indptr, v), i] = scaled[v]
    assert np.allclose(got, expected)


def mk_csr_row(indptr, entry):
    """Row that CSR entry `entry` belongs to."""
    return int(np.searchsorted(indptr, entry, side="right")) - 1


# ------------------------------------------------------------------ COO


def test_coo_scatter_densifies(rng):
    dense = np.zeros((4, 4))
    rows = rng.integers(0, 4, 12).astype(np.int32)
    cols = rng.integers(0, 4, 12).astype(np.int32)
    values = np.ascontiguousarray(rng.normal(size=12))
    dense[rows, cols] = values
    got = mk.coo_scatter(rows, cols, values)
    assert np.allclose(got, dense)


def test_coo_scatter_lets_a_repeat_overwrite():
    rows = np.array([0, 0], dtype=np.int32)
    cols = np.array([1, 1], dtype=np.int32)
    values = np.array([2.0, 5.0])
    assert np.array_equal(
        mk.coo_scatter(rows, cols, values), [[0.0, 5.0], [0.0, 0.0]]
    )


def test_coo_matmul_sums_duplicates(rng):
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([0, 0, 1], dtype=np.int32)
    values = np.ascontiguousarray(np.array([1.0, 2.0, 4.0]))
    x = np.ascontiguousarray(rng.normal(size=(2, 3)))
    dense = np.zeros((2, 2))
    for r, c, v in zip(rows, cols, values):
        dense[r, c] += v
    assert np.allclose(mk.coo_matmul(rows, cols, values, x), dense @ x)


def test_coo_matmul_t_matches_the_transpose(rng):
    rows = rng.integers(0, 5, 15).astype(np.int32)
    cols = rng.integers(0, 5, 15).astype(np.int32)
    values = np.ascontiguousarray(rng.normal(size=15))
    x = np.ascontiguousarray(rng.normal(size=(5, 2)))
    dense = np.zeros((5, 5))
    np.add.at(dense, (rows, cols), values)
    assert np.allclose(mk.coo_matmul_t(rows, cols, values, x), dense.T @ x)


# ------------------------------------------------------------------ guards


def test_out_of_range_column_index_is_rejected():
    # A dangling column index would send the kernel past the end of `x` and
    # silently corrupt memory; the wrapper has to raise first.
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 7], dtype=np.int32)
    values = np.ones(2)
    with pytest.raises(ValueError, match="indices"):
        mk.csr_matvec(indptr, indices, values, np.zeros(2))


def test_negative_column_index_is_rejected():
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([-1], dtype=np.int32)
    with pytest.raises(ValueError, match="indices"):
        mk.clustering(indptr, indices)


def test_csr_triple_with_a_dangling_indptr_is_rejected():
    indptr = np.array([0, 1, 5], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="indptr"):
        mk.csr_scatter(indptr, indices, np.ones(1))


def test_csr_values_must_match_indices_length():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="values"):
        mk.csr_scatter(indptr, indices, np.ones(3))


def test_indptr_must_start_at_zero():
    indptr = np.array([1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="start at 0"):
        mk.clustering(indptr, np.array([0], dtype=np.int32))


def test_csr_row_scale_rejects_a_mismatched_diag():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="diag"):
        mk.csr_row_scale(np.ones(2), np.ones(3), indptr)


def test_csr_row_scale_rejects_an_indptr_that_overshoots_values():
    indptr = np.array([0, 1, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="indptr"):
        mk.csr_row_scale(np.ones(2), np.ones(2), indptr)


def test_csr_dense_rejects_a_column_beyond_the_row_count():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 9], dtype=np.int32)
    with pytest.raises(ValueError, match="indices"):
        mk.csr_dense(indptr, indices, np.ones(2))


def test_coo_rejects_mismatched_lengths():
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="same length"):
        mk.coo_scatter(rows, cols, np.ones(2))


def test_coo_matmul_rejects_a_column_beyond_x():
    rows = np.array([0], dtype=np.int32)
    cols = np.array([4], dtype=np.int32)
    with pytest.raises(ValueError, match="cols"):
        mk.coo_matmul(rows, cols, np.ones(1), np.zeros((2, 1)))


def test_coo_matmul_t_rejects_a_row_beyond_x():
    rows = np.array([4], dtype=np.int32)
    cols = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="rows"):
        mk.coo_matmul_t(rows, cols, np.ones(1), np.zeros((2, 1)))


def test_coo_rejects_a_negative_index():
    rows = np.array([-1], dtype=np.int32)
    cols = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        mk.coo_scatter(rows, cols, np.ones(1))


def test_dense_to_coo_rejects_a_non_square_block():
    with pytest.raises(ValueError, match="square"):
        mk.dense_to_coo(np.zeros((2, 3)))


def test_l1_normalize_rows_rejects_a_1d_array():
    with pytest.raises(ValueError, match="2-D"):
        mk.l1_normalize_rows(np.zeros(4))


def test_histogram_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="lo <= hi"):
        mk.histogram(np.zeros(3), 4, 1.0, 0.0)


@pytest.mark.parametrize(
    "call,args",
    [
        (mk.gemm, (np.zeros(4), np.zeros(4))),
        (mk.gemm_nt, (np.zeros((2, 2)), np.zeros(4))),
        (mk.matvec, (np.zeros(4), np.zeros(4))),
    ],
)
def test_products_reject_wrong_rank(call, args):
    with pytest.raises(ValueError):
        call(*args)


# ------------------------------------------------------------------ loading


def test_missing_library_names_the_build_task(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJOKARATECLUB_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="pixi run build"):
        mk.load_library()


def test_library_path_defaults_into_dist():
    assert mk.library_path().name == "libmojokarateclub.so"
    assert mk.library_path().parent.name == "dist"


def test_every_declared_symbol_exists_in_the_library():
    lib = mk.load_library()
    for name in mk._SIGNATURES:
        assert hasattr(lib, name)


def test_no_buffer_argument_is_narrower_than_mojo_int():
    # Mojo `Int` is 64-bit, so a narrower ctypes conversion would silently
    # truncate an address rather than raise.
    for name, (_, argtypes) in mk._SIGNATURES.items():
        for argtype in argtypes:
            assert argtype in (mk._I64, mk._F64), name
