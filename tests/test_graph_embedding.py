"""Parity against real upstream karateclub for the graph-level estimators.

`tests/vectors/upstream.npz` is produced by `tools/dump_upstream.py` inside the
`upstream` pixi environment, which is the only one where karateclub 1.3.3 and
its pins install. Every number asserted here comes from that dump, not from a
reference this repo wrote.
"""

import networkx as nx
import numpy as np
import pytest

import mojokarateclub as mk
from _graphs import build_graph

SMALL = ["karate", "gnp60", "lollipop", "isolated", "grid"]


def _permuted_insertion_order_graph():
    """Four nodes added out of order: labels are 0..3, insertion order is not."""
    graph = nx.Graph()
    graph.add_edge(2, 0)
    graph.add_node(1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 3)
    return graph


def test_rows_are_ordered_by_node_label_not_by_insertion_order():
    # networkx keeps insertion order, and every consumer reads row i as node i.
    graph = _permuted_insertion_order_graph()
    indptr, indices, _ = mk.estimator.csr_from_graph(graph)
    order = list(graph.nodes())
    assert sorted(order) == list(range(graph.number_of_nodes())), order
    for row, node in enumerate(sorted(order)):
        neighbours = {n for n in graph.neighbors(node) if n != node}
        assert set(indices[indptr[row] : indptr[row + 1]].tolist()) == neighbours


def test_an_embedding_row_belongs_to_its_node():
    graph = _permuted_insertion_order_graph()
    same = nx.Graph()
    same.add_nodes_from(range(graph.number_of_nodes()))
    same.add_edges_from(sorted({tuple(sorted(e)) for e in graph.edges()}))
    for model in (
        mk.LaplacianEigenmaps(dimensions=1),
        mk.NodeSketch(dimensions=4, iterations=1, seed=1),
    ):
        model.fit(graph)
        permuted = model.get_embedding()
        model.fit(same)
        assert np.array_equal(permuted, model.get_embedding())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mk.LDP(bins=0),
        lambda: mk.FGSD(hist_bins=0),
        lambda: mk.NetLSD(scale_steps=0),
        lambda: mk.NetLSD(approximations=0),
    ],
)
def test_a_degenerate_histogram_parameter_is_refused(factory):
    # Each of these reaches a kernel that would index a zero-length histogram,
    # a zero-length eigenvalue block or a zero-length output; numpy and
    # `eigsh` raise for all of them upstream.
    with pytest.raises(ValueError):
        factory()

def _weighted_path():
    """A 4-node path whose first edge is heavier than the rest."""
    weighted = nx.Graph()
    weighted.add_nodes_from(range(4))
    weighted.add_edge(0, 1, weight=10.0)
    weighted.add_edge(1, 2)
    weighted.add_edge(2, 3)
    plain = nx.Graph()
    plain.add_nodes_from(range(4))
    plain.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return weighted, plain


def test_the_spectral_estimators_read_edge_weights_like_upstream():
    # `nx.normalized_laplacian_matrix` defaults to `weight="weight"`, so an
    # edge weight has to reach the Laplacian or the answer is upstream's
    # unweighted one.
    weighted, plain = _weighted_path()
    laplacian = mk.LaplacianEigenmaps(dimensions=2)
    laplacian.fit(weighted)
    heavy = laplacian.get_embedding()
    laplacian.fit(plain)
    assert not np.allclose(heavy, laplacian.get_embedding())
    for model in (mk.SF(dimensions=2), mk.NetLSD(scale_steps=8, approximations=4)):
        model.fit([weighted])
        with_weights = model.get_embedding()
        model.fit([plain])
        assert not np.allclose(with_weights, model.get_embedding())


def test_ldp_ignores_edge_weights():
    # `LDP` counts degrees, which upstream takes unweighted.
    weighted, plain = _weighted_path()
    first, second = mk.LDP(bins=8), mk.LDP(bins=8)
    first.fit([weighted])
    second.fit([plain])
    assert np.array_equal(first.get_embedding(), second.get_embedding())


def test_ldp_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.LDP(bins=8)
        model.fit([graph])
        expected = upstream_vectors[f"ldp/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.array_equal(got, expected), name


def test_fgsd_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.FGSD(hist_bins=32, hist_range=20)
        model.fit([graph])
        expected = upstream_vectors[f"fgsd/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.allclose(got, expected, rtol=0, atol=1e-9), (
            name,
            float(np.abs(got - expected).max()),
        )


def test_sf_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.SF(dimensions=8)
        model.fit([graph])
        expected = upstream_vectors[f"sf/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.allclose(got, expected, rtol=0, atol=1e-9), (
            name,
            float(np.abs(got - expected).max()),
        )


def test_netlsd_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.NetLSD(scale_steps=16, approximations=4)
        model.fit([graph])
        expected = upstream_vectors[f"netlsd/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.allclose(got, expected, rtol=0, atol=1e-5), (
            name,
            float(np.abs(got - expected).max()),
        )


def test_laplacian_eigenmaps_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.LaplacianEigenmaps(dimensions=8)
        model.fit(graph)
        expected = upstream_vectors[f"laplacian/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        # Eigenvector signs are arbitrary on both sides, and where an
        # eigenspace is degenerate the basis inside it is arbitrary too, so
        # compare the subspace rather than the columns. The `isolated` fixture
        # is the case that forces it: six zero eigenvalues and a request for
        # eight vectors, so two of the four 1.5-eigenvectors get picked and
        # which two is not determined by anything.
        if name != "isolated":
            gap = np.abs(got @ got.T - expected @ expected.T).max()
            assert gap < 1e-9, (name, float(gap))
        _assert_eigenvectors(graph, got, name)


def _assert_eigenvectors(graph, vectors, name):
    """The vectors must be orthonormal and diagonalise the Laplacian."""
    n = graph.number_of_nodes()
    graph = mk.Estimator()._check_graph(graph)
    laplacian = np.asarray(
        nx.normalized_laplacian_matrix(graph, nodelist=range(n)).todense()
    )
    gram = vectors.T @ vectors
    assert np.abs(gram - np.eye(gram.shape[0])).max() < 1e-9, name
    compressed = vectors.T @ laplacian @ vectors
    off = compressed - np.diag(np.diag(compressed))
    assert np.abs(off).max() < 1e-9, (name, float(np.abs(off).max()))
    eigenvalues = np.diag(compressed)
    assert np.all(np.diff(eigenvalues) >= -1e-9), name


def test_wl_hashing_matches_upstream(upstream_vectors):
    from mojokarateclub.utils import WeisfeilerLehmanHashing

    for name in SMALL:
        for iterations in (1, 2):
            for erase in (False, True):
                key = f"wl/{name}/{iterations}/{int(erase)}"
                assert key in upstream_vectors, (
                    f"{key} missing from the upstream dump; regenerate it with "
                    "`pixi run vectors`"
                )
                graph, _ = build_graph(name, integrity=True)
                wl = WeisfeilerLehmanHashing(graph, iterations, False, erase)

                assert (
                    _as_object_array(wl).tolist()
                    == upstream_vectors[key].tolist()
                ), key

    graph, _ = build_graph("karate", integrity=True)
    for node in graph.nodes():
        graph.nodes[node]["feature"] = "attr-%d" % (node % 3)
    wl = WeisfeilerLehmanHashing(graph, 2, True, False)
    assert (
        _as_object_array(wl).tolist()
        == upstream_vectors["wl/attributed/2/0"].tolist()
    )


def _as_object_array(wl):
    return np.array(
        [
            np.array([node] + list(features), dtype=object)
            for node, features in wl.get_node_features().items()
        ],
        dtype=object,
    )


def test_laplacian_eigenmaps_rejects_full_rank_request():
    graph, _ = build_graph("karate")
    with pytest.raises(ValueError):
        mk.LaplacianEigenmaps(dimensions=34).fit(graph)


def test_estimate_is_deterministic_across_calls():
    for name in ("karate", "grid"):
        first, _ = build_graph(name)
        second, _ = build_graph(name)
        a = mk.FGSD(hist_bins=16, hist_range=5)
        a.fit([first])
        b = mk.FGSD(hist_bins=16, hist_range=5)
        b.fit([second])
        assert a.get_embedding().dtype == np.float64
        assert np.array_equal(a.get_embedding(), b.get_embedding())


def test_embedding_shapes_follow_the_constructors():
    graph, _ = build_graph("karate")
    shapes = {}
    for name, model in (
        ("ldp", mk.LDP(bins=32)),
        ("fgsd", mk.FGSD(hist_bins=64)),
        ("sf", mk.SF(dimensions=16)),
        ("netlsd", mk.NetLSD(scale_steps=32, approximations=4)),
    ):
        model.fit([graph])
        shapes[name] = model.get_embedding().shape
    assert shapes == {
        "ldp": (1, 160),
        "fgsd": (1, 64),
        "sf": (1, 16),
        "netlsd": (1, 32),
    }
    eigenmaps = mk.LaplacianEigenmaps(dimensions=8)
    eigenmaps.fit(graph)
    assert eigenmaps.get_embedding().shape == (34, 8)


def test_graph2vec_runs_end_to_end():
    """The one estimator whose second half is not ported: gensim's Doc2Vec."""
    graphs = [build_graph(name)[0] for name in ("karate", "grid", "lollipop")]
    model = mk.Graph2Vec(
        wl_iterations=2, dimensions=8, workers=1, epochs=2, min_count=1, seed=42
    )
    model.fit(graphs)
    embedding = model.get_embedding()
    assert embedding.shape == (3, 8)
    assert np.isfinite(embedding).all()
    assert len(model.model.dv) == 3


def test_graph2vec_documents_are_the_wl_graph_features():
    from mojokarateclub.utils import WeisfeilerLehmanHashing

    graph, _ = build_graph("karate")
    looped = mk.Estimator()._check_graph(build_graph("karate")[0])
    reference = WeisfeilerLehmanHashing(looped, 2, False, False).get_graph_features()
    model = mk.Graph2Vec(
        wl_iterations=2, dimensions=8, workers=1, epochs=1, min_count=1, seed=42
    )
    model.fit([graph])
    vocabulary = set(model.model.wv.key_to_index)
    missing = [word for word in reference if word not in vocabulary]
    assert missing == []
    assert len(vocabulary) == len(set(reference))
