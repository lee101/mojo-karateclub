"""Parity and contract tests for the node-level estimators and the walkers.

Everything asserted against `tests/vectors/upstream.npz` is real karateclub
1.3.3 output. The stochastic models divide into two groups: `NodeSketch` and
both `LINE` orders reproduce numpy's legacy MT19937, so their embeddings are
bit-comparable and are compared directly; the walk-based models cannot be,
because the walk draws come from this repo's own MT19937 stream and the
skip-gram training is stochastic, so they are compared on the invariants that
do hold.
"""

import hashlib
import random

import networkx as nx
import numpy as np
import pytest

import mojokarateclub as mk
from _graphs import build_graph

SMALL = ["karate", "gnp60", "lollipop", "isolated", "grid"]
SEEDED = ["karate", "gnp200"]

# The visit-count check is a max over hundreds of standard normals, where the
# expected maximum is about 3.3; 5 is far enough out to be stable run to run
# and far enough in that a wrong transition law cannot reach it.
Z_LIMIT = 5.0

# ------------------------------------------------------------------ hashing


@pytest.mark.parametrize(
    "message",
    [b"", b"a", b"abc", b"message digest", b"x" * 55, b"x" * 56, b"x" * 120, bytes(range(256))],
)
def test_md5_matches_hashlib(message):
    assert mk.md5_digest(message) == hashlib.md5(message).digest()


@pytest.mark.parametrize("seed", [42, 7])
def test_mt19937_double_stream_matches_numpy(seed):
    """The walkers draw from this generator, and numpy's legacy `rand` is it."""
    state = mk.utils._mt_state()
    mk.utils.mt_init_genrand(state, seed)
    doubles = []
    for _ in range(64):
        a = mk.utils.mt_genrand_uint32(state) >> 5
        b = mk.utils.mt_genrand_uint32(state) >> 6
        doubles.append((a * 67108864.0 + b) / 9007199254740992.0)
    assert np.array_equal(np.array(doubles), np.random.RandomState(seed).rand(64))


def test_mt19937_state_advances():
    state = mk.utils._mt_state()
    mk.utils.mt_init_genrand(state, 42)
    first = [mk.utils.mt_genrand_uint32(state) for _ in range(4)]
    second = [mk.utils.mt_genrand_uint32(state) for _ in range(4)]
    assert first != second


def test_mt19937_res53_matches_numpy():
    from mojokarateclub._ffi import _lib, _mt_state, _words

    state = _mt_state()
    _lib.kc_mt_init_genrand(_words(state, "state"), 42)
    ours = np.array(
        [_lib.kc_mt_genrand_res53(_words(state, "state")) for _ in range(64)]
    )
    assert np.array_equal(ours, np.random.RandomState(42).rand(64))


# ----------------------------------------------------------------- walkers


def test_random_walker_shape_matches_upstream(upstream_vectors):
    for name in SEEDED:
        graph, n = build_graph(name)
        _seed_everything()
        walker = mk.RandomWalker(walk_length=20, walk_number=5)
        walker.do_walks(graph)
        walks = walker.walks
        assert len(walks) == int(upstream_vectors[f"walk/count/{name}"][0])
        assert sorted({len(w) for w in walks}) == sorted(
            {int(v) for v in upstream_vectors[f"walk/length/{name}"]}
        )
        assert n == graph.number_of_nodes()


def test_walks_only_step_along_real_edges():
    graph, _ = build_graph("karate", integrity=True)
    for walker in (
        mk.RandomWalker(walk_length=25, walk_number=3),
        mk.BiasedRandomWalker(walk_length=25, walk_number=3, p=0.5, q=2.0),
    ):
        _seed_everything(1)
        walker.do_walks(graph)
        for walk in walker.walks:
            assert len(walk) == 25
            for left, right in zip(walk, walk[1:]):
                assert graph.has_edge(int(left), int(right)), (left, right)


def test_a_single_walk_starts_at_its_source_and_steps_along_real_edges():
    graph, _ = build_graph("karate", integrity=True)
    for walker in (
        mk.RandomWalker(walk_length=25, walk_number=3),
        mk.BiasedRandomWalker(walk_length=25, walk_number=3, p=0.5, q=2.0),
    ):
        _seed_everything(1)
        walker.do_walks(graph)
        for source in (0, 7, 33):
            walk = walker.do_walk(source)
            assert len(walk) == 25
            assert int(walk[0]) == source
            for left, right in zip(walk, walk[1:]):
                assert graph.has_edge(int(left), int(right)), (left, right)


@pytest.mark.parametrize("node", [-1, 34, 10**9])
def test_a_single_walk_rejects_a_node_the_graph_does_not_have(node):
    # The kernel indexes the CSR with the source directly, so an unchecked
    # node reads outside the buffers where upstream raises.
    graph, _ = build_graph("karate", integrity=True)
    for walker in (
        mk.RandomWalker(walk_length=10, walk_number=2),
        mk.BiasedRandomWalker(walk_length=10, walk_number=2, p=0.5, q=2.0),
    ):
        _seed_everything(1)
        walker.do_walks(graph)
        with pytest.raises(ValueError, match="node"):
            walker.do_walk(node)


def test_walk_visit_counts_match_the_upstream_transition_law(upstream_vectors):
    """The draws differ from upstream's, so compare the distribution they follow.

    Both sides take `n * walk_number * walk_length` steps under a transition
    law that is a property of the graph, so each node's visit share is a
    multinomial draw around the same mean. Upstream's sample is small - 3400
    steps on the karate club - so the test is a z-score per node against
    upstream's own standard error rather than a fixed tolerance on the share.
    """
    worst = 0.0
    for name in SEEDED:
        for kind, factory in (
            ("walk", lambda: mk.RandomWalker(walk_length=40, walk_number=60)),
            (
                "biased",
                lambda: mk.BiasedRandomWalker(walk_length=40, walk_number=60, p=0.5, q=2.0),
            ),
        ):
            graph, n = build_graph(name)
            _seed_everything()
            walker = factory()
            walker.do_walks(graph)
            counts = np.zeros(n, dtype=np.int64)
            for walk in walker.walks:
                for token in walk:
                    counts[int(token)] += 1
            expected = upstream_vectors[f"{kind}/histogram/{name}"]
            assert counts.sum() == n * 60 * 40
            share = counts / counts.sum()
            reference = expected / expected.sum()
            probability = np.clip(reference, 1e-12, 1.0 - 1e-12)
            # Both sides are multinomial draws around the same mean, so the
            # standard error of the difference carries a term for each sample.
            error = np.sqrt(
                probability
                * (1.0 - probability)
                * (1.0 / expected.sum() + 1.0 / counts.sum())
            )
            z = float((np.abs(share - reference) / error).max())
            worst = max(worst, z)
            assert z < Z_LIMIT, (name, kind, z)
    assert worst < Z_LIMIT


def _seed_everything(value=42):
    random.seed(value)
    np.random.seed(value)


def test_walkers_are_reproducible_after_set_seed():
    """Each walker seeds from the global stream upstream draws from.

    `RandomWalker` uses `random.sample`, so its seed comes from `random`;
    `BiasedRandomWalker` uses `np.random.choice`, so its seed comes from numpy.
    `Estimator._set_seed` sets both, which is what makes `fit` reproducible.
    """
    for factory in (
        lambda: mk.RandomWalker(walk_length=12, walk_number=4),
        lambda: mk.BiasedRandomWalker(walk_length=12, walk_number=4, p=0.5, q=2.0),
    ):
        runs = []
        for _ in range(2):
            _seed_everything()
            walker = factory()
            walker.do_walks(build_graph("karate")[0])
            runs.append(walker.walks)
        assert runs[0] == runs[1], factory


def test_biased_walker_rejects_a_zero_parameter():
    with pytest.raises(ValueError):
        mk.BiasedRandomWalker(walk_length=5, walk_number=1, p=0.0, q=1.0)


def test_isolated_nodes_produce_a_one_token_walk():
    graph, _ = build_graph("isolated", integrity=False)
    _seed_everything(3)
    walker = mk.RandomWalker(walk_length=10, walk_number=1)
    walker.do_walks(graph)
    walks = walker.walks
    short = [w for w in walks if len(w) == 1]
    assert len(short) == 4
    assert {int(w[0]) for w in short} == {6, 7, 8, 9}


# ---------------------------------------------------------------- NodeSketch


def test_nodesketch_matches_upstream(upstream_vectors):
    for name in SMALL:
        graph, _ = build_graph(name)
        model = mk.NodeSketch(dimensions=8, iterations=2, decay=0.01, seed=42)
        model.fit(graph)
        expected = upstream_vectors[f"nodesketch/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.array_equal(got, expected), name


def test_nodesketch_embedding_is_a_node_index_per_column():
    graph, n = build_graph("karate")
    model = mk.NodeSketch(dimensions=6, iterations=3, seed=1)
    model.fit(graph)
    embedding = model.get_embedding()
    assert embedding.shape == (n, 6)
    assert np.array_equal(embedding, embedding.astype(np.int64))


# --------------------------------------------------------------------- LINE


def test_first_order_line_matches_upstream(upstream_vectors):
    for name in SEEDED:
        graph, _ = build_graph(name)
        model = mk.FirstOrderLINE(
            dimensions=8, epochs=3, mini_batch_size=32, verbose=False, seed=42
        )
        model.fit(graph)
        expected = upstream_vectors[f"line1/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.abs(got - expected).max() < 1e-12, (
            name,
            float(np.abs(got - expected).max()),
        )


def test_second_order_line_matches_upstream(upstream_vectors):
    for name in SEEDED:
        graph, _ = build_graph(name)
        model = mk.SecondOrderLINE(
            dimensions=8, epochs=3, mini_batch_size=32, verbose=False, seed=42
        )
        model.fit(graph)
        expected = upstream_vectors[f"line2/{name}"]
        got = model.get_embedding()
        assert got.shape == expected.shape, name
        assert np.abs(got - expected).max() < 1e-12, (
            name,
            float(np.abs(got - expected).max()),
        )


# ------------------------------------------- skip-gram models (not exact)


def _embed(factory, name="karate"):
    graph, n = build_graph(name)
    model = factory()
    model.fit(graph)
    return model.get_embedding(), n


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mk.DeepWalk(
            walk_number=4, walk_length=20, dimensions=8, workers=1, seed=42
        ),
        lambda: mk.Node2Vec(
            walk_number=4, walk_length=20, dimensions=8, workers=1, seed=42
        ),
        lambda: mk.Walklets(
            walk_number=4,
            walk_length=20,
            dimensions=4,
            window_size=2,
            workers=1,
            seed=42,
        ),
    ],
)
def test_skipgram_models_cover_every_node(factory):
    embedding, n = _embed(factory)
    assert embedding.shape[0] == n
    assert np.isfinite(embedding).all()


def test_walklets_concatenates_one_table_per_window():
    embedding, n = _embed(
        lambda: mk.Walklets(
            walk_number=4,
            walk_length=20,
            dimensions=4,
            window_size=3,
            workers=1,
            seed=42,
        )
    )
    assert embedding.shape == (n, 12)


# ---------------------------------------------------------------- diffuser


def _ref_sample(sequence, state):
    """`random.sample(seq, 1)[0]` and `random.choice(seq)` on the port's stream.

    Both are `seq[randbelow(len(seq))]`, and `randbelow` is the
    `mt19937ar` rejection loop `walker.randbelow` implements, so driving it
    from `mt_genrand_uint32` consumes the generator word for word exactly as
    the kernel does.
    """
    size = len(sequence)
    if size <= 0:
        raise IndexError("Cannot choose from an empty sequence")
    bits = size.bit_length()
    if bits % 8 == 0:
        bits += 8
    while True:
        draw = mk.utils.mt_genrand_uint32(state) >> (32 - bits)
        if draw < size:
            return sequence[draw]


def _ref_diffusion(graph, node, cover, state):
    """`EulerianDiffuser._run_diffusion_process`, verbatim but for the draws
    and for the boundary exit the port adds.

    The subgraph and the tour are upstream's: a real `nx.DiGraph` and a real
    `networkx.eulerian_circuit`, so the comparison is against the algorithm
    this port claims to reproduce and not against itself. Neighbours are read
    sorted because the kernel reads a CSR row, which `csr_from_graph` sorts.

    `boundary` is the count of edges leaving the infected set, which is what
    the kernel maintains; it is 0 exactly when no further infection is
    possible, which is where upstream loops forever.
    """
    infected = [node]
    sub_graph = nx.DiGraph()
    sub_graph.add_node(node)
    infected_counter = 1
    marked = {node}
    boundary = sum(1 for u in graph.neighbors(node) if u != node)
    while infected_counter < cover and boundary > 0:
        end_point = _ref_sample(infected, state)
        sample = _ref_sample(sorted(graph.neighbors(end_point)), state)
        if sample not in infected:
            infected_counter = infected_counter + 1
            infected = infected + [sample]
            sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
            marked.add(sample)
            outside = sum(
                1 for u in graph.neighbors(sample) if u != sample and u not in marked
            )
            inside = sum(
                1 for u in graph.neighbors(sample) if u != sample and u in marked
            )
            boundary += outside - inside
            if infected_counter == cover:
                break
    euler = [str(u) for u, v in nx.eulerian_circuit(sub_graph, infected[0])]
    return euler


def _ref_diffusions(graph, number, cover, seed_value):
    state = mk.utils._mt_state()
    mk.utils.mt_init_genrand(state, seed_value)
    return [
        _ref_diffusion(graph, node, cover, state)
        for node in range(graph.number_of_nodes())
        for _ in range(number)
    ]


def _seeded_diffusions(graph, number, cover, seed=42):
    """`do_diffusions` plus the seed it drew, which the reference replays."""
    random.seed(seed)
    seed_value = random.getrandbits(32)
    random.seed(seed)
    diffuser = mk.EulerianDiffuser(number, cover)
    diffuser.do_diffusions(graph)
    return diffuser, seed_value


@pytest.mark.parametrize("name", ["karate", "gnp60", "lollipop", "grid"])
@pytest.mark.parametrize("cover", [2, 5, 12])
def test_diffusions_match_a_networkx_reference(name, cover):
    """The kernel's growth and its tour against `networkx.eulerian_circuit`.

    Only the draws differ from upstream, so the reference is upstream's own
    algorithm replayed on the same generator: the tours must be identical,
    node for node.
    """
    graph, _ = build_graph(name, integrity=True)
    diffuser, seed_value = _seeded_diffusions(graph, 2, cover)
    assert diffuser.diffusions == _ref_diffusions(graph, 2, cover, seed_value)


def test_run_diffusion_process_matches_the_reference():
    graph, _ = build_graph("karate", integrity=True)
    diffuser, seed_value = _seeded_diffusions(graph, 2, 9)
    # `_run_diffusion_process` draws from the stream `do_diffusions` left, so
    # rewind it to the start and replay one source.
    mk.utils.mt_init_genrand(diffuser.state, seed_value)
    state = mk.utils._mt_state()
    mk.utils.mt_init_genrand(state, seed_value)
    for node in (0, 7, 33):
        assert diffuser._run_diffusion_process(node) == _ref_diffusion(
            graph, node, 9, state
        )


def test_every_tour_is_an_eulerian_circuit_of_its_own_subgraph():
    """Each tour uses every edge of the infected subgraph exactly once.

    The subgraph grows outward from the source, so the circuit closes: the
    consecutive pairs plus the pair from the last id back to the source are
    the subgraph's edges, each of them once. The subgraph is a `DiGraph` and
    every edge was added in both directions, so an unordered edge is walked
    twice, once each way, and only the directed pair is unique.
    """
    graph, _ = build_graph("karate", integrity=True)
    diffuser, _ = _seeded_diffusions(graph, 2, 10)
    non_empty = 0
    for index, tour in enumerate(diffuser.diffusions):
        nodes = [int(token) for token in tour]
        source = index // 2
        assert nodes[0] == source, index
        pairs = list(zip(nodes, nodes[1:] + nodes[:1]))
        assert len(pairs) == len(set(pairs)), (index, nodes)
        for pair in pairs:
            assert graph.has_edge(*pair), (index, pair)
        undirected = {frozenset(pair) for pair in pairs}
        assert len(undirected) * 2 == len(pairs), (index, nodes)
        non_empty += bool(pairs)
    assert non_empty > 0.9 * len(diffuser.diffusions)


def test_a_tour_has_one_id_per_edge_of_the_cover():
    """Every tour is two ids per infected node past the source, bounded by the
    cover, and on a connected fixture the cover is reached essentially always.

    A tour can come back short: the growth walks the infected set looking for a
    node outside it, and a run of draws that land back inside ends it early.
    That is the port's frontier-miss exit, and on a self-looped graph it fires
    now and then at a small cover.
    """
    graph, _ = build_graph("karate", integrity=True)
    for cover in (1, 2, 7, 20):
        diffuser, _ = _seeded_diffusions(graph, 1, cover)
        lengths = [len(tour) for tour in diffuser.diffusions]
        assert all(length % 2 == 0 for length in lengths), (cover, lengths)
        assert all(length <= 2 * max(cover - 1, 0) for length in lengths), cover
        if cover >= 7:
            reached = sum(1 for length in lengths if length == 2 * (cover - 1))
            assert reached >= 0.9 * len(lengths), (cover, reached)


def test_diffusions_are_reproducible_after_set_seed():
    graph, _ = build_graph("karate", integrity=True)
    runs = []
    for _ in range(2):
        runs.append(_seeded_diffusions(graph, 2, 8)[0].diffusions)
    assert runs[0] == runs[1]


def test_diffuser_rejects_a_node_without_neighbours():
    """`random.choice([])` on a node with no edges, which the kernel cannot raise."""
    graph, _ = build_graph("isolated", integrity=False)
    with pytest.raises(IndexError):
        mk.EulerianDiffuser(1, 4).do_diffusions(graph)


def test_diff2vec_covers_every_node():
    graph, n = build_graph("karate")
    model = mk.Diff2Vec(
        diffusion_number=2, diffusion_cover=8, dimensions=8, workers=1, seed=42
    )
    model.fit(graph)
    embedding = model.get_embedding()
    assert embedding.shape == (n, 8)
    assert np.isfinite(embedding).all()
