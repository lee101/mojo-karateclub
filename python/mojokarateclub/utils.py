"""Random walks and Weisfeiler-Lehman features, mirroring `karateclub.utils`.

The walk and the WL recursion now live in Mojo, so this module is the argument
marshalling, the scratch allocation and the one place a kernel's flat answer
becomes the lists of Python strings `gensim` and the callers expect. The graph
arrives as a networkx graph, upstream's own convention, and is converted to the
CSR the kernels take; no self-loop is added or removed here, because
`Estimator._check_graph` has already done that where the model wants it.
"""

import operator
import random
from typing import List

import networkx as nx
import numpy as np

from ._ffi import _bytes, _csr, _data, _index, _lib, _mt_state, _words, _zeros
from .estimator import csr_from_graph

__all__ = [
    "RandomWalker",
    "BiasedRandomWalker",
    "EulerianDiffuser",
    "WeisfeilerLehmanHashing",
    "check_value",
    "mt_init_genrand",
    "mt_genrand_uint32",
]


def _tour_capacity(diffusion_cover: int) -> int:
    """Longest Eulerian tour the growth can produce.

    The growth adds two directed edges per infected node past the source, and
    the tour has one id per edge: the search emits `(last_vertex,
    current_vertex)` on every pop but the first, and there is one more pop
    than push.
    """
    return max(2 * diffusion_cover - 2, 1)


# `hashlib` scratch: `m` is the 64-word message schedule, `state` the four
# chaining words, `tail` the under-64-byte remainder buffer.
_MD5_WORDS = 64
_MD5_STATE = 4
_MD5_TAIL = 128
# A digest is four words and a feature slot is a digest.
_WL_SLOT_WORDS = 4
# `hexdigest()` is 32 lowercase hex chars over 16 digest bytes.
_WL_HEX_CHARS = 32
# A digest slot is four little-endian words, hexlified two chars per byte.
_WL_SLOT_BYTES = _WL_SLOT_WORDS * 4


def mt_init_genrand(state, seed):
    """`init_genrand` from mt19937ar.c, over a `UInt32[625]` state.

    Word `i` is `mt[i]` for `i < 624` and the draw index for `i == 624`, so
    mirroring the stream in NumPy is a transcription of this state layout.
    """
    _lib.kc_mt_init_genrand(_words(state, "state"), seed & 0xFFFFFFFF)


def mt_genrand_uint32(state) -> int:
    """One tempered 32-bit word, advancing the state in place."""
    return _lib.kc_mt_genrand_uint32(_words(state, "state")) & 0xFFFFFFFF


def check_value(value, name):
    """`_check_value`: upstream tries `1 / value` and re-raises `ValueError`."""
    try:
        _ = 1 / value

    except ZeroDivisionError:
        raise ValueError(
            f"The value of {name} is too small " f"or zero to be used in 1/{name}."
        )


def _node_bound(n, node):
    """Reject a source node the CSR cannot index; upstream raises, we would read."""
    node = operator.index(node)
    if not 0 <= node < n:
        raise ValueError(f"node must be in [0, {n}), got {node}")
    return node

def _split_walks(walk, walk_length, degrees, count, written):
    """The `count` walks the kernel wrote back to back, as lists of node ids.

    A walk stops as soon as it stands on a node with no neighbours, which the
    flat buffer does not say on its own, so the split is recovered by replaying
    that rule over the CSR row lengths. `written` is the id count the kernel
    reported, and the two have to agree or the buffer is not the walks.
    """
    nodes = walk.tolist()
    at = 0
    walks = []
    for _ in range(count):
        current = nodes[at]
        at += 1
        one = [current]
        for _ in range(walk_length - 1):
            if degrees[current] == 0:
                break
            current = nodes[at]
            at += 1
            one.append(current)
        walks.append(one)
    if at != written:
        # The kernel counts the ids it wrote; if the replay above consumed a
        # different number, the walk buffer does not hold the walks the
        # kernel claims and slicing it would silently return the wrong ids.
        raise RuntimeError(
            f"walker wrote {written} ids but the split consumed {at}"
        )
    return walks


class RandomWalker:
    """
    Class to do fast first-order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """

    def __init__(self, walk_length: int, walk_number: int):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        node = _node_bound(int(self._csr[0].size) - 1, node)
        walk = np.zeros(self.walk_length, dtype=np.int32)
        written = _lib.kc_random_walker_do_walk(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            node,
            self.walk_length,
            _index(walk, "walk"),
            _words(self.state, "state"),
        )
        walk = [str(w) for w in walk[:written]]
        return walk

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        self._csr = _csr(*csr_from_graph(graph))
        n = int(self._csr[0].size) - 1
        degrees = np.diff(self._csr[0].astype(np.int64))
        # Upstream draws from the global `random` module here, so the MT19937
        # stream is reseeded from it: `Estimator._set_seed` still makes a fit
        # reproducible, which is the one property of the upstream walk the
        # kernel cannot inherit.
        self.state = _mt_state()
        mt_init_genrand(self.state, random.getrandbits(32))
        walk = np.zeros(n * self.walk_number * self.walk_length, dtype=np.int32)
        total = _lib.kc_random_walker_do_walks(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            n,
            self.walk_length,
            self.walk_number,
            _index(walk, "walk"),
            _words(self.state, "state"),
        )
        self.walks = [
            [str(w) for w in one]
            for one in _split_walks(
                walk, self.walk_length, degrees, n * self.walk_number, total
            )
        ]


class BiasedRandomWalker:
    """
    Class to do biased second order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
        p (float): Return parameter (1/p transition probability) to move towards previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

    walks: list
    graph: nx.classes.graph.Graph
    edge_fn: object
    weight_fn: object

    def __init__(self, walk_length: int, walk_number: int, p: float, q: float):
        self.walk_length = walk_length
        self.walk_number = walk_number

        check_value(p, "p")
        self.p = p

        check_value(q, "q")
        self.q = q

    def do_walk(self, node: int) -> List[str]:
        """
        Doing a single truncated second order random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        node = _node_bound(int(self._csr[0].size) - 1, node)
        walk = np.zeros(self.walk_length, dtype=np.int32)
        written = _lib.kc_biased_random_walker_do_walk(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            _data(self._csr[2], "values"),
            int(self.weighted),
            node,
            self.walk_length,
            float(self.p),
            float(self.q),
            _index(walk, "walk"),
            _words(self.state, "state"),
            _data(self.weights, "weights"),
            _index(self.mark, "mark"),
        )
        walk = [str(w) for w in walk[:written]]
        return walk

    def do_walks(self, graph) -> None:
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        # `_get_weight_fn`'s dispatch, and `_get_edge_fn` is the CSR row.
        self.weighted = int(nx.classes.function.is_weighted(graph))
        self._csr = _csr(*csr_from_graph(graph, self.weighted))
        n = int(self._csr[0].size) - 1
        degrees = np.diff(self._csr[0].astype(np.int64))
        # `weights` is the widest row and `mark` is `n`, both zero on entry; the
        # kernel restores `mark` to all zero, so the walks chain without a
        # clear between them.
        self.weights = _zeros(int(degrees.max()) if n else 1)
        self.mark = np.zeros(n, dtype=np.int32)
        # `np.random.choice` upstream, so the seed comes from `np.random` here
        # and `Estimator._set_seed` reaches this walk the same way.
        self.state = _mt_state()
        mt_init_genrand(self.state, int(np.random.randint(1 << 32, dtype=np.uint64)))
        walk = np.zeros(n * self.walk_number * self.walk_length, dtype=np.int32)
        total = _lib.kc_biased_random_walker_do_walks(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            _data(self._csr[2], "values"),
            int(self.weighted),
            n,
            self.walk_length,
            self.walk_number,
            float(self.p),
            float(self.q),
            _index(walk, "walk"),
            _words(self.state, "state"),
            _data(self.weights, "weights"),
            _index(self.mark, "mark"),
        )
        self.walks = [
            [str(w) for w in one]
            for one in _split_walks(
                walk, self.walk_length, degrees, n * self.walk_number, total
            )
        ]


class EulerianDiffuser:
    """
    Class to make diffusions for a given graph.

    Args:
        diffusion_number (int): Number of diffusions
        diffusion_cover (int): Number of nodes in diffusion.

    The growth and the Eulerian tour are one Mojo kernel, so the subgraph the
    upstream `nx.DiGraph` holds is the caller's scratch here. The draws come
    from this module's MT19937 rather than CPython's `_random`, so the tours
    are not upstream's for a given seed; what they share is the shape, which is
    what `_run_diffusion_process` documents.
    """

    def __init__(self, diffusion_number: int, diffusion_cover: int):
        self.diffusion_number = diffusion_number
        self.diffusion_cover = diffusion_cover

    def _scratch(self, n, diffusions):
        """Subgraph, stack, tour and output buffers, in kernel argument order.

        `infected` holds the source plus one id per infected neighbour, the
        `n`-sized ones are the subgraph indexed by node with `marked` holding
        the generation each node was infected in, and `next_edge`, `edge_dst`
        and `stack` are sized for the widest subgraph the cover allows.
        """
        cover = max(int(self.diffusion_cover), 1)
        tour = _tour_capacity(cover)
        edges = 2 * (cover - 1)
        return (
            np.zeros(cover, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.zeros(edges, dtype=np.int32),
            np.zeros(edges, dtype=np.int32),
            np.zeros(edges + 1, dtype=np.int32),
            np.zeros(tour * diffusions + 1, dtype=np.int32),
            np.zeros(tour * diffusions, dtype=np.int32),
        )

    def _run_diffusion_process(self, node):
        """
        Generating a diffusion tree from a given source node and linearizing it
        with a directed Eulerian tour.

        Arg types:
            * **node** *(int)* - The source node of the diffusion.
        Return types:
            * **euler** *(list of strings)* - The list of nodes in the walk.
        """
        n = int(self._csr[0].size) - 1
        (
            infected,
            marked,
            head,
            tail,
            degree,
            stamp,
            next_edge,
            edge_dst,
            stack,
            offsets,
            flat,
        ) = self._scratch(n, 1)
        node = _node_bound(int(self._csr[0].size) - 1, node)
        written = _lib.kc_eulerian_diffuser_run_diffusion_process(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            node,
            self.diffusion_cover,
            1,
            _index(infected, "infected"),
            _index(marked, "marked"),
            _index(head, "head"),
            _index(tail, "tail"),
            _index(degree, "degree"),
            _index(stamp, "stamp"),
            _index(next_edge, "next_edge"),
            _index(edge_dst, "edge_dst"),
            _index(stack, "stack"),
            _index(flat, "circuit"),
            _words(self.state, "state"),
        )
        euler = [str(w) for w in flat[:written]]
        return euler

    def do_diffusions(self, graph):
        """
        Running diffusions from every node.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run diffusions on.
        """
        self.graph = graph
        self._csr = _csr(*csr_from_graph(graph))
        n = int(self._csr[0].size) - 1
        degrees = np.diff(self._csr[0].astype(np.int64))
        if n and int((degrees == 0).any()):
            # `random.choice(nebs)` on a node with no neighbours; `_ensure_integrity`
            # rules this out for every estimator, which is why the kernel cannot
            # raise it and this does.
            raise IndexError("Cannot choose from an empty sequence")
        self.state = _mt_state()
        mt_init_genrand(self.state, random.getrandbits(32))
        diffusions = n * self.diffusion_number
        (
            infected,
            marked,
            head,
            tail,
            degree,
            stamp,
            next_edge,
            edge_dst,
            stack,
            offsets,
            flat,
        ) = self._scratch(n, diffusions)
        total = _lib.kc_eulerian_diffuser_do_diffusions(
            _index(self._csr[0], "indptr"),
            _index(self._csr[1], "indices"),
            n,
            self.diffusion_number,
            self.diffusion_cover,
            _index(flat, "diffusions"),
            _index(offsets, "offsets"),
            _words(self.state, "state"),
            _index(infected, "infected"),
            _index(marked, "marked"),
            _index(head, "head"),
            _index(tail, "tail"),
            _index(degree, "degree"),
            _index(stamp, "stamp"),
            _index(next_edge, "next_edge"),
            _index(edge_dst, "edge_dst"),
            _index(stack, "stack"),
        )
        # The buffer is capacity-sized, so only the prefix the kernel filled is
        # an offsets table; the rest is still zero.
        table = offsets[: diffusions + 1].astype(np.int64)
        if int(table[-1]) != int(total) or np.any(np.diff(table) < 0):
            # hand back the wrong ids instead of failing.
            raise RuntimeError(
                f"diffuser wrote {int(total)} ids but its offsets end at "
                f"{int(offsets[diffusions])}"
            )
        self.diffusions = [
            [str(w) for w in flat[offsets[i] : offsets[i + 1]]]
            for i in range(diffusions)
        ]


def _base_labels(graph, nodes, attributed):
    """`{node: [str(v)]}`, upstream's `_set_features`, as bytes for the kernel.

    A label cannot cross the boundary as a Python string, so the `n` base
    labels go over as one `UInt8` buffer of concatenated UTF-8 plus an
    `Int32[n+1]` offset table. The lengths handed back are byte lengths, which
    upper-bound the character counts the Mojo helper bounds its staging buffer
    with.
    """
    if attributed:
        features = nx.get_node_attributes(graph, "feature")
    else:
        features = {node: graph.degree(node) for node in graph.nodes()}
    labels = [str(features[node]) for node in nodes]
    encoded = [label.encode("utf-8") for label in labels]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int32)
    np.cumsum([len(one) for one in encoded], out=offsets[1:])
    base = np.zeros(int(offsets[-1]), dtype=np.uint8)
    at = 0
    for one in encoded:
        base[at : at + len(one)] = np.frombuffer(one, dtype=np.uint8)
        at += len(one)
    return labels, base, offsets, max((len(one) for one in encoded), default=0)


def _wl_max_message_bytes(max_degree: int, longest_base_label: int) -> int:
    """Mirrors `treefeatures.wl_max_message_bytes`, which is not exported.

    A message is the node's own label, then one `"_"` and one neighbour label
    per neighbour, and every label from the first recursion on is a 32-character
    digest, so the bound never drops below that.
    """
    chars = max(longest_base_label, _WL_HEX_CHARS)
    return (max_degree + 1) * chars + max_degree


class WeisfeilerLehmanHashing:
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        attributed: bool,
        erase_base_features: bool,
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.graph = graph
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.indptr, self.indices = _csr(*csr_from_graph(graph))[:2]
        self.nodes = list(graph.nodes())
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        """
        Creating the features.
        """
        self._labels, self._base, self._base_offsets, longest = _base_labels(
            self.graph, self.nodes, self.attributed
        )
        self._max_degree = (
            int(np.diff(self.indptr.astype(np.int64)).max()) if self.nodes else 0
        )
        self._longest_base_label = longest

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        n = len(self.nodes)
        stride = self.wl_iterations + 1
        extracted = np.zeros(n * stride * _WL_SLOT_WORDS, dtype=np.uint32)
        msg = np.zeros(
            _wl_max_message_bytes(self._max_degree, self._longest_base_label),
            dtype=np.uint8,
        )
        order = np.zeros(max(self._max_degree, 1), dtype=np.int32)
        order_tmp = np.zeros(max(self._max_degree, 1), dtype=np.int32)
        m = np.zeros(_MD5_WORDS, dtype=np.uint32)
        state = np.zeros(_MD5_STATE, dtype=np.uint32)
        tail = np.zeros(_MD5_TAIL, dtype=np.uint8)
        _lib.kc_wl_hash(
            n,
            _index(self.indptr, "indptr"),
            _index(self.indices, "indices"),
            _bytes(self._base, "base_bytes"),
            _index(self._base_offsets, "base_offsets"),
            self.wl_iterations,
            int(self.erase_base_features),
            _words(extracted, "extracted"),
            _bytes(msg, "msg"),
            _index(order, "order"),
            _index(order_tmp, "order_tmp"),
            _words(m, "m"),
            _words(state, "state"),
            _bytes(tail, "tail"),
        )
        # The kernel already shifted each run down by one slot when
        # `erase_base_features` is set, so the read stride is the erased one and
        # the base label is simply not read back.
        effective = stride - 1 if self.erase_base_features else stride
        # One little-endian copy of the whole digest block, so a slot is a
        # 16-byte slice of it rather than four numpy scalar reads and a join.
        blob = extracted.astype("<u4", copy=False).tobytes()
        self.extracted_features = {
            node: self._run(blob, stride, effective, index)
            for index, node in enumerate(self.nodes)
        }

    def _run(self, blob, stride, effective, index):
        """One node's run of slots: the digests, and the base label if kept.

        A digest slot is four words; `hexdigest` writes each digest byte as two
        hex chars in that same order, so the little-endian bytes of a slot
        hexlified are exactly that digest's `hexdigest`.
        """
        erased = stride != effective
        run = [] if erased else [self._labels[index]]
        slot = index * effective + (0 if erased else 1)
        for _ in range(self.wl_iterations):
            at = slot * _WL_SLOT_BYTES
            run.append(blob[at : at + _WL_SLOT_BYTES].hex())
            slot += 1
        return run

    def get_node_features(self):
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]
