"""Node-level estimators, mirroring `karateclub.node_embedding.neighbourhood`.

Each class keeps upstream's name, constructor argument order and defaults and
its `fit` / `get_embedding` surface. What is left here is the argument
marshalling and the scratch allocation; the walks, the sketching, the LINE
updates and the eigen decomposition all happen in the Mojo kernel named in each
`fit`, except where upstream also delegates to gensim and so does this:
`DeepWalk`, `Node2Vec` and `Walklets` train a `Word2Vec` exactly as upstream
does.

`Estimator._check_graph` has already added a self loop to every node by the
time a kernel runs, which is upstream's `_ensure_integrity` and is part of the
contract several of these kernels depend on. Nothing is bit-comparable with
upstream except `NodeSketch` and the two `LINE` models, which reproduce
numpy's legacy MT19937; the walks do not, because upstream mixes CPython's
`_random` with numpy's `RandomState`.
"""

import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec

from ._ffi import _csr, _data, _index, _lib, _mt_state, _words, _zeros
from .estimator import Estimator, csr_from_graph
from .utils import BiasedRandomWalker, EulerianDiffuser, RandomWalker

__all__ = [
    "DeepWalk",
    "Node2Vec",
    "Walklets",
    "Diff2Vec",
    "NodeSketch",
    "FirstOrderLINE",
    "SecondOrderLINE",
    "LaplacianEigenmaps",
]

# `linalg.jacobi_eigh` returns its sweeps used, and a run that returns
# `sweeps` never reached `tol`. `tol` is a summed-squared-off-diagonal budget,
# not a per-element one: at `eigsh`'s `1e-12` the near-null eigenvector of
# Zachary's club is only about 2.3e-7 accurate, and at 1e-24 the eigenvalues
# agree with ARPACK to 4.7e-15.
EIGEN_SWEEPS = 400
EIGEN_TOL = 1e-24


def _kernel_csr(graph, weighted=False):
    """`graph` as the kernels take it: sorted, symmetric, self-looped CSR."""
    return _csr(*csr_from_graph(graph, weighted=weighted))[:3]


def _edges(graph):
    """`graph.edges(data=False)` as the `Int32[m, 2]` the LINE kernels read."""
    return np.ascontiguousarray(
        np.array(graph.edges(data=False), dtype=np.int32).reshape(-1, 2)
    )


class DeepWalk(Estimator):
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    The walks come from the Mojo kernel rather than `random.sample`, so the
    embedding is not bit-comparable with upstream's for a given seed; the
    `Word2Vec` that consumes them is upstream's call for call.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)

        model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(num_of_nodes)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)


class Node2Vec(Estimator):
    r"""An implementation of `"Node2Vec" <https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf>`_
    from the KDD '16 paper "node2vec: Scalable Feature Learning for Networks".
    The procedure uses biased second order random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    The walks come from the Mojo kernel rather than `np.random.choice`, so the
    embedding is not bit-comparable with upstream's for a given seed.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        p (float): Return parameter (1/p transition probability) to move towards from previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super(Node2Vec, self).__init__()

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        walker = BiasedRandomWalker(self.walk_length, self.walk_number, self.p, self.q)
        walker.do_walks(graph)

        model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        n_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(n_nodes)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)


class Walklets(Estimator):
    r"""An implementation of `"Walklets" <https://arxiv.org/abs/1605.02115>`_
    from the ASONAM '17 paper "Don't Walk, Skip! Online Learning of Multi-scale
    Network Embeddings". The procedure uses random walks to approximate the
    pointwise mutual information matrix obtained by individual normalized
    adjacency matrix powers. These are all decomposed by an approximate
    factorization technique and the embeddings are concatenated together.

    The walks come from the Mojo kernel rather than `random.sample`, so the
    embedding is not bit-comparable with upstream's for a given seed.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 32.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 4.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 32,
        workers: int = 4,
        window_size: int = 4,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def _select_walklets(self, walks, power):
        walklets = []
        for walk in walks:
            for step in range(power + 1):
                neighbors = [n for i, n in enumerate(walk[step:]) if i % power == 0]
                walklets.append(neighbors)
        return walklets

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Walklets model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)
        num_of_nodes = graph.number_of_nodes()

        self._embedding = []
        for power in range(1, self.window_size + 1):
            walklets = self._select_walklets(walker.walks, power)
            model = Word2Vec(
                walklets,
                hs=0,
                alpha=self.learning_rate,
                epochs=self.epochs,
                vector_size=self.dimensions,
                window=1,
                min_count=self.min_count,
                workers=self.workers,
                seed=self.seed,
            )

            embedding = np.array([model.wv[str(n)] for n in range(num_of_nodes)])
            self._embedding.append(embedding)

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.concatenate(self._embedding, axis=1)


class Diff2Vec(Estimator):
    r"""An implementation of `"Diff2Vec" <https://arxiv.org/abs/1703.03546>`_
    from the CompleNet '17 paper "Diff2Vec: Fast Sequence Based Embedding with
    Diffusion Graphs". The procedure creates diffusion trees from every source
    node in the graph. These graphs are linearized by a directed Eulerian tour,
    the walks are used for running the skip-gram algorithm the learn the
    embedding of the nodes.

    The diffusions come from the Mojo kernel, and draw from its MT19937 rather
    than CPython's `_random`, so the tours are not upstream's for a given
    seed. The `Word2Vec` that consumes them is upstream's call for call.

    Args:
        diffusion_number (int): Number of diffusions. Default is 10.
        diffusion_cover (int): Number of nodes in diffusion. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        diffusion_number: int = 10,
        diffusion_cover: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        self.diffusion_number = diffusion_number
        self.diffusion_cover = diffusion_cover
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Diff2Vec model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        diffuser = EulerianDiffuser(self.diffusion_number, self.diffusion_cover)
        diffuser.do_diffusions(graph)

        model = Word2Vec(
            diffuser.diffusions,
            hs=1,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(num_of_nodes)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)


class NodeSketch(Estimator):
    r"""An implementation of `"NodeSketch" <https://exascale.info/assets/pdf/yang2019nodesketch.pdf>`_
    from the KDD '19 paper "NodeSketch: Highly-Efficient Graph Embeddings
    via Recursive Sketching". The procedure  starts by sketching the self-loop-augmented
    adjacency matrix of the graph to output low-order node embeddings, and then recursively
    generates k-order node embeddings based on the self-loop-augmented adjacency matrix
    and (k-1)-order node embeddings.

    The kernel reproduces numpy's legacy MT19937 and the sketch arithmetic, so
    the embedding is bit-comparable with upstream's for a given seed.

    Args:
        dimensions (int): Embedding dimensions. Default is 32.
        iterations (int): Number of iterations (sketch order minus one). Default is 2.
        decay (float): Exponential decay rate. Default is 0.01.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 32,
        iterations: int = 2,
        decay: float = 0.01,
        seed: int = 42,
    ):

        self.dimensions = dimensions
        self.iterations = iterations
        self.decay = decay
        self.seed = seed
        self._weight = self.decay / self.dimensions

    def fit(self, graph):
        """
        Fitting a NodeSketch model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        num_of_nodes = len(graph.nodes)
        indptr, indices, _ = _kernel_csr(graph)
        orig_nnz = int(indices.size)
        # `_augment_sla` appends one entry per distinct target the previous
        # sketch chose inside a node's neighbourhood. It walks
        # `dimensions * degree(i)` (node, slot) pairs for node `i` and keeps
        # one entry per distinct target, so the count is bounded by
        # `min(dimensions * degree(i), num_of_nodes)` - bounding it by
        # `dimensions` alone under-counts as soon as a node's degree exceeds
        # one and the sketch spreads its choices, and the kernel then refuses
        # the run with a short capacity.
        degrees = np.diff(indptr.astype(np.int64))
        sla_cap = orig_nnz + int(
            np.minimum(self.dimensions * degrees, num_of_nodes).sum()
        )

        orig_rows = np.zeros(orig_nnz, dtype=np.int32)
        orig_cols = np.zeros(orig_nnz, dtype=np.int32)
        orig_vals = _zeros(orig_nnz)
        sla_rows = np.zeros(sla_cap, dtype=np.int32)
        sla_cols = np.zeros(sla_cap, dtype=np.int32)
        sla_vals = _zeros(sla_cap)
        hashes = _zeros(self.dimensions * num_of_nodes)
        sketch = np.zeros(self.dimensions * num_of_nodes, dtype=np.int32)
        state = _mt_state()
        stamp = np.zeros(num_of_nodes, dtype=np.int32)
        counts = np.zeros(num_of_nodes, dtype=np.int32)
        touched = np.zeros(num_of_nodes, dtype=np.int32)
        acc = _zeros(num_of_nodes)
        min_val = _zeros(num_of_nodes)
        status = _lib.kc_nodesketch_fit(
            num_of_nodes,
            self.dimensions,
            self.iterations,
            float(self.decay),
            self.seed,
            _index(indptr, "indptr"),
            _index(indices, "indices"),
            _index(orig_rows, "orig_rows"),
            _index(orig_cols, "orig_cols"),
            _data(orig_vals, "orig_vals"),
            _index(sla_rows, "sla_rows"),
            _index(sla_cols, "sla_cols"),
            _data(sla_vals, "sla_vals"),
            sla_cap,
            _data(hashes, "hashes"),
            _index(sketch, "sketch"),
            _words(state, "state"),
            _index(stamp, "stamp"),
            _index(counts, "counts"),
            _index(touched, "touched"),
            _data(acc, "acc"),
            _data(min_val, "min_val"),
        )
        if status != 0:
            raise ValueError(
                f"the augmented adjacency needs more than {sla_cap} entries"
            )
        self._num_nodes = num_of_nodes
        self._sketch = sketch

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = _zeros(self._num_nodes * self.dimensions)
        _lib.kc_nodesketch_get_embedding(
            self._num_nodes,
            self.dimensions,
            _index(self._sketch, "sketch"),
            _data(embedding, "embedding"),
        )
        return embedding.reshape(self._num_nodes, self.dimensions)


class FirstOrderLINE(Estimator):
    r"""An implementation of `"First-order LINE" <https://arxiv.org/abs/1503.03578>`_
    from the paper "LINE: Large-scale Information Network Embedding".

    The kernel reproduces numpy's legacy MT19937 and the fancy-index
    `+=` semantics upstream's `_update` depends on, so the embedding is
    bit-comparable with upstream's for a given seed.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        epochs (int): Number of epochs. Default is 100.
        mini_batch_size (int): Number of samples in each mini-batch. Default is 128.
        learning_rate (float): learning rate. Default is 0.05.
        learning_rate_decay (float): learning rate decay per epoch. Default is 0.999.
        verbose (bool): whether to show a loading bar using TQDM while queering.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 128,
        epochs: int = 100,
        mini_batch_size: int = 128,
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.999,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.mini_batch_size = mini_batch_size
        self.dimensions = dimensions
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.seed = seed
        self.verbose = verbose
        self.embedding = None

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a LINE model.

        Upstream shows a tqdm bar over the epochs; the epochs run inside one
        kernel call, so `verbose` prints the count instead.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()

        number_of_nodes = graph.number_of_nodes()

        edges = _edges(graph)
        number_of_edges = edges.shape[0]

        half = self.mini_batch_size // 2
        self.embedding = _zeros(number_of_nodes * self.dimensions)
        gradient = _zeros(number_of_nodes * self.dimensions)
        positive_index = np.zeros(half, dtype=np.int32)
        positive_batch = np.zeros(half * 2, dtype=np.int32)
        negative_batch = np.zeros(half * 2, dtype=np.int32)
        src_embedding = _zeros(half * self.dimensions)
        dst_embedding = _zeros(half * self.dimensions)
        activations = _zeros(half)
        base = _zeros(half * self.dimensions)
        # The generator state has to outlive the call: `_words(_mt_state())`
        # would hand the kernel the address of an array nothing references, and
        # the write would land in the freed block.
        state = _mt_state()
        if self.verbose:
            print(f"Epochs: {self.epochs}")
        _lib.kc_line_fit_first_order(
            number_of_nodes,
            self.dimensions,
            self.epochs,
            self.mini_batch_size,
            float(self.learning_rate),
            float(self.learning_rate_decay),
            number_of_edges,
            _index(edges, "edges"),
            self.seed,
            _data(self.embedding, "embedding"),
            _data(gradient, "gradient"),
            _index(positive_index, "positive_index"),
            _index(positive_batch, "positive_batch"),
            _index(negative_batch, "negative_batch"),
            _data(src_embedding, "src_embedding"),
            _data(dst_embedding, "dst_embedding"),
            _data(activations, "activations"),
            _data(base, "base"),
            _words(state, "state"),
        )
        self.embedding = self.embedding.reshape(
            number_of_nodes, self.dimensions
        )

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self.embedding


class SecondOrderLINE(Estimator):
    r"""An implementation of `"Second-order LINE" <https://arxiv.org/abs/1503.03578>`_
    from the paper "LINE: Large-scale Information Network Embedding".

    The kernel reproduces numpy's legacy MT19937 and the fancy-index
    `+=` semantics upstream's `_update` depends on, so the embedding is
    bit-comparable with upstream's for a given seed.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        epochs (int): Number of epochs. Default is 100.
        mini_batch_size (int): Number of samples in each mini-batch. Default is 128.
        learning_rate (float): learning rate. Default is 0.05.
        learning_rate_decay (float): learning rate decay per epoch. Default is 0.999.
        verbose (bool): whether to show a loading bar using TQDM while queering.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 128,
        epochs: int = 100,
        mini_batch_size: int = 128,
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.999,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.mini_batch_size = mini_batch_size
        self.dimensions = dimensions
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.seed = seed
        self.verbose = verbose
        self.src_embedding = None
        self.dst_embedding = None

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a LINE model.

        Upstream shows a tqdm bar over the epochs; the epochs run inside one
        kernel call, so `verbose` prints the count instead.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()

        number_of_nodes = graph.number_of_nodes()

        edges = _edges(graph)
        number_of_edges = edges.shape[0]

        half = self.mini_batch_size // 2
        # Upstream halves `dimensions` for the two tables, and an odd
        # `dimensions` loses a column there too.
        half_dimensions = self.dimensions // 2
        cells = number_of_nodes * half_dimensions
        self.src_embedding = _zeros(cells)
        self.dst_embedding = _zeros(cells)
        src_gradient = _zeros(cells)
        dst_gradient = _zeros(cells)
        positive_index = np.zeros(half, dtype=np.int32)
        positive_batch = np.zeros(half * 2, dtype=np.int32)
        negative_batch = np.zeros(half * 2, dtype=np.int32)
        src_batch = _zeros(half * half_dimensions)
        dst_batch = _zeros(half * half_dimensions)
        activations = _zeros(half)
        base = _zeros(half * half_dimensions)
        # The generator state has to outlive the call, as above.
        state = _mt_state()
        if self.verbose:
            print(f"Epochs: {self.epochs}")
        _lib.kc_line_fit_second_order(
            number_of_nodes,
            self.dimensions,
            self.epochs,
            self.mini_batch_size,
            float(self.learning_rate),
            float(self.learning_rate_decay),
            number_of_edges,
            _index(edges, "edges"),
            self.seed,
            _data(self.src_embedding, "src_embedding"),
            _data(self.dst_embedding, "dst_embedding"),
            _data(src_gradient, "src_gradient"),
            _data(dst_gradient, "dst_gradient"),
            _index(positive_index, "positive_index"),
            _index(positive_batch, "positive_batch"),
            _index(negative_batch, "negative_batch"),
            _data(src_batch, "src_batch"),
            _data(dst_batch, "dst_batch"),
            _data(activations, "activations"),
            _data(base, "base"),
            _words(state, "state"),
        )
        self.src_embedding = self.src_embedding.reshape(
            number_of_nodes, half_dimensions
        )
        self.dst_embedding = self.dst_embedding.reshape(
            number_of_nodes, half_dimensions
        )

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.hstack(
            [
                self.src_embedding,
                self.dst_embedding,
            ]
        )


class LaplacianEigenmaps(Estimator):
    r"""An implementation of `"Laplacian Eigenmaps" <https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering>`_
    from the NIPS '01 paper "Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering".
    The procedure extracts the eigenvectors corresponding to the largest eigenvalues
    of the graph Laplacian. These vectors are used as the node embedding.

    Upstream's ARPACK is replaced by a dense symmetric diagonalisation, so the
    eigenvectors agree with upstream only up to a per-column sign, which is
    arbitrary on both sides; compare the subspace, or the magnitudes, not the
    signs. `maximum_number_of_iterations` and `seed` are accepted for
    signature parity and unused: the dense path has no iteration budget, and the
    seed only ever reached ARPACK's start vector.

    Args:
        dimensions (int): Dimensionality of embedding. Default is 128.
        maximum_number_of_iterations (int): Maximum number of iterations to execute with ARPACK. The value will be multiplied by the number of nodes.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        dimensions: int = 128,
        maximum_number_of_iterations: int = 100,
        seed: int = 42,
    ):
        self.dimensions = dimensions
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Laplacian EigenMaps model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        number_of_nodes = graph.number_of_nodes()
        if self.dimensions >= number_of_nodes:
            raise ValueError(
                f"dimensions must be below the node count: {self.dimensions} "
                f">= {number_of_nodes}."
            )
        # Upstream's `nx.normalized_laplacian_matrix` defaults to
        # `weight="weight"` and 1.0 for an edge without the attribute, so the
        # weights are always read. It is not `nx.is_weighted`, which
        # `_ensure_integrity`'s unweighted self-loops make false.
        indptr, indices, values = _kernel_csr(graph, weighted=True)
        self._embedding = _zeros(number_of_nodes * self.dimensions)
        eigenvalues = _zeros(self.dimensions)
        laplacian = _zeros(number_of_nodes * number_of_nodes)
        vectors = _zeros(number_of_nodes * number_of_nodes)
        vt = _zeros(number_of_nodes * number_of_nodes)
        picked = np.zeros(number_of_nodes, dtype=np.int32)
        ok = _lib.kc_laplacian_eigenmaps_fit(
            _index(indptr, "indptr"),
            _index(indices, "indices"),
            _data(values, "values"),
            number_of_nodes,
            self.dimensions,
            self.maximum_number_of_iterations,
            self.seed,
            _data(self._embedding, "embedding"),
            _data(eigenvalues, "eigenvalues"),
            _data(laplacian, "laplacian"),
            _data(vectors),
            _data(vt),
            _index(picked, "picked"),
            EIGEN_SWEEPS,
            EIGEN_TOL,
        )
        if not ok:
            raise ValueError(
                f"the Jacobi sweep did not reach {EIGEN_TOL} in {EIGEN_SWEEPS} "
                f"passes, or dimensions={self.dimensions} is outside [1, "
                f"{number_of_nodes}]."
            )
        self._embedding = self._embedding.reshape(
            number_of_nodes, self.dimensions
        )

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
