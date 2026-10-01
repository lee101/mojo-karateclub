"""Graph-level estimators, mirroring `karateclub.graph_embedding`.

Each class keeps upstream's name, constructor argument order and defaults, and
its `fit` / `get_embedding` / `infer` surface. The compute upstream spends in
numpy, scipy and networkx happens in the Mojo kernels named in each method;
what is left here is the argument marshalling, the scratch allocation and, for
`Graph2Vec`, the doc2vec training upstream also delegates to gensim.

`Estimator._check_graph` has already added a self loop to every node by the
time a kernel runs, which is upstream's `_ensure_integrity` and is part of the
contract several of these kernels depend on.
"""

import networkx as nx
import numpy as np

from ._ffi import _csr, _data, _index, _lib, _zeros
from .estimator import Estimator, csr_from_graph

__all__ = ["LDP", "FGSD", "SF", "NetLSD", "Graph2Vec"]

# `linalg.jacobi_eigh` returns its sweeps used, and a run that returns
# `sweeps` never reached `tol`. `tol` is a summed-squared-off-diagonal budget,
# not a per-element one: at 1e-12 the near-null eigenvector of the karate club
# is only about 2.3e-7 accurate, at 1e-24 it agrees with ARPACK to 4.7e-15.
EIGEN_SWEEPS = 400
EIGEN_TOL = 1e-24
# `numpy.linalg.pinv` default.
PINV_RCOND = 1e-15

def _graph_csr(graph, weighted=False):
    """The graph as the kernels take it: symmetric CSR, one node per row.

    The self loops are already there: `Estimator._check_graph` put them in
    before `_calculate_*` is called, and adding them a second time would give
    every node the degree it would have with a double self loop.

    `weighted` is for the estimators whose upstream calls
    `nx.normalized_laplacian_matrix`, which reads the `weight` attribute by
    default. `LDP` counts degrees and is unweighted, as upstream.
    """
    indptr, indices, values = csr_from_graph(graph, weighted=weighted)
    return _csr(indptr, indices, values)


def _spectral_csr(graph):
    """`_graph_csr` for a spectral estimator.

    Always weighted: upstream's `nx.normalized_laplacian_matrix` defaults to
    `weight="weight"` and uses 1.0 for an edge without the attribute, which is
    what `csr_from_graph(weighted=True)` does. It is not
    `nx.is_weighted`, which `_ensure_integrity`'s unweighted self-loops make
    false for every graph that reaches a model.
    """
    return _graph_csr(graph, weighted=True)


def _dense_scratch(n, count=3):
    """`count` fresh `n x n` blocks.

    Three is what a spectral path needs: the matrix, the eigenvector block,
    and that block transposed, which is the orientation the Jacobi sweep can
    update with contiguous row writes.
    """
    return tuple(_zeros(n * n) for _ in range(count))


def _picked(n):
    """The `Int32[n]` marker `spectral.eig_select` uses while it picks."""
    return np.zeros(n, dtype=np.int32)


class LDP(Estimator):
    r"""An implementation of `"LDP" <https://arxiv.org/abs/1811.03508>`_ from the
    ICLR Representation Learning on Graphs and Manifolds Workshop '19 paper "A
    Simple Yet Effective Baseline for Non-Attributed Graph Classification". The
    procedure calculates histograms of degree profiles. These concatenated
    histograms form the graph representations.

    Args:
        bins (int): Number of histogram bins. Default is 32.
    """

    def __init__(self, bins: int = 32):
        self.bins = bins
        if bins < 1:
            # `numpy.histogram` raises `ValueError` for a non-positive count.
            raise ValueError(f"bins must be at least 1, got {bins}")

    def _calculate_ldp(self, graph):
        """Calculating the local degree profile features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        indptr, indices, values, n = _graph_csr(graph)
        degrees = _zeros(n)
        _lib.kc_ldp_log_degrees(_index(indptr), _index(indices), n, _data(degrees))
        features = _zeros(5 * n)
        _lib.kc_ldp_features(
            _index(indptr), _index(indices), _data(degrees), n, _data(features)
        )
        embedding = _zeros(5 * self.bins)
        column = _zeros(n)
        _lib.kc_ldp_embedding(
            _data(features), n, self.bins, _data(embedding), _data(column)
        )
        return embedding

    def fit(self, graphs):
        """Fitting an LDP model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_ldp(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs):
        r"""Infer the embedding of graphs.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        graphs = self._check_graphs(graphs)
        embedding = np.array([self._calculate_ldp(graph) for graph in graphs])
        return embedding


class FGSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure calculates the Moore-Penrose spectrum of the normalized Laplacian.
    Using this spectrum the histogram of the spectral features is used as a whole graph representation.

    Args:
        hist_bins (int): Number of histogram bins. Default is 200.
        hist_range (int): Histogram range considered. Default is 20.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(self, hist_bins: int = 200, hist_range: int = 20, seed: int = 42):

        if hist_bins < 1:
            # `numpy.histogram` raises `ValueError` for a non-positive count.
            raise ValueError(f"hist_bins must be at least 1, got {hist_bins}")
        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.seed = seed

    def _calculate_fgsd(self, graph):
        """Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        indptr, indices, values, n = _spectral_csr(graph)
        # Five `n x n` buffers: the Laplacian, the pseudo-inverse, and the
        # three work areas the eigensolver inside it needs (the matrix, the
        # eigenvector block, and that block transposed).
        laplacian, scratch_a, scratch_b, scratch_c = _dense_scratch(n, 4)
        pinv = _zeros(n * n)
        histogram = _zeros(self.hist_bins)
        if not _lib.kc_fgsd_embedding(
            _index(indptr),
            _index(indices),
            _data(values),
            n,
            _data(laplacian),
            _data(pinv),
            _data(scratch_a),
            _data(scratch_b),
            _data(scratch_c),
            _index(_picked(n)),
            PINV_RCOND,
            EIGEN_SWEEPS,
            EIGEN_TOL,
            self.hist_bins,
            float(self.hist_range[0]),
            float(self.hist_range[1]),
            _data(histogram),
        ):
            raise np.linalg.LinAlgError(
                f"the Jacobi sweep did not converge in {EIGEN_SWEEPS} passes"
            )
        return histogram

    def fit(self, graphs):
        """Fitting a FGSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_fgsd(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Inferring the embedding for a list of graphs.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        embedding = np.array([self._calculate_fgsd(graph) for graph in graphs])
        return embedding


class SF(Estimator):
    r"""An implementation of `"SF" <https://arxiv.org/abs/1810.09155>`_
    from the NeurIPS Relational Representation Learning Workshop '18 paper "A Simple Baseline Algorithm for Graph Classification".
    The procedure calculates the k lowest eigenvalues of the normalized Laplacian.
    If the graph has a lower number of eigenvalues than k the representation is padded with zeros.

    Args:
        dimensions (int): Number of lowest eigenvalues. Default is 128.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(self, dimensions: int = 128, seed: int = 42):
        self.dimensions = dimensions
        self.seed = seed

    def _calculate_sf(self, graph):
        """Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of a single graph.
        """
        indptr, indices, values, n = _spectral_csr(graph)
        laplacian, vectors, vt = _dense_scratch(n)
        picked = _picked(n)
        # `k = dimensions` unless the graph is smaller, in which case `k = n - 1`.
        selected = _zeros(self.dimensions)
        embedding = _zeros(self.dimensions)
        vectors_out = _zeros(n * self.dimensions)
        if not _lib.kc_sf_calculate(
            _index(indptr),
            _index(indices),
            _data(values),
            n,
            self.dimensions,
            _data(laplacian),
            _data(vectors),
            _data(vt),
            _index(picked),
            _data(selected),
            _data(embedding),
            _data(vectors_out),
            EIGEN_SWEEPS,
            EIGEN_TOL,
        ):
            raise np.linalg.LinAlgError(
                f"the Jacobi sweep did not converge in {EIGEN_SWEEPS} passes"
            )
        return embedding

    def fit(self, graphs):
        """Fitting a SF model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_sf(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Inferring the embedding vectors.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        embedding = np.array([self._calculate_sf(graph) for graph in graphs])
        return embedding


class NetLSD(Estimator):
    r"""An implementation of `"NetLSD" <https://arxiv.org/abs/1805.10712>`_
    from the KDD '18 paper "NetLSD: Hearing the Shape of a Graph". The procedure
    calculate the heat kernel trace of the normalized Laplacian matrix over a
    vector of time scales. If the matrix is large it switches to an approximation
    of the eigenvalues.

    Args:
        scale_min (float): Time scale interval minimum. Default is -2.0.
        scale_max (float): Time scale interval maximum. Default is 2.0.
        scale_steps (int): Number of steps in time scale. Default is 250.
        scale_approximations (int): Number of eigenvalue approximations. Default is 200.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        scale_min: float = -2.0,
        scale_max: float = 2.0,
        scale_steps: int = 250,
        approximations: int = 200,
        seed: int = 42,
    ):

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_steps = scale_steps
        self.approximations = approximations
        self.seed = seed
        if scale_steps < 1:
            raise ValueError(f"scale_steps must be at least 1, got {scale_steps}")
        if approximations < 1:
            # The approximate branch takes `approximations` eigenvalues from
            # each end of the spectrum; with none, upstream's `eigsh(k=0)`
            # raises and the kernel would index an empty block.
            raise ValueError(
                f"approximations must be at least 1, got {approximations}"
            )

    def _calculate_netlsd(self, graph):
        """Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        indptr, indices, values, n = _spectral_csr(graph)
        nnz = indices.size
        laplacian, vectors, vt = _dense_scratch(n)
        picked = _picked(n)
        loopless = (
            np.zeros(n + 1, dtype=np.int32),
            np.zeros(nnz, dtype=np.int32),
            _zeros(nnz),
        )
        eigenvalues_lower = _zeros(self.approximations)
        eigenvalues_upper = _zeros(self.approximations)
        eigenvalues = _zeros(n)
        timescales = _zeros(self.scale_steps)
        embedding = _zeros(self.scale_steps)
        if not _lib.kc_netlsd_calculate(
            _index(indptr),
            _index(indices),
            _data(values),
            n,
            float(self.scale_min),
            float(self.scale_max),
            self.scale_steps,
            self.approximations,
            _index(loopless[0]),
            _index(loopless[1]),
            _data(loopless[2]),
            _data(laplacian),
            _data(vectors),
            _data(vt),
            _index(picked),
            _data(eigenvalues_lower),
            _data(eigenvalues_upper),
            _data(eigenvalues),
            _data(timescales),
            _data(embedding),
            EIGEN_SWEEPS,
            EIGEN_TOL,
        ):
            raise np.linalg.LinAlgError(
                f"the Jacobi sweep did not converge in {EIGEN_SWEEPS} passes"
            )
        return embedding

    def fit(self, graphs):
        """Fitting a NetLSD model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        self._embedding = [self._calculate_netlsd(graph) for graph in graphs]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs):
        """Infer the embedding of the graphs.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        embedding = np.array([self._calculate_netlsd(graph) for graph in graphs])
        return embedding


class Graph2Vec(Estimator):
    r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurrence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):

        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

    def fit(self, graphs):
        """Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        from .utils import WeisfeilerLehmanHashing

        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        self._embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        from .utils import WeisfeilerLehmanHashing

        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]
        return np.array(embedding)
