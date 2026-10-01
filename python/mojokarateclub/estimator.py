"""Estimator base class, mirroring `karateclub.estimator.Estimator`.

Upstream is one file with seven methods; this is the same thing with the graph
checks split out into `csr.py`, because the Mojo kernels take a CSR and not a
networkx graph. `fit`, `get_embedding`, `get_memberships`,
`get_cluster_centers`, `get_params` and `_set_seed` are unchanged.
"""

import random

import networkx as nx
import numpy as np

__all__ = ["Estimator", "csr_from_graph"]


class Estimator(object):
    """Estimator base class with constructor and public methods."""

    seed: int

    def fit(self):
        """Fitting a model."""

    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""

    def get_memberships(self):
        """Getting the membership dictionary."""

    def get_cluster_centers(self):
        """Getting the cluster centers."""

    def get_params(self):
        """Get parameter dictionary for this estimator."""
        import re

        rx = re.compile(r"^\_")
        params = self.__dict__
        params = {key: params[key] for key in params if not rx.search(key)}
        return params

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_integrity(graph)

        return graph

    def _check_graphs(self, graphs):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs


def csr_from_graph(graph: nx.classes.graph.Graph, weighted: bool = False):
    """`(indptr, indices, values)` of `graph`, node `i` at row `i`.

    Both directions of every undirected edge are present and each row is in
    ascending column order, which is the layout every kernel in
    `src/mojokarateclub` assumes. Self-loops are NOT added here: upstream
    adds them in `_ensure_integrity`, and the tests check the layer that does
    it, so the two stay separable.
    """
    number_of_nodes = graph.number_of_nodes()
    # Row `i` is node `i`, so the rows are walked in label order and not in
    # networkx's insertion order: upstream builds every matrix with
    # `nodelist=range(n)`, and `_check_indexing` only guarantees the labels
    # are `0..n-1`, not the order they were added in.
    nodes = sorted(graph.nodes())
    index_of = {node: index for index, node in enumerate(nodes)}
    rows = []
    cols = []
    vals = []
    for node in nodes:
        source = index_of[node]
        for neighbour in graph.neighbors(node):
            rows.append(source)
            cols.append(index_of[neighbour])
            if weighted:
                vals.append(float(graph[node][neighbour].get("weight", 1.0)))
            else:
                vals.append(1.0)
    if not rows:
        indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
        return indptr, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(number_of_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=number_of_nodes), out=indptr[1:])
    return (
        _as_indices(indptr),
        _as_indices(cols),
        np.ascontiguousarray(vals),
    )


def _as_indices(array):
    """Narrow an index vector to the `int32` the kernels take, or say why not.

    The kernels read these as `int32`; a silent wrap would produce a CSR that
    fails the `indptr` invariants far from here, so the overflow is refused
    where it happens.
    """
    if array.size and int(array.max()) > np.iinfo(np.int32).max:
        raise ValueError("the CSR needs more than 2**31 - 1 entries")
    return array.astype(np.int32)
