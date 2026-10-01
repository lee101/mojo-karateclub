"""Deterministic graphs used by the parity tests and the benchmarks.

Graphs are built from their edge lists here rather than with `networkx`
generators: the default environment has networkx 3.x and the upstream
reference environment has networkx 2.6, and a generator whose RNG schedule
changed between those releases would silently stop being the same graph. An
edge list is an edge list.

Importable without the rest of the package (no ctypes, no numpy-only helpers
beyond `numpy`), so `tools/dump_upstream.py` can load it by path inside the
upstream environment.

Every builder returns `(edges, number_of_nodes)` where `edges` is a list of
`(u, v)` with `u != v`, each unordered pair appearing once.
"""

import numpy as np

__all__ = [
    "karate_edges",
    "barabasi_albert_edges",
    "newman_watts_strogatz_edges",
    "gnp_edges",
    "grid_edges",
    "lollipop_edges",
    "isolated_edges",
    "to_csr",
]


def karate_edges():
    """Zachary's karate club, the 78 undirected edges of the canonical graph."""
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 32), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32),
        (32, 33),
    ]
    return pairs, 34


def barabasi_albert_edges(n, m, seed):
    """Preferential attachment; the initial seed clique is nodes `0..m-1`."""
    rng = np.random.default_rng(seed)
    edges = set()
    repeated = []
    for node in range(m, n):
        targets = set()
        while len(targets) < m:
            if repeated and rng.random() < 0.8:
                targets.add(int(rng.choice(repeated)))
            else:
                targets.add(int(rng.integers(0, node)))
        for target in sorted(targets):
            if target == node:
                continue
            edges.add((min(node, target), max(node, target)))
            repeated.append(target)
            repeated.append(node)
    return sorted(edges), n


def newman_watts_strogatz_edges(n, k, p, seed):
    """Ring lattice of `k` neighbours per side with `p` rewiring probability."""
    rng = np.random.default_rng(seed)
    edges = set()
    for node in range(n):
        for offset in range(1, k + 1):
            other = (node + offset) % n
            if node != other:
                edges.add((min(node, other), max(node, other)))
    for node in range(n):
        for offset in range(1, k + 1):
            other = (node + offset) % n
            if node == other:
                continue
            if rng.random() >= p:
                continue
            replacement = int(rng.integers(0, n))
            if replacement == node or (min(node, replacement), max(node, replacement)) in edges:
                continue
            edges.discard((min(node, other), max(node, other)))
            edges.add((min(node, replacement), max(node, replacement)))
    return sorted(edges), n


def gnp_edges(n, p, seed):
    """Erdos-Renyi `G(n, p)`: every pair is an edge with probability `p`."""
    rng = np.random.default_rng(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return sorted(edges), n


def grid_edges(rows, cols):
    """A `rows x cols` lattice; disconnected when either side is small."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                edges.append((node, node + 1))
            if r + 1 < rows:
                edges.append((node, node + cols))
    return sorted(edges), rows * cols


def lollipop_edges(clique, tail):
    """A `clique`-clique with a path of `tail` nodes hanging off node 0."""
    edges = []
    for u in range(clique):
        for v in range(u + 1, clique):
            edges.append((u, v))
    for step in range(tail):
        edges.append((step, step + 1))
    return sorted(set(edges)), clique + tail


def isolated_edges():
    """Two triangles plus four nodes with no edges at all."""
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    return sorted(edges), 10


def to_csr(edges, n):
    """`(indptr, indices)` of the symmetric, self-loop-free adjacency.

    Both directions are present, which is what `networkx.to_scipy_sparse_array`
    and `nx.adjacency_matrix` hand to a downstream consumer. The Python layer
    adds the self-loops that `Estimator._ensure_integrity` adds, so this
    returns the graph as it is given to a model, not as the model sees it.
    """
    rows = np.zeros(2 * len(edges), dtype=np.int64)
    cols = np.zeros(2 * len(edges), dtype=np.int64)
    for index, (u, v) in enumerate(edges):
        rows[2 * index] = u
        cols[2 * index] = v
        rows[2 * index + 1] = v
        cols[2 * index + 1] = u
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    for row in rows:
        indptr[row + 1] += 1
    np.cumsum(indptr, out=indptr)
    return indptr.astype(np.int32), cols.astype(np.int32)
