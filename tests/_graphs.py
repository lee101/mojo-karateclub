"""The fixture graphs, built the same way `tools/dump_upstream.py` builds them.

The builders are imported by path rather than through the package so the
upstream environment can use them too, where the package's ctypes layer is not
loadable. Nothing here depends on networkx's generators, whose RNG schedules
differ between the 2.6 the upstream environment resolves and the 3.x this one
does; an edge list is an edge list.
"""

import importlib.util
import pathlib

import networkx as nx

_PATH = pathlib.Path(__file__).resolve().parents[1] / "python" / "mojokarateclub" / "_datasets.py"
_spec = importlib.util.spec_from_file_location("kc_datasets", _PATH)
_datasets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_datasets)

__all__ = ["build_graph", "FIXTURES", "datasets"]

datasets = _datasets

FIXTURES = (
    "karate",
    "gnp60",
    "gnp200",
    "lollipop",
    "isolated",
    "grid",
    "ws300",
    "ba2000",
)


def build_graph(name, integrity=False):
    """`(networkx graph, n)` for one fixture, nodes labelled `0..n-1`.

    `integrity=True` applies `Estimator._ensure_integrity`, which is the
    self-loop pass every model runs before it looks at the graph, and which the
    WL feature extractor therefore sees too.
    """
    builder = {
        "karate": datasets.karate_edges,
        "gnp60": lambda: datasets.gnp_edges(60, 0.12, 20260926),
        "gnp200": lambda: datasets.gnp_edges(200, 0.04, 20260926),
        "lollipop": lambda: datasets.lollipop_edges(8, 12),
        "isolated": datasets.isolated_edges,
        "grid": lambda: datasets.grid_edges(6, 5),
        "ws300": lambda: datasets.newman_watts_strogatz_edges(300, 4, 0.1, 20260926),
        "ba2000": lambda: datasets.barabasi_albert_edges(2000, 3, 20260926),
    }[name]
    edges, n = builder()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    if integrity:
        graph.add_edges_from((index, index) for index in range(n))
    return graph, n
