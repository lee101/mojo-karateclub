"""Dump reference outputs from the real upstream karateclub, for the parity tests.

Runs inside the `upstream` pixi environment, which is the only one whose pins
(`numpy<1.23`, `networkx<2.7`, `pandas<=1.3.5`) let karateclub 1.3.3 install at
all. It writes `tests/vectors/upstream.npz`, the fixture the default
environment's pytest run compares against; re-run it with `pixi run vectors`
whenever the ported set changes.

    pixi run vectors
"""

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests" / "vectors" / "upstream.npz"


def load_datasets():
    """Load `_datasets.py` by path, without importing the package.

    The package's `__init__` loads the Mojo shared library, which the upstream
    environment has no reason to have.
    """
    path = ROOT / "python" / "mojokarateclub" / "_datasets.py"
    spec = importlib.util.spec_from_file_location("kc_datasets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_graph(datasets, name):
    """`(networkx graph, n)` for one fixture, nodes labelled `0..n-1`."""
    import networkx as nx

    if name == "karate":
        edges, n = datasets.karate_edges()
    elif name == "gnp60":
        edges, n = datasets.gnp_edges(60, 0.12, 20260926)
    elif name == "gnp200":
        edges, n = datasets.gnp_edges(200, 0.04, 20260926)
    elif name == "lollipop":
        edges, n = datasets.lollipop_edges(8, 12)
    elif name == "isolated":
        edges, n = datasets.isolated_edges()
    elif name == "grid":
        edges, n = datasets.grid_edges(6, 5)
    elif name == "ws300":
        edges, n = datasets.newman_watts_strogatz_edges(300, 4, 0.1, 20260926)
    elif name == "ba2000":
        edges, n = datasets.barabasi_albert_edges(2000, 3, 20260926)
    else:
        raise KeyError(name)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph, n


def _with_integrity(graph):
    """`Estimator._ensure_integrity`: the self loops every model runs first."""
    graph.add_edges_from((index, index) for index in range(graph.number_of_nodes()))
    return graph


def main():
    import karateclub
    from karateclub.graph_embedding import FGSD, LDP, NetLSD, SF
    from karateclub.node_embedding.neighbourhood import (
        LaplacianEigenmaps,
        NodeSketch,
    )
    from karateclub.node_embedding.neighbourhood.first_order_line import FirstOrderLINE
    from karateclub.node_embedding.neighbourhood.second_order_line import (
        SecondOrderLINE,
    )
    from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
    from karateclub.utils.walker import BiasedRandomWalker, RandomWalker

    datasets = load_datasets()
    out = {}
    out["karateclub_version"] = np.array(karateclub.__version__)

    small = ["karate", "gnp60", "lollipop", "isolated", "grid"]

    for name in small:
        graph, _ = build_graph(datasets, name)

        model = LDP(bins=8)
        model.fit([graph])
        out[f"ldp/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        model = FGSD(hist_bins=32, hist_range=20)
        model.fit([graph])
        out[f"fgsd/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        model = SF(dimensions=8)
        model.fit([graph])
        out[f"sf/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        model = NetLSD(scale_steps=16, approximations=4)
        model.fit([graph])
        out[f"netlsd/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        graph, _ = build_graph(datasets, name)
        model = LaplacianEigenmaps(dimensions=8)
        model.fit(graph)
        out[f"laplacian/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        graph, _ = build_graph(datasets, name)
        model = NodeSketch(dimensions=8, iterations=2, decay=0.01, seed=42)
        model.fit(graph)
        out[f"nodesketch/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        # `Graph2Vec.fit` runs `_check_graphs` before it hashes, so the graph
        # the WL extractor sees always carries the self loops. Make that
        # explicit here rather than relying on which estimator ran last.
        for iterations in (1, 2):
            for erase in (False, True):
                graph, _ = build_graph(datasets, name)
                graph = _with_integrity(graph)
                wl = WeisfeilerLehmanHashing(graph, iterations, False, erase)
                key = f"wl/{name}/{iterations}/{int(erase)}"
                out[key] = np.array(
                    [
                        np.array([node] + list(features), dtype=object)
                        for node, features in wl.get_node_features().items()
                    ],
                    dtype=object,
                )

    attributed = _with_integrity(build_graph(datasets, "karate")[0])
    for node in attributed.nodes():
        attributed.nodes[node]["feature"] = "attr-%d" % (node % 3)
    wl = WeisfeilerLehmanHashing(attributed, 2, True, False)
    out["wl/attributed/2/0"] = np.array(
        [
            np.array([node] + list(features), dtype=object)
            for node, features in wl.get_node_features().items()
        ],
        dtype=object,
    )

    for name in ("karate", "gnp200"):
        graph, n = build_graph(datasets, name)
        walker = RandomWalker(20, 5)
        walker.do_walks(graph)
        out[f"walk/count/{name}"] = np.array([len(walker.walks)])
        out[f"walk/length/{name}"] = np.array([len(w) for w in walker.walks])
        histogram = np.zeros(n, dtype=np.int64)
        for walk in walker.walks:
            for token in walk:
                histogram[int(token)] += 1
        out[f"walk/histogram/{name}"] = histogram

        graph, _ = build_graph(datasets, name)
        walker = BiasedRandomWalker(20, 5, 0.5, 2.0)
        walker.do_walks(graph)
        out[f"biased/count/{name}"] = np.array([len(walker.walks)])
        out[f"biased/length/{name}"] = np.array([len(w) for w in walker.walks])
        histogram = np.zeros(n, dtype=np.int64)
        for walk in walker.walks:
            for token in walk:
                histogram[int(token)] += 1
        out[f"biased/histogram/{name}"] = histogram

    for name in ("karate", "gnp200"):
        graph, _ = build_graph(datasets, name)
        model = FirstOrderLINE(
            dimensions=8, epochs=3, mini_batch_size=32, verbose=False, seed=42
        )
        model.fit(graph)
        out[f"line1/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

        graph, _ = build_graph(datasets, name)
        model = SecondOrderLINE(
            dimensions=8, epochs=3, mini_batch_size=32, verbose=False, seed=42
        )
        model.fit(graph)
        out[f"line2/{name}"] = np.asarray(model.get_embedding(), dtype=np.float64)

    # Legacy MT19937 stream, so the NodeSketch and LINE ports can be checked
    # against the generator they are supposed to reproduce.
    legacy = np.random.RandomState(42)
    out["mt19937/rand"] = legacy.rand(64)
    out["mt19937/randint"] = legacy.randint(0, 97, size=64)
    legacy = np.random.RandomState(7)
    out["mt19937/rand7"] = legacy.rand(64)
    out["mt19937/randint7"] = legacy.randint(0, 97, size=64)

    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VECTORS, **out)
    print(f"wrote {VECTORS} with {len(out)} arrays")


if __name__ == "__main__":
    sys.exit(main())
