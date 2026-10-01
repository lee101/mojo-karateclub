"""Time the real karateclub on the same fixtures `bench/bench.py` times the port on.

Runs inside the `upstream` pixi environment, the only one whose pins let
karateclub 1.3.3 install. The result is the left-hand column of the table
`bench/bench.py` prints:

    pixi run -e upstream python bench/bench_upstream.py

Each case constructs the model, fits it and reads the embedding as three
separate statements, because upstream `fit` returns `None` and a caller cannot
chain it. The graph is built once per case, outside the timed callable, and
reused across repetitions: `Estimator._check_graph` adds one self loop per node
on the first fit, and each fit after it re-adds the same edges in O(1), so the
first repetition pays an O(n) edge pass the rest do not and best-of-N drops it.
The three non-estimator cases (the walkers and the WL extractor) do not mutate
the graph at all. `bench/bench.py` builds and reuses its graphs the same way, so
the two columns are timed alike.

Timing is best-of-N, not mean: the machine is shared with the other agents in
this repo, the only samples a scheduler preempts are the slow ones, and a
preempted sample should not be charged to the library. N is 7 for a case that
measures under a millisecond and 3 for anything slower, so a sub-millisecond
case is not timed on a single sample.
"""

import datetime
import importlib.util
import json
import pathlib
import platform
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
TIMINGS = ROOT / "dist" / "upstream_timings.json"
GRAPHS = ROOT / "tests" / "_graphs.py"

FAST_REPEATS = 7
SLOW_REPEATS = 3
FAST_THRESHOLD = 1e-3


def load_fixture_module():
    """`tests/_graphs.py` by path: it is not a package, and the upstream
    environment cannot import `mojokarateclub` at all."""
    spec = importlib.util.spec_from_file_location("kc_fixtures", GRAPHS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn):
    """`(best-of-N seconds, N)` for one call, with the probe call as warmup."""
    start = time.perf_counter()
    fn()
    best = time.perf_counter() - start
    repeats = FAST_REPEATS if best < FAST_THRESHOLD else SLOW_REPEATS
    for _ in range(repeats - 1):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best, repeats


def estimator(factory, graphs):
    """A callable that constructs, fits and reads, as three statements.

    `graphs` selects upstream's two fit signatures: the graph-level models take
    a list of graphs, the node-level ones the bare graph.
    """

    def call(graph):
        model = factory()
        model.fit([graph] if graphs else graph)
        embedding = model.get_embedding()
        return embedding

    return call


def build_cases():
    """`(key, workload, fixture, callable)` in table order.

    The keys are the join key with `bench/bench.py` and the fixture names are
    the ones `tools/dump_upstream.py` dumps, so all three agree.
    """
    from karateclub.graph_embedding import FGSD, LDP, NetLSD, SF
    from karateclub.node_embedding.neighbourhood import (
        LaplacianEigenmaps,
        NodeSketch,
    )
    from karateclub.node_embedding.neighbourhood.first_order_line import FirstOrderLINE
    from karateclub.node_embedding.neighbourhood.second_order_line import (
        SecondOrderLINE,
    )
    from karateclub.utils.diffuser import EulerianDiffuser
    from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
    from karateclub.utils.walker import BiasedRandomWalker, RandomWalker

    def line(kind):
        return (
            f"{kind}(dimensions=8, epochs=5, mini_batch_size=64, verbose=False), "
            "fit(graph) + get_embedding()"
        )

    cases = [
        ("LDP/karate", "LDP(bins=32), fit([graph]) + get_embedding()", "karate",
         estimator(lambda: LDP(bins=32), True)),
        ("LDP/ws300", "LDP(bins=32), fit([graph]) + get_embedding()", "ws300",
         estimator(lambda: LDP(bins=32), True)),
        ("FGSD/karate", "FGSD(), fit([graph]) + get_embedding()", "karate",
         estimator(FGSD, True)),
        ("FGSD/ws300", "FGSD(), fit([graph]) + get_embedding()", "ws300",
         estimator(FGSD, True)),
        ("SF/karate", "SF(dimensions=16), fit([graph]) + get_embedding()", "karate",
         estimator(lambda: SF(dimensions=16), True)),
        ("SF/ws300", "SF(dimensions=32), fit([graph]) + get_embedding()", "ws300",
         estimator(lambda: SF(dimensions=32), True)),
        ("NetLSD/karate",
         "NetLSD(scale_steps=32, approximations=8), fit([graph]) + get_embedding()",
         "karate",
         estimator(lambda: NetLSD(scale_steps=32, approximations=8), True)),
        ("NetLSD/ws300",
         "NetLSD(scale_steps=32, approximations=16), fit([graph]) + get_embedding()",
         "ws300",
         estimator(lambda: NetLSD(scale_steps=32, approximations=16), True)),
        ("LaplacianEigenmaps/karate",
         "LaplacianEigenmaps(dimensions=8), fit(graph) + get_embedding()", "karate",
         estimator(lambda: LaplacianEigenmaps(dimensions=8), False)),
        ("LaplacianEigenmaps/ws300",
         "LaplacianEigenmaps(dimensions=8), fit(graph) + get_embedding()", "ws300",
         estimator(lambda: LaplacianEigenmaps(dimensions=8), False)),
        ("NodeSketch/karate",
         "NodeSketch(dimensions=16), fit(graph) + get_embedding()", "karate",
         estimator(lambda: NodeSketch(dimensions=16), False)),
        ("NodeSketch/ws300",
         "NodeSketch(dimensions=16), fit(graph) + get_embedding()", "ws300",
         estimator(lambda: NodeSketch(dimensions=16), False)),
        ("FirstOrderLINE/karate", line("FirstOrderLINE"), "karate",
         estimator(lambda: FirstOrderLINE(dimensions=8, epochs=5, mini_batch_size=64,
                                          verbose=False), False)),
        ("FirstOrderLINE/ws300", line("FirstOrderLINE"), "ws300",
         estimator(lambda: FirstOrderLINE(dimensions=8, epochs=5, mini_batch_size=64,
                                          verbose=False), False)),
        ("SecondOrderLINE/karate", line("SecondOrderLINE"), "karate",
         estimator(lambda: SecondOrderLINE(dimensions=8, epochs=5,
                                           mini_batch_size=64, verbose=False),
                   False)),
        ("SecondOrderLINE/ws300", line("SecondOrderLINE"), "ws300",
         estimator(lambda: SecondOrderLINE(dimensions=8, epochs=5,
                                           mini_batch_size=64, verbose=False),
                   False)),
        ("WeisfeilerLehmanHashing/ws300",
         "WeisfeilerLehmanHashing(graph, wl_iterations=3, attributed=False, "
         "erase_base_features=False).get_graph_features()", "ws300",
         lambda graph: WeisfeilerLehmanHashing(graph, 3, False, False)
         .get_graph_features()),
        ("RandomWalker/ws300",
         "RandomWalker(walk_length=40, walk_number=10).do_walks(graph)", "ws300",
         lambda graph: RandomWalker(40, 10).do_walks(graph)),
        ("BiasedRandomWalker/ws300",
         "BiasedRandomWalker(walk_length=40, walk_number=10, p=0.5, q=2.0)"
         ".do_walks(graph)", "ws300",
         lambda graph: BiasedRandomWalker(40, 10, 0.5, 2.0).do_walks(graph)),
        ("EulerianDiffuser/karate",
         "EulerianDiffuser(diffusion_number=10, diffusion_cover=80)"
         ".do_diffusions(graph)", "karate",
         lambda graph: EulerianDiffuser(10, 80).do_diffusions(graph)),
    ]
    return cases


def environment():
    import karateclub
    import networkx
    import numpy
    import scipy

    return {
        "karateclub": karateclub.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "networkx": networkx.__version__,
        "scipy": scipy.__version__,
    }


def main():
    fixtures = load_fixture_module()
    cases = build_cases()
    timings = {}
    repeats = {}
    failures = []
    for key, _workload, fixture, call in cases:
        graph, _ = fixtures.build_graph(fixture)
        try:
            seconds, count = timed(lambda call=call, graph=graph: call(graph))
        except Exception:
            failures.append((key, traceback.format_exc()))
            print(f"  {key:<32} FAILED", flush=True)
            continue
        timings[key] = seconds
        repeats[key] = count
        print(f"  {key:<32} {seconds * 1e3:>10.3f} ms  (best of {count})",
              flush=True)

    payload = {
        "measured": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "environment": environment(),
        "repeats": repeats,
        "timings": timings,
    }
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {TIMINGS} with {len(timings)} timings, measured "
          f"{payload['measured']}")

    for key, tb in failures:
        print(f"\nupstream {key} raised:\n{tb}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
