"""Time this port against the real karateclub, and print the comparison table.

Runs in the default pixi environment - the one with the Mojo toolchain, numpy
2 and the package on `PYTHONPATH` - against numbers `bench/bench_upstream.py`
measured in the `upstream` environment, which is the only one karateclub 1.3.3
installs into. The two environments cannot be merged (numpy 1.22 versus 2, and
the ctypes layer needs the default env's ABI), so they are two processes that
join on a case key:

    pixi run bench        # bench/run.sh, which runs both halves

Each case constructs the model, fits it and reads the embedding, on the same
fixture graph and with the same parameters as the upstream half. The graph is
built once per case, outside the timed callable, and reused across
repetitions: `Estimator._check_graph` adds one self loop per node on the first
fit, and the fits after it re-add edges networkx already holds in O(1) each, so
best-of-N drops that one O(n) pass. The three non-estimator cases (the walkers
and the WL extractor) do not touch the graph at all. The upstream half builds
and reuses its graphs the same way, so the two columns are timed alike.

Timing is best-of-N rather than mean: this machine is shared, the only samples
a scheduler preempts are the slow ones, and a preempted sample should not be
charged to the library. N is 7 for a case that measures under a millisecond and
3 otherwise.

The spectral rows are expected to lose. Upstream reaches multithreaded
LAPACK/ARPACK through scipy; this port runs a serial dense Jacobi sweep, so
there the honest row is the one where the ratio is below 1.0, and it is
printed as it stands.
"""
import importlib.util
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "dist" / "upstream_timings.json"
GRAPHS = ROOT / "tests" / "_graphs.py"

FAST_REPEATS = 7
SLOW_REPEATS = 3
FAST_THRESHOLD = 1e-3


def load_fixture_module():
    """`tests/_graphs.py` by path rather than through the package, so this file
    and `bench/bench_upstream.py` build byte-identical graphs."""
    spec = importlib.util.spec_from_file_location("kc_fixtures", GRAPHS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(name):
    """The port's class, wherever the package re-exports it.

    A symbol the package does not export yet is a per-case failure, not a
    reason to drop the case or to abort the table.
    """
    problems = []
    for module_name in ("mojokarateclub", "mojokarateclub.node_embedding",
                        "mojokarateclub.utils"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            problems.append(f"{module_name}: {exc}")
            continue
        klass = getattr(module, name, None)
        if klass is not None:
            return klass
    raise ImportError(f"mojokarateclub exports no {name}\n  " + "\n  ".join(problems))


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


def estimator(name, kwargs, graphs):
    """Build a callable that constructs, fits and reads, as three statements.

    `graphs` selects the two fit signatures: the graph-level models take a list
    of graphs, the node-level ones the bare graph.
    """

    def build():
        klass = resolve(name)

        def call(graph):
            model = klass(**kwargs)
            model.fit([graph] if graphs else graph)
            embedding = model.get_embedding()
            return embedding

        return call

    return build


def obj(name, construct, read):
    """Build a callable for a case that is one object, not an estimator: the
    walkers and the WL feature extractor, which have no `fit`."""

    def build():
        klass = resolve(name)

        def call(graph):
            instance = construct(klass, graph)
            return read(instance, graph)

        return call

    return build


def build_cases():
    """`(key, workload, fixture, callable-builder)` in table order.

    The keys and the parameters are the ones `bench/bench_upstream.py` times,
    so the join in `read_upstream()` finds every row.
    """
    graph_level = "fit([graph]) + get_embedding()"
    node_level = "fit(graph) + get_embedding()"
    line = ("dimensions=8, epochs=5, mini_batch_size=64, verbose=False")
    return [
        ("LDP/karate", f"LDP(bins=32), {graph_level}", "karate",
         estimator("LDP", {"bins": 32}, True)),
        ("LDP/ws300", f"LDP(bins=32), {graph_level}", "ws300",
         estimator("LDP", {"bins": 32}, True)),
        ("FGSD/karate", f"FGSD(), {graph_level}", "karate",
         estimator("FGSD", {}, True)),
        ("FGSD/ws300", f"FGSD(), {graph_level}", "ws300",
         estimator("FGSD", {}, True)),
        ("SF/karate", f"SF(dimensions=16), {graph_level}", "karate",
         estimator("SF", {"dimensions": 16}, True)),
        ("SF/ws300", f"SF(dimensions=32), {graph_level}", "ws300",
         estimator("SF", {"dimensions": 32}, True)),
        ("NetLSD/karate",
         f"NetLSD(scale_steps=32, approximations=8), {graph_level}", "karate",
         estimator("NetLSD", {"scale_steps": 32, "approximations": 8}, True)),
        ("NetLSD/ws300",
         f"NetLSD(scale_steps=32, approximations=16), {graph_level}", "ws300",
         estimator("NetLSD", {"scale_steps": 32, "approximations": 16}, True)),
        ("LaplacianEigenmaps/karate",
         f"LaplacianEigenmaps(dimensions=8), {node_level}", "karate",
         estimator("LaplacianEigenmaps", {"dimensions": 8}, False)),
        ("LaplacianEigenmaps/ws300",
         f"LaplacianEigenmaps(dimensions=8), {node_level}", "ws300",
         estimator("LaplacianEigenmaps", {"dimensions": 8}, False)),
        ("NodeSketch/karate", f"NodeSketch(dimensions=16), {node_level}", "karate",
         estimator("NodeSketch", {"dimensions": 16}, False)),
        ("NodeSketch/ws300", f"NodeSketch(dimensions=16), {node_level}", "ws300",
         estimator("NodeSketch", {"dimensions": 16}, False)),
        ("FirstOrderLINE/karate", f"FirstOrderLINE({line}), {node_level}", "karate",
         estimator("FirstOrderLINE",
                   {"dimensions": 8, "epochs": 5, "mini_batch_size": 64,
                    "verbose": False}, False)),
        ("FirstOrderLINE/ws300", f"FirstOrderLINE({line}), {node_level}", "ws300",
         estimator("FirstOrderLINE",
                   {"dimensions": 8, "epochs": 5, "mini_batch_size": 64,
                    "verbose": False}, False)),
        ("SecondOrderLINE/karate", f"SecondOrderLINE({line}), {node_level}",
         "karate",
         estimator("SecondOrderLINE",
                   {"dimensions": 8, "epochs": 5, "mini_batch_size": 64,
                    "verbose": False}, False)),
        ("SecondOrderLINE/ws300", f"SecondOrderLINE({line}), {node_level}",
         "ws300",
         estimator("SecondOrderLINE",
                   {"dimensions": 8, "epochs": 5, "mini_batch_size": 64,
                    "verbose": False}, False)),
        ("WeisfeilerLehmanHashing/ws300",
         "WeisfeilerLehmanHashing(graph, wl_iterations=3, attributed=False, "
         "erase_base_features=False).get_graph_features()", "ws300",
         obj("WeisfeilerLehmanHashing", lambda k, g: k(g, 3, False, False),
             lambda instance, _g: instance.get_graph_features())),
        ("RandomWalker/ws300",
         "RandomWalker(walk_length=40, walk_number=10).do_walks(graph)", "ws300",
         obj("RandomWalker", lambda k, _g: k(40, 10),
             lambda instance, g: instance.do_walks(g))),
        ("BiasedRandomWalker/ws300",
         "BiasedRandomWalker(walk_length=40, walk_number=10, p=0.5, q=2.0)"
         ".do_walks(graph)", "ws300",
         obj("BiasedRandomWalker", lambda k, _g: k(40, 10, 0.5, 2.0),
             lambda instance, g: instance.do_walks(g))),
        ("EulerianDiffuser/karate",
         "EulerianDiffuser(diffusion_number=10, diffusion_cover=80)"
         ".do_diffusions(graph)", "karate",
         obj("EulerianDiffuser", lambda k, _g: k(10, 80),
             lambda instance, g: instance.do_diffusions(g))),
    ]


def read_upstream():
    """`(timings, repeats, environment)` from the upstream half's JSON.

    `bench/run.sh` exports `BENCH_UPSTREAM=failed` when the upstream step did
    not finish, and a JSON left over from an earlier run is not evidence for
    this one, so it is ignored rather than printed as if it were current. A
    missing file, a truncated one or one that is not a timings file mean the
    same thing, and the table prints without a left-hand column instead of
    dying.
    """
    if os.environ.get("BENCH_UPSTREAM") == "failed":
        print(f"note: the upstream step failed in this run, so {UPSTREAM} is "
              f"from an earlier run and is not used", file=sys.stderr)
        return {}, {}, {}
    try:
        payload = json.loads(UPSTREAM.read_text())
    except FileNotFoundError:
        print(f"note: {UPSTREAM} does not exist; the upstream step did not run",
              file=sys.stderr)
        return {}, {}, {}
    except (OSError, ValueError) as exc:
        print(f"note: {UPSTREAM} is unreadable ({exc}); no upstream column",
              file=sys.stderr)
        return {}, {}, {}
    if not isinstance(payload, dict) or "timings" not in payload:
        print(f"note: {UPSTREAM} is not a timings file written by "
              f"bench_upstream.py; no upstream column", file=sys.stderr)
        return {}, {}, {}
    return (payload["timings"], payload.get("repeats", {}),
            dict(payload.get("environment", {}),
                 measured=payload.get("measured", "unknown")))


def cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mojo_version():
    try:
        out = subprocess.run(["mojo", "--version"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or out.stderr.strip() or "unknown"


def seconds(value):
    """Seconds for the table: microsecond-scale cases keep four decimals."""
    if value < 1e-2:
        return f"{value * 1e3:.3f} ms"
    return f"{value:.4f} s"


def main():
    fixtures = load_fixture_module()
    upstream, upstream_repeats, upstream_env = read_upstream()
    cases = build_cases()
    rows = []
    failures = []
    for key, workload, fixture, build in cases:
        try:
            call = build()
            graph, _ = fixtures.build_graph(fixture)
            port_s, port_n = timed(lambda call=call, graph=graph: call(graph))
        except Exception:
            failures.append((key, traceback.format_exc()))
            rows.append((key, workload, upstream.get(key), None,
                         upstream_repeats.get(key), None))
            print(f"  {key:<32} FAILED", flush=True)
            continue
        rows.append((key, workload, upstream.get(key), port_s,
                     upstream_repeats.get(key), port_n))
        print(f"  {key:<32} {seconds(port_s):>12} (best of {port_n})", flush=True)

    missing = [key for key, _w, _f, _b in cases if key not in upstream]
    if missing:
        print(f"\nnote: no upstream timing for {', '.join(missing)}; the upstream "
              f"half did not finish those cases", file=sys.stderr)

    print()
    print(f"machine: {cpu_model()} | {os.cpu_count()} cores | "
          f"{mojo_version()} | "
          f"karateclub {upstream_env.get('karateclub', 'unavailable')} "
          f"(upstream env: Python {upstream_env.get('python', '?')}, "
          f"numpy {upstream_env.get('numpy', '?')}, "
          f"networkx {upstream_env.get('networkx', '?')}, "
          f"scipy {upstream_env.get('scipy', '?')}, measured "
          f"{upstream_env.get('measured', '?')}) | "
          f"this port: Python {platform.python_version()} "
          f"({sys.implementation.name})")
    print()
    print("| case | workload | karateclub 1.3.3 | mojo-karateclub | "
          "best-of (up/port) | speedup |")
    print("| --- | --- | --- | --- | --- | --- |")
    ratios = []
    for key, workload, up_s, port_s, up_n, port_n in rows:
        name, _, graph_name = key.partition("/")
        if up_s is None or port_s is None:
            up_cell = "n/a" if up_s is None else seconds(up_s)
            port_cell = "failed" if port_s is None else seconds(port_s)
            ratio_cell = "n/a"
        else:
            ratio = up_s / port_s
            ratios.append(ratio)
            up_cell, port_cell = seconds(up_s), seconds(port_s)
            ratio_cell = f"{ratio:.2f}x" + (" (slower)" if ratio < 1.0 else "")
        counts = f"{up_n if up_n is not None else '-'} / " \
                 f"{port_n if port_n is not None else '-'}"
        print(f"| {name} on {graph_name} | {workload} | {up_cell} | {port_cell} | "
              f"{counts} | {ratio_cell} |")

    print()
    if ratios:
        print(f"geomean speedup over the {len(ratios)} case(s) that ran on both "
              f"sides: {statistics.geometric_mean(ratios):.2f}x")
    else:
        print("no case ran on both sides, so there is no geomean")
    slower = sorted((up_s / port_s, key) for key, _w, up_s, port_s, _u, _p in rows
                    if up_s is not None and port_s is not None and up_s < port_s)
    if slower:
        print("slower than upstream: " + ", ".join(
            f"{key} ({ratio:.2f}x)" for ratio, key in slower))
    for key, tb in failures:
        print(f"\nmojo-karateclub {key} raised:\n{tb}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} case(s) raised; the rows above are what ran.",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
