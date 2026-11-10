"""One timed case, one process, so the process-wide counters belong to it alone.

Read-only with respect to src/ and tests/: it imports the package the way
bench/bench.py does and times the same callables. Run by bench/_analyze_driver.py
under the machine lock.

    python bench/_analyze_worker.py <json-case>

Prints one JSON object: best-of-N seconds for each level of the call, the
process's own bytes/RSS delta over the timed region, and (optionally) a
cProfile dump of the best sample, so ctypes glue can be separated from kernel.
"""
import cProfile
import importlib.util
import io
import json
import pathlib
import pstats
import sys
import time
import tracemalloc

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

GRAPHS = ROOT / "tests" / "_graphs.py"


def fixtures():
    spec = importlib.util.spec_from_file_location("kc_fixtures", GRAPHS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def our_rss():
    """This process's resident bytes, from /proc: the RSS column of statm."""
    return int(pathlib.Path("/proc/self/statm").read_text().split()[1]) * 4096


def other_rss():
    """Resident bytes of everything except this process: other agents' jobs and
    the OS cache are in here, which is why memory numbers prefer tracemalloc's
    numpy view when it is available."""
    own = int(pathlib.Path("/proc/self/statm").read_text().split()[1]) * 4096
    total = 0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == str(pathlib.Path("/proc/self").resolve().name):
            continue
        try:
            fields = (entry / "statm").read_text().split()
            total += int(fields[1]) * 4096
        except (OSError, IndexError, ValueError):
            continue
    return max(total, 0), own


def best_of(fn, repeats):
    start = time.perf_counter()
    fn()
    best = time.perf_counter() - start
    for _ in range(repeats - 1):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    case = json.loads(sys.argv[1])
    fx = fixtures()
    import mojokarateclub as mk

    name = case["klass"]
    graph_name = case["graph"]
    graph_level = case.get("graph_level", False)
    kwargs = case["kwargs"]
    # A fresh graph per case, built outside the timed region, exactly as bench.py.
    graph, n = fx.build_graph(graph_name)
    klass = getattr(mk, name)
    repeats = case.get("repeats", 3)

    out = {"case": case["key"], "n": n}

    if case.get("kind") == "estimator":
        def construct():
            return klass(**kwargs)

        def fit(model):
            if graph_level:
                model.fit([graph])
            else:
                model.fit(graph)

        if case.get("split"):
            # Constructor / fit / get_embedding timed separately on one reused
            # graph; fit mutates the graph only on the first call but its
            # _check_graph pass is the same each time, and best-of-N drops the
            # first, slower one, as the harness does.
            construct()
            tracemalloc.start()
            before = tracemalloc.take_snapshot()
            out["construct_s"] = best_of(construct, repeats)
            model = klass(**kwargs)
            out["fit_s"] = best_of(lambda: fit(model), repeats)
            model = klass(**kwargs)
            fit(model)
            out["get_embedding_s"] = best_of(model.get_embedding, repeats)
            after = tracemalloc.take_snapshot()
            stats = after.compare_to(before, "filename")
            out["traced_bytes_top"] = [
                [str(s.traceback[0].filename), str(s.traceback[0].lineno), int(s.size_diff)]
                for s in stats[:12]
            ]
            out["traced_total_bytes"] = int(sum(s.size_diff for s in stats))
            tracemalloc.stop()

        def whole():
            model = construct()
            fit(model)
            model.get_embedding()

        out["whole_s"] = best_of(whole, repeats)

        prof = cProfile.Profile()
        buf = io.StringIO()
        prof.enable()
        whole()
        prof.disable()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        out["profile"] = buf.getvalue().splitlines()
    else:
        construct = case["construct"]

        def whole():
            if construct == "wl":
                instance = klass(graph, kwargs["wl_iterations"],
                                 kwargs["attributed"], kwargs["erase_base_features"])
                return instance.get_graph_features()
            if construct == "walker":
                instance = klass(**kwargs)
                return instance.do_walks(graph)
            if construct == "diffuser":
                instance = klass(**kwargs)
                return instance.do_diffusions(graph)
            raise SystemExit(f"unknown construct {construct}")

        out["whole_s"] = best_of(whole, repeats)
        prof = cProfile.Profile()
        buf = io.StringIO()
        prof.enable()
        whole()
        prof.disable()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        out["profile"] = buf.getvalue().splitlines()

    rss_before = our_rss()
    whole()
    out["rss_delta_bytes"] = our_rss() - rss_before

    # syscall counts for this one call, to price the ctypes crossings.
    if case.get("getrusage"):
        import resource
        before = resource.getrusage(resource.RUSAGE_SELF)
        whole()
        after = resource.getrusage(resource.RUSAGE_SELF)
        out["rusage"] = {
            "utime": after.ru_utime - before.ru_utime,
            "stime": after.ru_stime - before.ru_stime,
            "minflt": after.ru_minflt - before.ru_minflt,
            "majflt": after.ru_majflt - before.ru_majflt,
            "inblock": after.ru_inblock - before.ru_inblock,
        }

    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
