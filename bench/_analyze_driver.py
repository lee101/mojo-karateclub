"""Driver for bench/_analyze_worker.py: the cases bench/bench.py times, plus
per-call syscall counts, each in its own process.

    pixi run bench   # the canonical table; then
    flock ... python bench/_analyze_driver.py   # this, for the ANALYZE map

The worker is the source of every number: this file only assembles the cases
and collects the JSON. It writes bench/_analyze_raw.json.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "bench" / "_analyze_worker.py"
OUT = ROOT / "bench" / "_analyze_raw.json"


def cases():
    graph_level = True
    node_level = False
    line = {"dimensions": 8, "epochs": 5, "mini_batch_size": 64, "verbose": False}
    est = lambda key, klass, graph, kwargs, gl, split=False: {
        "key": key, "kind": "estimator", "klass": klass, "graph": graph,
        "kwargs": kwargs, "graph_level": gl, "split": split, "getrusage": True,
    }
    return [
        est("LDP/karate", "LDP", "karate", {"bins": 32}, graph_level, True),
        est("LDP/ws300", "LDP", "ws300", {"bins": 32}, graph_level, True),
        est("LDP/ba2000", "LDP", "ba2000", {"bins": 32}, graph_level, True),
        est("FGSD/karate", "FGSD", "karate", {}, graph_level, True),
        est("FGSD/ws300", "FGSD", "ws300", {}, graph_level, True),
        est("SF/karate", "SF", "karate", {"dimensions": 16}, graph_level, True),
        est("SF/ws300", "SF", "ws300", {"dimensions": 32}, graph_level, True),
        est("NetLSD/karate", "NetLSD", "karate",
            {"scale_steps": 32, "approximations": 8}, graph_level, True),
        est("NetLSD/ws300", "NetLSD", "ws300",
            {"scale_steps": 32, "approximations": 16}, graph_level, True),
        est("LaplacianEigenmaps/karate", "LaplacianEigenmaps", "karate",
            {"dimensions": 8}, node_level, True),
        est("LaplacianEigenmaps/ws300", "LaplacianEigenmaps", "ws300",
            {"dimensions": 8}, node_level, True),
        est("NodeSketch/karate", "NodeSketch", "karate", {"dimensions": 16},
            node_level, True),
        est("NodeSketch/ws300", "NodeSketch", "ws300", {"dimensions": 16},
            node_level, True),
        est("FirstOrderLINE/karate", "FirstOrderLINE", "karate", line,
            node_level, True),
        est("FirstOrderLINE/ws300", "FirstOrderLINE", "ws300", line,
            node_level, True),
        est("SecondOrderLINE/karate", "SecondOrderLINE", "karate", line,
            node_level, True),
        est("SecondOrderLINE/ws300", "SecondOrderLINE", "ws300", line,
            node_level, True),
        {"key": "WeisfeilerLehmanHashing/ws300", "kind": "obj", "klass":
         "WeisfeilerLehmanHashing", "construct": "wl", "graph": "ws300",
         "kwargs": {"wl_iterations": 3, "attributed": False,
                    "erase_base_features": False}, "getrusage": True},
        {"key": "RandomWalker/ws300", "kind": "obj", "klass": "RandomWalker",
         "construct": "walker", "graph": "ws300",
         "kwargs": {"walk_length": 40, "walk_number": 10}, "getrusage": True},
        {"key": "BiasedRandomWalker/ws300", "kind": "obj", "klass":
         "BiasedRandomWalker", "construct": "walker", "graph": "ws300",
         "kwargs": {"walk_length": 40, "walk_number": 10, "p": 0.5, "q": 2.0},
         "getrusage": True},
        {"key": "EulerianDiffuser/karate", "kind": "obj", "klass":
         "EulerianDiffuser", "construct": "diffuser", "graph": "karate",
         "kwargs": {"diffusion_number": 10, "diffusion_cover": 80},
         "getrusage": True},
        # The same entry points at a larger size, so scaling can be read off
        # rather than guessed: karateclub callers do pass graphs this big.
        est("LDP/ba2000_bins64", "LDP", "ba2000", {"bins": 64}, graph_level),
        est("FGSD/ba2000", "FGSD", "ba2000", {}, graph_level),
        est("SF/ba2000", "SF", "ba2000", {"dimensions": 32}, graph_level),
        est("NetLSD/ba2000", "NetLSD", "ba2000",
            {"scale_steps": 32, "approximations": 16}, graph_level),
        est("LaplacianEigenmaps/ba2000", "LaplacianEigenmaps", "ba2000",
            {"dimensions": 8}, node_level, True),
        est("NodeSketch/ba2000", "NodeSketch", "ba2000", {"dimensions": 16},
            node_level, True),
        est("FirstOrderLINE/ba2000", "FirstOrderLINE", "ba2000", line,
            node_level, True),
        est("SecondOrderLINE/ba2000", "SecondOrderLINE", "ba2000", line,
            node_level, True),
        {"key": "WeisfeilerLehmanHashing/ba2000", "kind": "obj", "klass":
         "WeisfeilerLehmanHashing", "construct": "wl", "graph": "ba2000",
         "kwargs": {"wl_iterations": 3, "attributed": False,
                    "erase_base_features": False}, "getrusage": True},
        {"key": "RandomWalker/ba2000", "kind": "obj", "klass": "RandomWalker",
         "construct": "walker", "graph": "ba2000",
         "kwargs": {"walk_length": 40, "walk_number": 10}, "getrusage": True},
        {"key": "BiasedRandomWalker/ba2000", "kind": "obj", "klass":
         "BiasedRandomWalker", "construct": "walker", "graph": "ba2000",
         "kwargs": {"walk_length": 40, "walk_number": 10, "p": 0.5, "q": 2.0},
         "getrusage": True},
    ]


def main(argv):
    selected = set(argv[1:])
    todo = [c for c in cases() if not selected or c["key"] in selected]
    results = {}
    if OUT.exists():
        results = {r["case"]: r for r in json.loads(OUT.read_text())}
    for case in todo:
        proc = subprocess.run([sys.executable, str(WORKER), json.dumps(case)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{case['key']:<32} FAILED\n{proc.stderr[-3000:]}", flush=True)
            results[case["key"]] = {"case": case["key"], "failed": proc.stderr[-3000:]}
            continue
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        record["case_json"] = case
        results[case["key"]] = record
        print(f"{case['key']:<32} whole {record.get('whole_s', 0) * 1e3:>12.4f} ms",
              flush=True)
    OUT.write_text(json.dumps(list(results.values()), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv)
