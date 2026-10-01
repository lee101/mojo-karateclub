#!/usr/bin/env bash
# Time the real karateclub, then this port, and print the comparison table.
#
#   pixi run bench          # what the pixi task runs, holding the machine lock
#   bash bench/run.sh       # same thing: re-enters `pixi run bench` when the
#                           # default environment is not already active
#
# The two halves cannot share an interpreter. Upstream karateclub 1.3.3 needs
# the `upstream` environment's numpy 1.22 / networkx 2.6 / Python 3.9; the port
# needs the default environment's Mojo toolchain and numpy 2. So this is two
# processes that meet at dist/upstream_timings.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TIMINGS="dist/upstream_timings.json"

if [ "${PIXI_IN_SHELL:-0}" != "1" ] && [ "${BENCH_IN_PIXI:-0}" != "1" ]; then
    # Launched outside `pixi run`, so `python` on PATH is not the environment
    # the port is built against. Go back in through the task that owns the
    # machine-wide lock, and mark the re-entry so this cannot recurse.
    export BENCH_IN_PIXI=1
    exec pixi run bench
fi

echo "upstream karateclub 1.3.3 (pixi env: upstream, Python 3.9)"
# Assume failure until the step says otherwise: a $TIMINGS left behind by an
# earlier run must not be printed as if it had been measured just now.
export BENCH_UPSTREAM=failed
if pixi run -e upstream python bench/bench_upstream.py; then
    export BENCH_UPSTREAM=ok
    echo
else
    status=$?
    echo
    echo "upstream karateclub timings unavailable: the upstream step exited" \
         "$status (see the traceback above)." 1>&2
    echo "The karateclub column of the table below is left empty rather than" \
         "filled from the last run's $TIMINGS; the rows are still timed and" \
         "reported." 1>&2
    echo
fi

echo "mojo-karateclub (pixi env: default, Mojo toolchain)"
python bench/bench.py
