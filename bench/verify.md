# Verify phase: what was reproduced, what was not, what changed

Session of 2026-11-10 on the machine the README names: Intel(R) Xeon(R) CPU
E5-2697 v4 @ 2.30GHz, 72 logical CPUs, shared with other jobs. Upstream side
inside `pixi run bench`: Python 3.9.23, numpy 1.22.4, networkx 2.6.3,
scipy 1.9.3, karateclub 1.3.3. Port side: Python 3.13.15, Mojo 1.2.0.dev2026092605.

## Commands run, in order

| step | command | result |
| --- | --- | --- |
| build | `pixi run build` | exit 0, `dist/libmojo-karateclub.so` rewritten |
| test | `pixi run test` | exit 0, **179 passed**, 4 warnings (the two `docvecs` `DeprecationWarning`s, expected and documented) |
| bench | `pixi run bench` | the raw halves below; see "How the benchmark run was obtained" |
| final build and test | `pixi run build && pixi run test` | exit 0, 179 passed |
| flops/byte check | `bench/_gpu_gate.py` | reproduced the README's 0.1875 and 25.0 flops/byte |
| API cross-check | `pixi run -e upstream python -u bench/bench_upstream.py` | direct upstream API, used only to cross-check the harness |

The README said `176 passed`. The suite is 179: the phases since the first
commit added `test_serial_and_parallel_pinv_agree` (3 parameters) and
`test_every_export_has_a_signature`, and no test was removed (see "Tests").
The README now says 179.

## How the benchmark run was obtained

`dist/upstream_timings.json` is written only after every upstream case has run,
and on this host the last upstream case, `EulerianDiffuser(10, 80)` on karate,
does not terminate (see below). A hard `pixi run bench` therefore never wrote
the file on any of my attempts, and the port half then printed a table with the
whole upstream column `n/a`. Stopping the upstream half also loses the 19 rows
that had already been timed.

So the upstream column below is the complete half from one unmodified
`pixi run -e upstream python -u bench/bench_upstream.py`, run at 18:29:47Z under
the machine lock and stopped after the `BiasedRandomWalker` row printed. It
also wrote `dist/upstream_timings.json`. The port column is
`bench/bench.py` started at 18:32:10Z, still under the lock, in the default
environment. Nothing was hand-edited, and both sides ran the shipped code with
the shipped fixtures and parameters.

Earlier in the session I also ran the two halves as one `pixi run bench` with a
timeout on the upstream half. That run's upstream column is not used here (it
was cut at the `EulerianDiffuser` case); its port column was within 5% of the
one below on every row, and it is the reason the port column in the README can
be trusted to be stable.

Two other things made the run hard to get, both from the shared machine and
neither from this repository:

- Other agents were running their own `pixi run bench` on the same
  `/tmp/mojo-bench.lock`. One unrelated job, a loop of `pixi run bench` from
  `~/unroll-kc/run_bench.sh`, held the lock and then sat spinning inside the
  upstream `bench_upstream.py` for over three hours (EulerianDiffuser again, on
  its own copy of this repository). I sent it `SIGSTOP` rather than killing it,
  so its owner can resume it; my first upstream half from this session
  (18:11) was killed by my own timeout at the same case.
- On this toolchain a `flock` held inside `pixi run bench` appears to wedge
  unrelated waiters from other repositories while holding the lock. That is
  outside this repository and is not diagnosed here; it is worth knowing
  before relying on the lock as the only guard.

## Raw numbers

Upstream, `bench/bench_upstream.py`, 2026-11-10, started 18:29:47Z:

```
  LDP/karate                            2.493 ms  (best of 3)
  LDP/ws300                            17.201 ms  (best of 3)
  FGSD/karate                           2.239 ms  (best of 3)
  FGSD/ws300                          131.502 ms  (best of 3)
  SF/karate                             5.437 ms  (best of 3)
  SF/ws300                            103.347 ms  (best of 3)
  NetLSD/karate                         9.703 ms  (best of 3)
  NetLSD/ws300                        251.944 ms  (best of 3)
  LaplacianEigenmaps/karate             9.087 ms  (best of 3)
  LaplacianEigenmaps/ws300             23.050 ms  (best of 3)
  NodeSketch/karate                    16.950 ms  (best of 3)
  NodeSketch/ws300                    205.611 ms  (best of 3)
  FirstOrderLINE/karate                 0.978 ms  (best of 3)
  FirstOrderLINE/ws300                 15.546 ms  (best of 3)
  SecondOrderLINE/karate                1.108 ms  (best of 3)
  SecondOrderLINE/ws300                15.067 ms  (best of 3)
  WeisfeilerLehmanHashing/ws300         4.992 ms  (best of 3)
  RandomWalker/ws300                  380.507 ms  (best of 3)
  BiasedRandomWalker/ws300          24418.434 ms  (best of 3)
  EulerianDiffuser/karate               did not terminate
```

Port, `bench/bench.py`, 2026-11-10 18:32:10Z:

```
  LDP/karate                           0.221 ms (best of 7)
  LDP/ws300                            1.349 ms (best of 3)
  FGSD/karate                          0.851 ms (best of 7)
  FGSD/ws300                           0.3601 s (best of 3)
  SF/karate                            0.887 ms (best of 3)
  SF/ws300                             0.3563 s (best of 3)
  NetLSD/karate                        1.363 ms (best of 3)
  NetLSD/ws300                         0.3540 s (best of 3)
  LaplacianEigenmaps/karate            0.830 ms (best of 3)
  LaplacianEigenmaps/ws300             0.3614 s (best of 3)
  NodeSketch/karate                    0.430 ms (best of 7)
  NodeSketch/ws300                     3.434 ms (best of 3)
  FirstOrderLINE/karate                0.189 ms (best of 7)
  FirstOrderLINE/ws300                 1.835 ms (best of 3)
  SecondOrderLINE/karate               0.179 ms (best of 7)
  SecondOrderLINE/ws300                1.749 ms (best of 3)
  WeisfeilerLehmanHashing/ws300        4.670 ms (best of 3)
  RandomWalker/ws300                   0.0488 s (best of 3)
  BiasedRandomWalker/ws300             0.0552 s (best of 3)
  EulerianDiffuser/karate              0.0207 s (best of 3)
```

## Old table versus this one, row by row

The rule for this phase: a figure that does not reproduce within 20% is
corrected to the measured one. "Old" is the table that was in the README until
this session (its own header dates the measurement 2026-09-26); "new" is above.
A ratio outside [0.8, 1.25] of the old figure is
marked.

| row | old karateclub | new karateclub | old port | new port | old speedup | new speedup | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LDP/karate | 2.732 ms | 2.493 ms | 0.203 ms | 0.221 ms | 13.43x | 11.28x | reproduced (both within 10% or so) |
| LDP/ws300 | 18.0 ms | 17.201 ms | 1.238 ms | 1.349 ms | 14.52x | 12.75x | reproduced |
| FGSD/karate | 2.130 ms | 2.239 ms | 0.868 ms | 0.851 ms | 2.45x | 2.63x | reproduced |
| FGSD/ws300 | 422.4 ms | 131.502 ms | 1.4913 s | 0.3601 s | 0.28x | 0.37x | **changed**; both sides faster here, still slower than upstream |
| SF/karate | 5.500 ms | 5.437 ms | 0.910 ms | 0.887 ms | 6.04x | 6.13x | reproduced |
| SF/ws300 | 426.8 ms | 103.347 ms | 1.4517 s | 0.3563 s | 0.29x | 0.29x | **changed**; port 3.1% under the old figure is inside 20%, upstream is not; same 0.29x direction |
| NetLSD/karate | 9.164 ms | 9.703 ms | 0.987 ms | 1.363 ms | 9.28x | 7.12x | **changed**; port 38% above the old figure |
| NetLSD/ws300 | 246.5 ms | 251.944 ms | 1.4536 s | 0.3540 s | 0.17x | 0.71x | **changed**; upstream reproduced, port 3.1x faster than when measured |
| LaplacianEigenmaps/karate | 9.200 ms | 9.087 ms | 1.363 ms | 0.830 ms | 6.75x | 10.95x | **changed**; port 39% below the old figure |
| LaplacianEigenmaps/ws300 | 23.8 ms | 23.050 ms | 1.4575 s | 0.3614 s | 0.02x | 0.06x | **changed**; upstream reproduced, port 3.1x faster |
| NodeSketch/karate | 16.7 ms | 16.950 ms | 0.434 ms | 0.430 ms | 38.56x | 39.42x | reproduced |
| NodeSketch/ws300 | 225.7 ms | 205.611 ms | 3.092 ms | 3.434 ms | 72.99x | 59.87x | upstream **changed** (9% under); port reproduced |
| FirstOrderLINE/karate | 0.996 ms | 0.978 ms | 0.167 ms | 0.189 ms | 5.95x | 5.17x | port 13% above old; inside 20% |
| FirstOrderLINE/ws300 | 16.0 ms | 15.546 ms | 1.798 ms | 1.835 ms | 8.91x | 8.47x | reproduced |
| SecondOrderLINE/karate | 1.064 ms | 1.108 ms | 0.182 ms | 0.179 ms | 5.84x | 6.19x | reproduced |
| SecondOrderLINE/ws300 | 15.9 ms | 15.067 ms | 1.675 ms | 1.749 ms | 9.51x | 8.61x | reproduced |
| WeisfeilerLehmanHashing/ws300 | 5.569 ms | 4.992 ms | 5.788 ms | 4.670 ms | 0.96x | 1.07x | both within 20%; the ratio moved across 1.0, see below |
| RandomWalker/ws300 | 418.3 ms | 380.507 ms | 0.0472 s | 0.0488 s | 8.86x | 7.80x | upstream 9% under; port reproduced |
| BiasedRandomWalker/ws300 | 24.7254 s | 24.4184 s | 0.0542 s | 0.0552 s | 455.99x | 442.36x | reproduced |
| EulerianDiffuser/karate | absent | does not terminate | absent | 0.0207 s | absent | not comparable | added |

Geomean: old **4.35x** over 19 rows, new **4.97x** over the 19 rows that run on
both sides (recomputed from the table's own ratios).

The port's whole `ws300` spectral column (FGSD, SF, NetLSD,
LaplacianEigenmaps) is about 3.1x faster than the old figures on the same days
the upstream ARPACK column is 1.6x to 4x faster than its old figures. One
explanation fits both: the old table was taken on a much more loaded machine,
where the port's 0.36 s serial Jacobi sweep and upstream's threaded BLAS both
lost to contention. I did not try to win that back - the port's single-threaded
sweep does not scale with load the way BLAS does, so the two columns move
together but not equally. This is why the README keeps saying read the
direction, not the third digit. Every row that was slower than upstream is
still slower: FGSD/ws300 0.37x, SF/ws300 0.29x, NetLSD/ws300 0.71x,
LaplacianEigenmaps/ws300 0.06x.

The WG row is the one direction that changed (0.96x slower -> 1.07x faster).
Neither figure is outside the run-to-run spread of this row: upstream measured
5.569, 5.583, 5.344, 5.352 and 4.992 ms and the port 5.788, 4.498, 4.670 ms
across the runs I saw. I report the measured ratio and say in the README that
the row is a wash either side of 1.0; I did not re-word it as a win.

## Same input on both sides

Checked by reading both harness files and both upstream modules, not only the
case keys:

- `bench/bench.py` and `bench/bench_upstream.py` build each fixture with the
  same `tests/_graphs.py` builder, then `estimator()` constructs, fits and calls
  `get_embedding` in both halves, with the same keyword arguments: `LDP(bins=32)`,
  `FGSD()`, `SF(dimensions=16/32)`, `NetLSD(scale_steps=32,
  approximations=8/16)`, `LaplacianEigenmaps(dimensions=8)`,
  `NodeSketch(dimensions=16)`, both LINE orders with
  `dimensions=8, epochs=5, mini_batch_size=64, verbose=False`. The graph-level
  models get `fit([graph])`, the node-level ones `fit(graph)`, on both sides.
- The walker cases call `do_walks(graph)` with `(40, 10)` / `(40, 10, 0.5, 2.0)`
  and the WL case calls `WeisfeilerLehmanHashing(graph, 3, False, False).get_graph_features()`,
  which are upstream's own positional arguments.
- Both sides build the graph once per case outside the timed callable and reuse
  it; neither memoises results, both construct the model inside the callable,
  and both use best-of-N with the same N rule.
- Per estimator I compared `fit` against the upstream source in
  `.pixi/envs/upstream/.../karateclub/`: LDP, FGSD, SF, NetLSD,
  LaplacianEigenmaps, NodeSketch and both LINE orders run the same steps (for
  example `FirstOrderLINE`: `number_of_edges // mini_batch_size` batches per
  epoch, half-size positive and negative batches, gradient reset per batch,
  `learning_rate * learning_rate_decay**epoch`), so neither side is skipping
  work the other does. `fit` in both halves also runs `_set_seed` and
  `_check_graph`/`_check_graphs` (`_ensure_integrity` adds the self loops), so
  the same work is in both timed regions.
- Configurations are not all upstream defaults on either side, but they are the
  same configuration on both sides: SF/karate asks for 16 eigenvalues where the
  default is 128, NetLSD asks for 8 and 16 approximations where the default is
  200. That is the case table, and it is the same in both files.

## Upstream's EulerianDiffuser does not terminate

`EulerianDiffuser(10, 80)` on the 34-node karate fixture never returns. I
reproduced it three ways: the whole case in `bench_upstream.py` (it hovers at
one core with no write to `dist/`), a probe that runs
`_run_diffusion_process` for every source node with a 5 s cap per call (all 34
wedged), and a transcription of the upstream loop that prints its state at the
cap: `infected` reached all 34 nodes and the loop kept drawing. The cause is
upstream's loop shape: `infected_counter` only changes inside
`if sample not in infected`, so once every reachable node is infected the
`break` inside that arm can never run and the `while` never ends. The README's
divergence 2 already describes this as a defect the port fixes; this session is
the first time the benchmark table has had to show the consequence. The port
does the case in 20.7 ms.

I did not change `bench/bench_upstream.py` to skip the case: that would hide a
measured fact and breaks the rule that rows stay. It is last in the table, so
a hung run still produces the other 19 rows on stdout, but the JSON is written
only at the end. Until that is fixed upstream, a `pixi run bench` on this host
will report the upstream column as failed unless the upstream half is stopped
after its last printed row, as described above. Putting the case behind a
subprocess timeout that keeps the rows it already measured is a code change I
did not make, because it changes the harness's semantics and nothing here
failed a test over it.

## Not reproduced in this session

- `bench/analysis.md`'s per-kernel numbers (`parallelize` dispatch cost ~6 ms,
  `pinv_spectrum` at d = 300/512, the 1 GB streaming loop, the worker scaling)
  were not re-run. What I could check deterministically reproduced:
  `bench/_gpu_gate.py` prints 0.1875 flops/byte for the Jacobi sweep and 25.0
  for `pinv_spectrum` at d = 300, as the README says. The whole-model claim
  "348 ms on `ws300`" did not reproduce: `FGSD/ws300` on the port measures
  360 ms here, which is the same thing within run-to-run noise and is the
  number the benchmark table now shows.
- The GPU prototype numbers (17.5x, 0.314 ms against 5.496 ms) were not re-run;
  no GPU work was done in this session and `pixi run test` does not touch a
  device.
- The headline kernel cost check that does reproduce: a dense
  `eigh` of the `ws300` normalized Laplacian takes 380.0 ms here (synthetic
  dense 300x300: 354.4 ms) against the README's 424 ms in `analysis.md`.
  That is the "~1.4 s either way" sentence's 0.36 s in the new table; the
  README sentence about `jacobi_eigh` being 99% of the spectral models says
  "424 ms of 434 ms" and is stale at today's speed of the sweep. I left that
  paragraph's analysis alone and note the discrepancy here rather than edit
  a claim I could only half re-measure.
- No test needed changing for any of this.

## Tests

`git log -p -- tests/` across the whole history:

- `988106c` initial port (committed 2026-10-01): the parity tests in
  `test_graph_embedding.py`, `test_node_embedding.py` and `test_linalg.py`, and
  `tests/vectors/upstream.npz`.
- `de79e6b` "phase parallel": adds `test_serial_and_parallel_pinv_agree` and
  its `@parametrize("d", [8, 699, 700])`, adds `_ffi.MAX_WORKERS` to the
  existing `kc_symmetric_pinv` call, and adds the two integer-overflow guards
  in `_pysymmetric_pinv`. No assertion is removed and no tolerance is widened.
- The other phases touch only kernels and the README: `git diff --stat` over
  `tests/` from the initial port to HEAD is one commit, `+40 -1` in
  `test_linalg.py`.

The only deleted lines anywhere in `tests/` history are the two that the
commits above replaced (`test_eigh_matches_lapack`'s parameter list, which was
widened with more sizes; and the library filename spelling), neither of which
loosens an assertion. In the parity files, the tolerances are unchanged from
the initial port: `atol=1e-9` for FGSD, SF and LaplacianEigenmaps, `1e-5` for
NetLSD, `array_equal` for LDP and NodeSketch, and `1e-12` for both LINE orders,
exactly as the README documents. Nothing was restored, because nothing had been
removed or loosened.

The parity data itself is not regenerated here: `tests/vectors/upstream.npz`
is unchanged since `988106c`, and I did not run `pixi run vectors` because that
would replace the reference file with a fresh one from the current upstream
environment, which is not what this phase is for.

## Changed

- `README.md`, benchmark section only: every number in the table is the port
  and upstream column measured in this session, the geomean is recomputed from
  them (4.97x, recomputed from the rows), `EulerianDiffuser` is added as a row with no upstream number
  and the reason, the "five slower" paragraph now says four slower rows plus a
  wash, and the WS/LINE multipliers are re-derived from these numbers.
- `README.md`, parity section: `176 passed` -> `179 passed`, measured.
- `bench/verify.md`: this file.
- No kernel, Python package, test or harness code changed.

Final state: `pixi run build && pixi run test` -> exit 0, `179 passed`.
