# Parallel phase: where the time goes, and why only one loop was split

Toolchain: `mojo ==1.2.0.dev2026092605`, `max ==26.7.0.dev2026092605`
(`from max.algorithm import parallelize`, per `MOJO_NOTES.md` section 4).
Host: 72 cores, load average 24-50 throughout - a shared box, which matters for
every number below. All timings are best-of-N on this host, not means.

Numbers here are measured; none are estimated. The scripts that produced them
lived in a scratch directory and are not shipped - the commands are given so
each can be re-run.

## 1. What `parallelize` costs before it does any work

A kernel whose body is empty, dispatched through `parallelize`:

| trips | workers=1 | workers=2 | workers=4 |
| --- | --- | --- | --- |
| 34 | 0.002 ms | 6.88 ms | 9.04 ms |
| 128 | 0.001 ms | 5.89 ms | 8.18 ms |
| 300 | 0.001 ms | 8.08 ms | 7.38 ms |
| 1 000 | 0.001 ms | 7.24 ms | 6.25 ms |
| 10 000 | 0.001 ms | 5.85 ms | 5.04 ms |

The cost is **flat at about 6 ms and independent of the trip count**, and it does
not amortise: over 200 back-to-back calls on the same buffer the per-call cost
stays at 5.7-6.8 ms for `workers >= 2`. It is thread dispatch, not work.

So any loop has to be worth several milliseconds of serial time before splitting
it can possibly pay. A parallel path that costs 6 ms to save 2 ms is a
regression, and one that costs 6 ms to save 20 ms is a 1.3x win at best.

Thread scaling on a pure streaming loop (`x[i] *= a`), best-of-7:

| bytes | workers=1 | workers=2 | workers=4 | speedup @4 |
| --- | --- | --- | --- | --- |
| 8 MB | 0.97 ms | 7.10 ms | 6.67 ms | 0.15x |
| 128 MB | 18.94 ms | 17.26 ms | 20.25 ms | 0.94x |
| 512 MB | 80.98 ms | 57.50 ms | 51.00 ms | 1.59x |
| 1 GB | 190.65 ms | 142.16 ms | 117.99 ms | 1.62x |

Even at 1 GB the gain is 1.6x, not 4x: the loop is memory-bound and the box is
busy. That is the ceiling this phase was working against.

## 2. Where the port's time actually goes

Per estimator on the `ws300` fixture (n = 300), whole `fit` + `get_embedding`:

| estimator | ws300 |
| --- | --- |
| `FGSD` | 346 ms |
| `SF` | 343 ms |
| `NetLSD` | 382 ms |
| `LaplacianEigenmaps` | 350 ms (karate 1.4 ms) |
| `NodeSketch` | 3.19 ms |
| `FirstOrderLINE` | 1.78 ms |

Four of those five dominant estimators are the same kernel: the dense
`jacobi_eigh` eigensolver, `O(n^3)`. Split of one full sweep-limited call on a
dense SPD matrix:

| d | one sweep | `jacobi_eigh` (to `tol = 1e-24`) | sweeps used |
| --- | --- | --- | --- |
| 34 | 0.08 ms | 0.51 ms | 8 |
| 100 | 1.26 ms | 12.20 ms | 10 |
| 200 | 9.55 ms | 102.17 ms | 11 |
| 300 | 37.54 ms | 423.76 ms | 12 |

The sweep is the whole cost, and it is **sequential by construction**: each
rotation `(p, q)` reads and writes rows and columns `p` and `q`, and the next
rotation's `c, s` depend on cells the previous one wrote. The only way to run
two rotations at once is to apply a disjoint matching of pairs per pass, which
is a different algorithm (different rounding, different sweep counts) and not
"identical to the serial path" or "within the existing tolerance" for a subset
of the tests. It was not attempted.

`pinv_spectrum`'s `U diag(1/s) U.T` reconstruction is the one loop in the
dense path with genuinely independent iterations. Its cost alone, with an input
the sweep converges on immediately:

| d | workers=1 | workers=2 | workers=4 | speedup @4 |
| --- | --- | --- | --- | --- |
| 34 | 0.02 ms | 0.02 ms | 0.02 ms | 1.11x |
| 128 | 0.34 ms | 5.34 ms | 5.44 ms | 0.06x |
| 300 | 4.35 ms | 11.13 ms | 9.69 ms | 0.45x |
| 512 | 22.58 ms | 32.92 ms | 22.18 ms | 1.02x |
| 768 | 70.74 ms | 71.04 ms | 47.23 ms | 1.50x |
| 1024 | 162.28 ms | 144.86 ms | 96.29 ms | 1.69x |

The crossover is between d = 512 and d = 768. Below it the ~6 ms dispatch floor
dominates and the parallel branch is a 2-16x *regression*.

## 3. The decision

`pinv_spectrum` is split across rows, which is bit-identical (each output cell
is one `dot` over the same operands in the same order; `tests/test_linalg.py`
asserts `np.array_equal`). It is gated at `PINV_PARALLEL_ROWS = 700`, the
measured crossover plus margin.

That gate is above every input this port is benchmarked on. karateclub's graphs
are tens to a few hundred nodes; the port's own fixtures run to 300 for the
estimators (`ws300`) and its README example is a 300-node graph. So **the
parallel branch is dead code for every real caller measured here** - which is
the honest description, not a win. End-to-end `FGSD`, which is the only caller:

| fixture | n | w=1 | w=2 | w=4 | w=8 |
| --- | --- | --- | --- | --- | --- |
| karate | 34 | 0.50 ms | 0.51 ms | 0.50 ms | 0.50 ms |
| gnp60 | 60 | 2.72 ms | 2.68 ms | 2.62 ms | 2.63 ms |
| gnp200 | 200 | 84.84 ms | 86.31 ms | 85.23 ms | 84.78 ms |
| ws300 | 300 | 348.37 ms | 354.58 ms | 346.03 ms | 343.48 ms |

Identical output at every worker count (`np.array_equal` on the histogram), and
no speedup: the 1-2% spread is host noise on a machine whose load average moved
between 24 and 50 during these runs.

**What was measured and reverted:** switching the parallel branch on at
d = 128 (an earlier threshold guess, below the real crossover) made whole-model
`FGSD` *slower* at the one fixture big enough to reach it - 84.89 ms serial
against 94.90 ms at 4 workers and 108.40 ms at 2, i.e. 0.89x and 0.78x. That is
the "loses at the sizes real callers pass" case, so the threshold was moved up
to where the measurement puts it rather than left low.

**What was rejected outright:** the four `jacobi_eigh`-bound estimators, at 99%
of their run time, because the sweep is not data-parallel without changing the
result. The walkers and both LINE models likewise: they consume one MT19937
stream in walk order, and splitting them would change the draws. `_do_single_sketch`
and `do_a_recursion` have independent per-node/`node` loops but are 3.2 ms and
5.8 ms whole, below the dispatch floor. `NodeSketch` and `LINE` are 1.8-3.2 ms
whole and want more threads about as much as they want a 6 ms dispatcher.

## 4. Where a win would have to come from

Not from more workers. At 4 workers this host delivers ~1.6x on a streaming
kernel and ~1.0-1.1x on the reconstruction above, and a 6 ms floor per
`parallelize` call. The four dense estimators spend their time in a sequential
sweep; the fix there is a better *serial* algorithm (ARPACK-style sparse
eigenvalues, or a blocked sweep), not parallelism, and that is a different
phase.

## 5. Reproducing

Structural counts (nnz, sweeps, buffer bytes) for the flops/bytes columns:
`pixi run python bench/_analyze_counts.py`. Whole-model timings:
`pixi run bench`. The worker-count sweeps above were taken by calling
`kc_symmetric_pinv` / `kc_fgsd_embedding` directly with an explicit `workers`
argument, which is why both take one.

## 6. GPU phase: one kernel cleared the gate, and it still loses

Host: RTX 5090, 32607 MiB total, 4249-5677 MiB free across 80 samples on
2026-11-10 (a shared device; it stayed above 4000 MiB free at every sample, so
the phase went ahead). PCIe enumerated gen 3 x16; pinned copies measured
12.4 GB/s. `max.gpu.host.DeviceContext` and `max.gpu.block_idx`/`thread_idx`
both work on this toolchain, and `ctx.name()` reports the 5090.

### The gate: flops per byte, counted

`pixi run python bench/_gpu_gate.py` counts the operations from the kernels'
own loop structure and the buffers they touch, at the fixture sizes real callers
pass (sweep counts are the kernels' own return values, not a guess):

| kernel | n = 34 | n = 60 | n = 200 | n = 300 | n = 2000 |
| --- | --- | --- | --- | --- | --- |
| `jacobi_eigh` (dense sweep) | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 |
| `pinv_spectrum` `U D U.T` | 2.83 | 5.00 | 16.67 | 25.00 | 166.67 |
| WL hashing (MD5 per label) | 1.21 | 0.82 | 0.79 | 0.80 | 1.00 |

- **`jacobi_eigh` is 0.19 flops/byte: CPU only, memory-bound.** It is the
  whole of `FGSD`, `SF`, `NetLSD` and `LaplacianEigenmaps` - 99% of their run
  time, per section 2. Each rotation does 12 flops for 16 `float64` accesses,
  so however `O(n^3)` the rotation count looks, the kernel is a stream. It is
  also sequential (section 2), so there is no device work to find either.
- **WL hashing is 0.79-1.21 flops/byte: CPU only, memory-bound.**
- **`pinv_spectrum` is the only candidate above the line**, and it clears it
  only at the top of the range callers use: 2.83 at n = 34, 5.00 at 60, 16.67
  at 200, 25.00 at 300.

### What the device did with it

`bench/_gpu_pinv_model.mojo` is the prototype: one `max.gpu` kernel launch
mirroring `pinv_spectrum` exactly - one thread per upper-triangle cell, one
`dot` per cell, the tail handled by a `t < total` guard - with the two inputs
copied in once and the result copied out once per call. Its output is
bit-identical to the host reconstruction (`max |gpu - host| = 0.0`) at every
size below. `bench/_gpu_compare.py` times it against the serial CPU loop in
alternating passes (best of 3 passes x 5 calls a side), load average 34-41:

| d | CPU serial ms | GPU per call ms | CPU/GPU |
| --- | --- | --- | --- |
| 34 | 0.029 | 0.022 | 1.33x |
| 60 | 0.066 | 0.034 | 1.92x |
| 200 | 1.553 | 0.152 | 10.24x |
| 300 | 5.496 | 0.314 | 17.52x |

So the device is 17.5x faster than the *serial* CPU loop at d = 300. The kernel
alone is 0.13 ms of that 0.314 ms; the copies, one launch and one sync are the
other 58%.

### Why it was not merged

The device wins against one CPU thread. It does not win against the CPU path
that exists:

- **The reconstruction is 1.5-3.1% of the call it sits in.** Whole `FGSD` on
  `ws300` is 374.9 ms (`bench/_gpu_footprint.py`); the 300-row reconstruction
  is 5.5-11.8 ms of it. At 0.314 ms on the device that saves at most ~11.5 ms,
  a 3% whole-model win, and only if nothing else moves. At every smaller
  fixture the saving is a fraction of a millisecond out of a 0.5-85 ms call.
- **The Jacobi sweep, which is the other 97%, is not a candidate at all** -
  0.19 flops/byte and sequential. Putting it on the device means a disjoint
  matching per pass and different rounding, which the same-results rule rules
  out.
- **Only one benchmarked caller reaches a size where the win is measurable**
  (`ws300` at 300 nodes), and the device set there is 2.1 MiB against a 3.4 MiB
  host working set - the transfer is all latency, no bandwidth problem.

**Reverted.** No GPU code is in `src/`, and nothing in `pixi run test` touches a
device: the suite is CPU-only by construction rather than by a skip marker,
because there is no GPU path to test. The prototype and the measurement scripts
stay in `bench/` so the numbers above can be re-run:

```bash
pixi run python bench/_gpu_gate.py            # the flops/byte table
pixi run python bench/_gpu_compare.py 3       # CPU vs device, interleaved
pixi run mojo run bench/_gpu_pinv_model.mojo  # device only, with the check
pixi run python bench/_gpu_footprint.py       # buffer sizes, reconstruction share
```

**One design problem that would have blocked a merge anyway.** `symmetric_pinv`
writes the reconstruction into `a`, the caller's own `float64` array, and knows
nothing about where that memory lives. A device path needs device-resident
buffers plus a copy back into that host array, so it cannot be a branch inside
the kernel: it has to be a decision the Python layer makes before the call -
a different interface for one kernel, to save 3%.
