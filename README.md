# mojo-karateclub

The compute-bound half of
[karateclub](https://github.com/benedekrozemberczki/karateclub) 1.3.3, in Mojo.

karateclub's estimators are short: a graph goes in, a few lines of numpy or
scipy go by, and an embedding comes out. What is worth rewriting is the work
inside those few lines - the degree profiles, the pseudo-inverse of a normalized
Laplacian, the count-min sketches, the second-order walk probabilities, the MD5
of every Weisfeiler-Lehman label. That work is here, as Mojo kernels, behind
the same class names, the same constructor arguments in the same order, and the
same numerical results.

The estimators themselves keep upstream's shape, so this is a drop-in for the
covered subset:

```python
import networkx as nx
import mojokarateclub

graph = nx.karate_club_graph()
model = mojokarateclub.LDP(bins=32)
model.fit([graph])
model.get_embedding().shape      # (1, 160)
```

## Install and run

Linux x86-64, and the Mojo toolchain pinned in `pixi.toml`.

```bash
pixi install
pixi run build      # mojo build --emit shared-lib -> dist/libmojo-karateclub.so
pixi run test       # pytest
pixi run vectors    # regenerate the upstream reference vectors
pixi run bench      # benchmark against the real karateclub
```

`pixi install` creates two environments. `default` has the Mojo toolchain,
Python 3.13 and this package. `upstream` has Python 3.9 and the real
karateclub 1.3.3, and exists only so the parity tests and the benchmark have a
genuine reference to compare against; it is a separate pixi environment
because karateclub pins `numpy<1.23`, `networkx<2.7` and `pandas<=1.3.5`, none
of which build on 3.13.

## Usage

Every estimator takes a networkx graph, exactly as upstream does, and the
self-loop pass `Estimator._ensure_integrity` runs before any model sees the
graph is upstream's.

```python
import networkx as nx
import mojokarateclub

graph = nx.newman_watts_strogatz_graph(300, 4, 0.1, seed=42)

# graph-level, fit over a list of graphs
for model in (
    mojokarateclub.LDP(bins=32),
    mojokarateclub.FGSD(hist_bins=200, hist_range=20),
    mojokarateclub.SF(dimensions=32),
    mojokarateclub.NetLSD(scale_steps=32, approximations=16),
    mojokarateclub.Graph2Vec(wl_iterations=2, dimensions=64, workers=1),
):
    model.fit([graph])
    print(type(model).__name__, model.get_embedding().shape)

# node-level, fit over one graph
for model in (
    mojokarateclub.LaplacianEigenmaps(dimensions=8),
    mojokarateclub.NodeSketch(dimensions=16, iterations=2),
    mojokarateclub.FirstOrderLINE(dimensions=8, epochs=5, verbose=False),
    mojokarateclub.SecondOrderLINE(dimensions=8, epochs=5, verbose=False),
    mojokarateclub.DeepWalk(walk_number=4, walk_length=20, dimensions=8, workers=1),
    mojokarateclub.Node2Vec(walk_number=4, walk_length=20, dimensions=8, workers=1),
    mojokarateclub.Walklets(walk_number=4, walk_length=20, dimensions=8, workers=1),
    mojokarateclub.Diff2Vec(diffusion_number=4, diffusion_cover=20, dimensions=8, workers=1),
):
    model.fit(graph)
    print(type(model).__name__, model.get_embedding().shape)
```

The low-level kernels are exported too, and take numpy arrays:
`fill`, `copy`, `axpy`, `scale`, `dot`, `total`, `sum_squares`, `frobenius`,
`gemm`, `gemm_nt`, `matvec`, `eigh`, `invert`, `linspace`, `histogram`,
`l1_normalize_rows`, `dense_to_coo`, `md5_digest`, and the `csr_*` / `coo_*`
family. See `python/mojokarateclub/linalg.py`.

## What is covered

Graph level (`karateclub.graph_embedding`):

| estimator | what is in Mojo | what stays upstream |
| --- | --- | --- |
| `LDP` | the whole `_calculate_ldp`: log degrees, the `n x 5` profile block, the five histograms | nothing |
| `FGSD` | normalized Laplacian, the Moore-Penrose pseudo-inverse, the similarity matrix, the histogram | nothing |
| `SF` | normalized Laplacian, the eigendecomposition, the padding arithmetic | nothing |
| `NetLSD` | self-loop stripping, normalized Laplacian, the `k` extreme eigenvalues, the up/down interpolation, the heat kernel trace | nothing |
| `Graph2Vec` | the Weisfeiler-Lehman recursion and its MD5 | `gensim`'s `Doc2Vec`, as upstream |

Node level (`karateclub.node_embedding.neighbourhood`):

| estimator | what is in Mojo | what stays upstream |
| --- | --- | --- |
| `LaplacianEigenmaps` | normalized Laplacian, the eigendecomposition, the eigenvector selection | nothing |
| `NodeSketch` | the whole fit: the MT19937 hash matrix, every sketch round, the SLA augmentation | nothing |
| `FirstOrderLINE` | the whole fit: the MT19937 draws, both batches, the scatter, the decay | nothing |
| `SecondOrderLINE` | the same, with the two embedding tables | nothing |
| `DeepWalk` | `RandomWalker.do_walk` / `do_walks` | `gensim`'s `Word2Vec` |
| `Node2Vec` | `BiasedRandomWalker.do_walk` / `do_walks`, including the `p`/`q` weighting | `gensim`'s `Word2Vec` |
| `Walklets` | `RandomWalker` and `_select_walklets` | `gensim`'s `Word2Vec` |
| `Diff2Vec` | `EulerianDiffuser.do_diffusions` / `_run_diffusion_process`: the growth and the Eulerian tour | `gensim`'s `Word2Vec` |

Utilities (`karateclub.utils`): `RandomWalker`, `BiasedRandomWalker`,
`EulerianDiffuser`, `WeisfeilerLehmanHashing`, and the MT19937 generator they
draw from.

## What is not covered

Everything else in karateclub, and it is most of the library: the community
detectors, the attributed embedders (`BANE`, `MUSAE`, `TENE`, `TADW`, `ASNE`,
`AE`, `FSCNMF`, `FeatherNode`, `SINE`), the remaining neighbourhood models
(`HOPE`, `NetMF`, `GraRep`, `NMFADMM`, `RandNE`, `SocioDim`, `GLEE`, `BoostNE`),
the structural models (`GraphWave`, `RolX`, `NEU`, `Role2Vec`) and the rest of
`graph_embedding` (`GL2Vec`, `GeoScattering`, `FeatherGraph`,
`WaveletCharacteristic`).

The reason is the same in every case: the part of the model that costs
something is a call into a library that is already fast. `HOPE` and `GraRep`
spend their time in `scipy.sparse.linalg.svds` and `sklearn`'s
`TruncatedSVD`; `GraphWave` and `NetMF` in ARPACK; `FeatherGraph` in `gf`;
`GeoScattering` in PyGSP. Rewriting a thin wrapper around LAPACK buys nothing.
The models here are the ones whose arithmetic *is* the model.

One karateclub behaviour is deliberately not reproduced, because it is a defect
rather than design and a caller can work around it:

- `eccentricity` returns `-1.0` for a node that cannot reach the whole graph,
  where `networkx.eccentricity` raises.

The `verbose` flag on both `LINE` classes prints a plain line where upstream
draws a tqdm bar, because a progress bar is not a number; the flag, its default
and its position in the argument list are upstream's.

## Documented divergences

Where the port is not bit-identical, this is where it is not, and why.

1. **The eigensolver is a dense symmetric Jacobi sweep, not ARPACK.** `SF`,
   `NetLSD`, `FGSD` and `LaplacianEigenmaps` all reach for
   `scipy.sparse.linalg.eigsh`, which is a restarted Lanczos on the sparse
   matrix. The port diagonalises the dense `n x n` matrix instead.
   The spectrum agrees to about 1e-14, and so do the eigenvectors, but
   eigenvector *signs* are arbitrary on both sides and the basis inside a
   degenerate eigenspace is arbitrary as well - the `isolated` fixture has six
   zero eigenvalues and asks for eight vectors, so which two of the four
   1.5-eigenvectors come back is not determined by anything. The tests compare
   the projector onto the returned subspace and the eigenvector residual, not
   the columns.
   `LaplacianEigenmaps`' `maximum_number_of_iterations` is the ARPACK iteration
   cap and has no dense counterpart; it is accepted and unused, and the module
   says so.
2. **`Diff2Vec` draws from the port's MT19937, and its growth stops when the
   infected set can grow no further.** Upstream's `random.sample` and
   `random.choice` are CPython's `_random`, so the tours are not upstream's
   for a given seed. The second difference is a fix rather than a choice:
   upstream's `while infected_counter < self.diffusion_cover` never
   terminates once the cover exceeds the source's component, so
   `EulerianDiffuser(10, 80)` on a 34-node graph hangs. The kernel carries a
   count of the edges leaving the infected set and stops when it reaches
   zero, which is exactly the condition under which no further infection is
   possible. Whenever such an edge does exist the walk can still find it, so
   the two agree wherever upstream returns at all - which the tests check by
   replaying upstream's own growth and a real `networkx.eulerian_circuit` on
   the same draws and comparing the tours node for node.
3. **`Graph2Vec`, `DeepWalk`, `Node2Vec` and `Walklets` share a PRNG, not a
   stream.** Upstream draws walks from the `random` module
   (`random.sample`) and from `np.random.choice`; the port uses its own
   MT19937, seeded from whichever global stream upstream's own draw would have
   come from, so `fit` is reproducible for a given `seed` but the walks
   themselves differ. The tests compare what does not depend on the draws: the
   walk count and lengths exactly, the visit counts as a z-score against
   upstream's own multinomial standard error, with both samples' variance in
   it (worst observed 3.67 over the 34- and 200-node fixtures, threshold 5,
   which is where a max over several hundred standard normals stops being
   noise), and the edge-validity of every step.
4. **`natural_log` is within one ulp of `math.log`, not equal to it, and
   `natural_exp` carries about one ulp of its argument into the result.**
   `std.math.log` in this toolchain is SLEEF u10 and is only good to about
   1e-10 relative, which is enough to move a histogram bin, so `mathfn.mojo`
   carries a range-reduced `atanh` series. Measured over 2020 arguments it
   lands one ulp from `math.log` on 45 of them, always on the low side.
   `natural_exp` reduces with `x - k*ln2`, which cancels two operands of size
   `|x|`, so it is good to 2e-14 at `x = -200` and 8e-14 at `x = -700` rather
   than to the 1e-16 its Taylor series would give on its own. NetLSD's heat
   trace is evaluated over exactly that range, which is four orders below the
   SLEEF bound this module exists to beat.
   `NodeSketch` feeds `-log` of a uniform draw into an argmin, so its parity
   is asserted against real upstream output rather than against this bound -
   and it matches exactly.
5. **The WL extractor takes base labels as bytes, not as Python strings.** A
   label cannot cross the C ABI as a string, so `WeisfeilerLehmanHashing`
   takes the `[str(v)]`-ed base features as one `uint8` buffer plus an offset
   table and returns raw digest words for the caller to hexlify. Everything
   downstream of that - the recursion, the `sorted()`, the `"_".join`, the
   MD5, the erasure, the node-major flattening - is upstream's, and the digests
   are byte-identical to `hashlib`'s. This is the only place a Python object
   is replaced by a buffer.
6. **`erase_base_features` changes the read stride.** The kernel compacts each
   node's run by one slot, so the Python layer reads the erased stride rather
   than dropping element 0 from the list it built. Same output as upstream's
   `del extracted_features[k][0]`.
7. **`Graph2Vec` calls `model.docvecs`,** which gensim 4.4 deprecates in favour
   of `model.dv`. Upstream's line is kept rather than silently updated, so the
   call emits a `DeprecationWarning` on a modern gensim.

## Parity testing

`tests/vectors/upstream.npz` is a dump of real karateclub 1.3.3 output,
produced by `tools/dump_upstream.py` inside the `upstream` environment:
`pixi run vectors` regenerates it. The test suite compares against that dump,
so the assertions are against upstream and not against a reference this
repository wrote. The fixtures in `python/mojokarateclub/_datasets.py` are
built from edge lists rather than networkx generators, because the two
environments resolve networkx 2.6 and 3.7 and a generator's RNG schedule
changed between them.

`176 passed` at the time of writing:

| test file | what it pins |
| --- | --- |
| `test_linalg.py` | the 28 low-level wrappers against numpy, scipy and hand-written references, and the C ABI: every `@export` in `capi.mojo` has a signature, so a kernel cannot be called through a default ctypes conversion |
| `test_graph_embedding.py` | `LDP`, `FGSD`, `SF`, `NetLSD`, `LaplacianEigenmaps`, the WL extractor and `Graph2Vec` against the upstream dump; node ordering, edge weighting and the parameter guards that stand between an estimator and a kernel |
| `test_node_embedding.py` | `NodeSketch` and both `LINE` orders against the dump, exactly; the walkers against upstream's walk counts and lengths and a z-score on the visit distribution; the diffuser against upstream's own growth driven on the same draws and a real `networkx.eulerian_circuit`, tour for tour; MD5 against `hashlib`; the MT19937 stream against `numpy.random.RandomState` |

The exact-parity results worth naming: `LDP` and `NodeSketch` match the
upstream arrays with `array_equal`; `FGSD`, `SF` and `LaplacianEigenmaps` agree
to better than 1e-9; `NetLSD` to better than 1e-5, and the residual there is
ARPACK's, not the port's - upstream moves by the same 3.7e-07 from the exact
float64 heat kernel trace, and the port sits 7.8e-16 from it; both `LINE`
orders to better than 1e-12.

`EulerianDiffuser` is the one model whose parity test does not need a dump,
because upstream's algorithm can be replayed in full: the reference in
`tests/test_node_embedding.py` is `_run_diffusion_process` transcribed, with
the draws taken from this port's own MT19937 - so both sides consume the
generator word for word - and a real `nx.DiGraph` and
`networkx.eulerian_circuit` doing the subgraph and the tour. Over 4 fixtures
and 4 covers the kernel's tours equal networkx's exactly. That pins the growth,
the insertion order the tour depends on, and the search itself, and it holds
for a cover above the component size, where upstream never returns at all.

## Benchmarks

`pixi run bench` measures this port and real karateclub 1.3.3 on identical
graphs, in the two environments, and prints a markdown table. The pixi task
takes a machine-wide `flock` so a concurrent run cannot distort the numbers,
which is why the benchmark must be started through `pixi run bench` and not by
hand.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs, shared with other jobs. Upstream side:
Python 3.9.23, numpy 1.22.4, networkx 2.6.3, scipy 1.9.3. Port side: Python
3.13.15, Mojo 1.2.0.dev2026092605. The table is one `pixi run bench`
invocation, at 2026-09-26T22:08:49Z.

| case | workload | karateclub 1.3.3 | mojo-karateclub | speedup |
| --- | --- | --- | --- | --- |
| LDP on karate | `LDP(bins=32)`, fit + `get_embedding` | 2.732 ms | 0.203 ms | 13.43x |
| LDP on ws300 | `LDP(bins=32)`, fit + `get_embedding` | 0.0180 s | 1.238 ms | 14.52x |
| FGSD on karate | `FGSD()`, fit + `get_embedding` | 2.130 ms | 0.868 ms | 2.45x |
| FGSD on ws300 | `FGSD()`, fit + `get_embedding` | 0.4224 s | 1.4913 s | 0.28x (slower) |
| SF on karate | `SF(dimensions=16)`, fit + `get_embedding` | 5.500 ms | 0.910 ms | 6.04x |
| SF on ws300 | `SF(dimensions=32)`, fit + `get_embedding` | 0.4268 s | 1.4517 s | 0.29x (slower) |
| NetLSD on karate | `NetLSD(scale_steps=32, approximations=8)` | 9.164 ms | 0.987 ms | 9.28x |
| NetLSD on ws300 | `NetLSD(scale_steps=32, approximations=16)` | 0.2465 s | 1.4536 s | 0.17x (slower) |
| LaplacianEigenmaps on karate | `LaplacianEigenmaps(dimensions=8)` | 9.200 ms | 1.363 ms | 6.75x |
| LaplacianEigenmaps on ws300 | `LaplacianEigenmaps(dimensions=8)` | 0.0238 s | 1.4575 s | 0.02x (slower) |
| NodeSketch on karate | `NodeSketch(dimensions=16)` | 0.0167 s | 0.434 ms | 38.56x |
| NodeSketch on ws300 | `NodeSketch(dimensions=16)` | 0.2257 s | 3.092 ms | 72.99x |
| FirstOrderLINE on karate | `FirstOrderLINE(dimensions=8, epochs=5)` | 0.996 ms | 0.167 ms | 5.95x |
| FirstOrderLINE on ws300 | `FirstOrderLINE(dimensions=8, epochs=5)` | 0.0160 s | 1.798 ms | 8.91x |
| SecondOrderLINE on karate | `SecondOrderLINE(dimensions=8, epochs=5)` | 1.064 ms | 0.182 ms | 5.84x |
| SecondOrderLINE on ws300 | `SecondOrderLINE(dimensions=8, epochs=5)` | 0.0159 s | 1.675 ms | 9.51x |
| WeisfeilerLehmanHashing on ws300 | `wl_iterations=3`, `get_graph_features` | 5.569 ms | 5.788 ms | 0.96x (slower) |
| RandomWalker on ws300 | `walk_length=40, walk_number=10` | 0.4183 s | 0.0472 s | 8.86x |
| BiasedRandomWalker on ws300 | `walk_length=40, walk_number=10, p=0.5, q=2.0` | 24.7254 s | 0.0542 s | 455.99x |

Geomean over the 19 cases: **4.35x**. This box is shared, and the upstream
`ws300` column moves by a factor of four between runs - `FGSD` on `ws300`
measured 0.0997 s, then 0.4224 s on consecutive runs, and `SF` on `ws300` 0.1016 s
then 0.4268 s, because the ARPACK side is sensitive to whatever else the machine
is doing. The port's own column is stable to about 5% across those runs. Read
the direction of each row, not its third digit; the `ws300` spectral rows are
slower in every run, and their exact ratio is not.

Five cases are slower, and the pattern is not a mystery. The four `ws300`
spectral rows are the dense-Jacobi-versus-ARPACK trade: upstream asks ARPACK
for the eight or sixteen eigenvalues it wants out of a sparse matrix and gets
them in 24 to 430 ms, while the port diagonalises all 300x300 and then
selects - about 1.4 s either way, and the gap is the whole cost of the dense
sweep. `LaplacianEigenmaps` is the worst at 0.02x because the requested `k` is
tiny (8) against a large `n`, which is exactly the regime ARPACK exists for.
The fifth is `WeisfeilerLehmanHashing` at 0.96x, where the work is one MD5 per
node per iteration and the port's MD5 is competitive but the surrounding
recursion is not; upstream's `hashlib` is backed by OpenSSL's assembly.

The wins are where upstream spends its time in Python-level loops.
`BiasedRandomWalker` is 456x because upstream rebuilds a numpy array and calls
`np.piecewise` and `np.random.choice` at every one of 120000 steps, and
`NodeSketch` is 39-73x because upstream counts with `Counter` objects in a
double Python loop. `LDP` is 13-15x for the same reason one level up: upstream
builds its feature block out of a Python list comprehension per node.

## How it works

**One compilation unit.** `build/build.sh` runs `mojo build --emit shared-lib`
on `src/mojokarateclub/capi.mojo` and writes `dist/libmojo-karateclub.so`. Mojo's
build cost is essentially fixed per unit, so all the kernels live in one
translation unit with `capi.mojo` re-exporting them; `-I src/mojokarateclub`
is what lets the sibling files resolve as bare module names. Editing a kernel
means re-running `pixi run build`, and `tests/test_linalg.py` has a test that
every declared symbol really is in the library, so a stale `.so` fails loudly
rather than mysteriously.

**FFI.** `@export` refuses parametric functions and an `abi("C")` function may
not be parametric, so a Mojo pointer cannot cross the boundary: every buffer
goes as an `Int` address and is rebuilt with
`Pointer[T, AnyOrigin[mut=True]](unsafe_from_address=addr)` inside the wrapper.
`Int` is 64-bit, so ctypes argtypes are `c_int64` throughout; leaving one off
lets ctypes pick a 32-bit conversion and truncate an address into a wild
pointer. Flags cross as `Int` 0/1 rather than `Bool`, matching the kernel
signatures; a `Bool` parameter was checked and works across the boundary in
this toolchain, so the choice is for symmetry with the `Int`-typed kernels
rather than a workaround.

**Memory layout.** Nothing in Mojo allocates. Every kernel writes into
caller-owned buffers and takes its scratch as explicit trailing arguments, and
the Python layer allocates it with numpy, sized from the estimator's own
parameters. There is no arena and nothing to leak; the cost is that
`_calculate_sf` reads as a list of buffer arguments, which is why each module
documents what each one is.

Graphs cross as CSR: `indptr: int32[n+1]`, `indices: int32[nnz]`, and
`values: float64[nnz]`. Both directions of every undirected edge are present
and each row is in ascending column order, which is what the kernels assume;
`python/mojokarateclub/estimator.py` builds that from a networkx graph, walks
the nodes in label order rather than networkx's insertion order (so row `i`
is node `i`, as upstream's `nodelist=range(n)`), and applies the self-loop
pass. The spectral estimators always read the `weight` attribute, defaulting
to `1.0` per edge, because that is what upstream's
`nx.normalized_laplacian_matrix` does; `LDP` and the walkers follow upstream's
own `nx.is_weighted` dispatch instead. Dense matrices are row-major `float64`.
The MD5 digest is four `uint32` words, little-endian, which is what
`hashlib.hexdigest()` prints when you hexlify them in that order.

**Threading.** There is none. `parallelize` does not exist in this toolchain
(see `MOJO_NOTES.md`, section 4), and the GPU host API is absent too (section
5), so this is single-threaded CPU code throughout. That is also the honest
reason the dense spectral paths lose to LAPACK's multithreaded BLAS on the
same matrices.

**Numerics.** Two toolchain facts shape the arithmetic. `std.math.log` and
`std.math.exp` are SLEEF u10, good to about 1e-10 relative, so `mathfn.mojo`
carries its own. And `stack_allocation` is not safe for kernel scratch: the
buffer is silently clobbered when the allocating function is inlined into a
caller that also has a stack frame, which is why nothing here uses it.
`MOJO_NOTES.md` has the rest, all of it verified by compiling rather than by
reading the docs.

## Layout

```
src/mojokarateclub/   Mojo kernels, one file per upstream module
  kclinalg.mojo       dense, CSR and COO primitives
  spectral.mojo       normalized adjacency and Laplacian, pseudo-inverse, eigenvalue selection
  mathfn.mojo         full-precision log and exp
  hashing.mojo        RFC 1321 MD5
  ldp.mojo fgsd.mojo sf.mojo netlsd.mojo          graph_embedding
  laplacianeigenmaps.mojo nodesketch.mojo line.mojo walker.mojo treefeatures.mojo
  diffuser.mojo       EulerianDiffuser, the growth and the Hierholzer tour
  capi.mojo           the C ABI, the single compilation unit
build/build.sh        mojo build --emit shared-lib
python/mojokarateclub/
  _ffi.py             library handle, signatures, buffer rules
  linalg.py           the low-level wrappers
  estimator.py        upstream's Estimator, plus the CSR conversion
  graph_embedding.py  LDP, FGSD, SF, NetLSD, Graph2Vec
  node_embedding.py   DeepWalk, Diff2Vec, Node2Vec, Walklets, NodeSketch,
                      FirstOrderLINE, SecondOrderLINE, LaplacianEigenmaps
  utils.py            RandomWalker, BiasedRandomWalker, EulerianDiffuser,
                      WeisfeilerLehmanHashing
  _datasets.py        the fixture graphs
tests/                pytest, against tests/vectors/upstream.npz
tools/dump_upstream.py  regenerates those vectors in the upstream environment
bench/                bench.py (port), bench_upstream.py (upstream),
                      bench_eig.py (the dense sweep alone), run.sh
```

## Licence

MIT, Lee Penkman. See `LICENSE`. karateclub is GPLv3 and is not vendored here;
it is a separate package in a separate environment, used as the reference.
