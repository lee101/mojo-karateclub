"""Local Degree Profile, the `LDP` estimator of `graph_embedding/ldp.py`.

The upstream `_calculate_ldp` body is three steps and this file keeps them in
that order: the log degrees of the graph, the `n x 5` block of local degree
profile features, and the per-column histograms that concatenate into the
`5 * bins` embedding.

A node's neighbour list is its CSR row, which upstream is
`[neb for neb in G.neighbors(n)]`. `Estimator._ensure_integrity` has put a self
loop in every node by the time this runs, so no row is empty and the isolated
node case is a row of length one, which the four statistics still handle. The
row holds each neighbour once, so `G.degree[n]` is `len(nbrs) + (n in nbrs)` --
networkx counting a self loop twice -- and that is what `log_degrees` gives.

Nothing here allocates. Every buffer is the caller's; the only scratch is the
`n` word gather buffer `ldp_embedding` needs because the block is row major and
`linalg.histogram` wants a contiguous column.
"""

from std.math import max, min, sqrt

import kclinalg as linalg
import mathfn

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

# `np.concatenate([degrees.reshape(-1, 1), np.array(features)], axis=1)`.
comptime LDP_COLS = 5


def log_degrees(indptr: IPtr, indices: IPtr, n: Int, dst: Ptr):
    """`np.log(np.array([G.degree[n] for n in range(G.number_of_nodes())]))`.

    The graph is unweighted, as upstream's datasets are, so the degree is the
    row length plus one more when the row carries the node itself.
    """
    for r in range(n):
        var bounds = linalg.row_range(indptr, r)
        var degree = bounds[1] - bounds[0]
        for e in range(bounds[0], bounds[1]):
            if linalg.at(indices, e) == r:
                degree += 1
                break
        dst.unsafe_offset(r)[] = mathfn.natural_log(Float64(degree))


def ldp_features(indptr: IPtr, indices: IPtr, degrees: Ptr, n: Int, dst: Ptr):
    """The `n x 5` block: `degrees.reshape(-1, 1)` beside the profile columns.

    Columns 1 to 4 hold `np.min`, `np.max`, `np.std` and `np.mean` of the log
    degrees of the node's neighbours, in that order. `np.std` is the population
    deviation, so the squared deviations are divided by the neighbour count.
    `dst` is `5 * n` words and must not overlap `degrees`.
    """
    for r in range(n):
        var bounds = linalg.row_range(indptr, r)
        var count = Float64(bounds[1] - bounds[0])
        dst.unsafe_offset(r * LDP_COLS)[] = degrees.unsafe_load(r)
        # The row is never empty, so the first neighbour seeds the three
        # accumulators and no infinity is needed to start them.
        var lo = degrees.unsafe_load(linalg.at(indices, bounds[0]))
        var hi = lo
        var total = lo
        var e = bounds[0] + 1
        while e < bounds[1]:
            var d = degrees.unsafe_load(linalg.at(indices, e))
            lo = min(lo, d)
            hi = max(hi, d)
            total += d
            e += 1
        var mean = total / count
        var sq = 0.0
        e = bounds[0]
        while e < bounds[1]:
            var diff = degrees.unsafe_load(linalg.at(indices, e)) - mean
            sq += diff * diff
            e += 1
        dst.unsafe_offset(r * LDP_COLS + 1)[] = lo
        dst.unsafe_offset(r * LDP_COLS + 2)[] = hi
        dst.unsafe_offset(r * LDP_COLS + 3)[] = sqrt(sq / count)
        dst.unsafe_offset(r * LDP_COLS + 4)[] = mean


def ldp_embedding(features: Ptr, n: Int, bins: Int, dst: Ptr, column: Ptr):
    """The `5 * bins` embedding: one histogram per column of the block.

    `column` is the caller's `n` word gather buffer. The block is row major
    with five columns, so a column is strided while `linalg.histogram` reads a
    contiguous run of `n` values.
    `dst` is `5 * bins` words and must not overlap `features` or `column`.
    """
    for c in range(LDP_COLS):
        for i in range(n):
            column.unsafe_offset(i)[] = features.unsafe_load(i * LDP_COLS + c)
        linalg.histogram(column, dst.unsafe_offset(c * bins), n, bins, 0.0, 10.0)
