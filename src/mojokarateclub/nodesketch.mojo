"""`NodeSketch`, the recursive sketching of the self-loop-augmented adjacency.

Port of `karateclub/node_embedding/neighbourhood/nodesketch.py`.

Upstream keeps four mutable objects alive across `fit`: `_hash_values`
(`dimensions x num_nodes`), the two COO triples `_sla` / `_sla_original` and
`_sketch` (`dimensions x num_nodes` of column ids). All four are caller-owned
buffers here, and the graph arrives as CSR as well, because `_augment_sla`
walks `list(self._graph[node])` and the caller already holds the neighbourhood
in that shape.

`_generate_hash_values` cannot be an ordinary PRNG. It is
`-np.log(np.random.rand(dimensions, num_nodes))` evaluated right after
`np.random.seed(seed)`, so the legacy MT19937 `random_sample` stream is spelled
out below exactly as `mt19937ar.c` and numpy spell it, and the seed is an
argument rather than hidden global state. Get this wrong and every embedding
after the first sketch diverges from upstream's. The `log` is `mathfn`'s rather
than `std.math`'s, which is a 1e-10 kernel and would put the hashes off
enough to change which entry a scan picks.

Nothing allocates. The self-loop-augmented adjacency is passed as COO
`(rows, cols, values)` plus a count, and the augmented entries are appended
into caller-provided capacity.
"""

from std.math import inf

import mathfn

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime MT = Pointer[UInt32, AnyOrigin[mut=True]]

comptime MT_N = 624
comptime MT_M = 397
comptime MT_MATRIX_A: UInt32 = UInt32(0x9908b0df)
comptime MT_UPPER_MASK: UInt32 = UInt32(0x80000000)
comptime MT_LOWER_MASK: UInt32 = UInt32(0x7fffffff)
# mt[0 .. MT_N - 1] then the draw index, which init_genrand leaves saturated at
# MT_N so the first draw twists.
comptime MT_STATE_WORDS = MT_N + 1


def _init_weight(decay: Float64, dimensions: Int) -> Float64:
    """`__init__`: `self._weight = self.decay / self.dimensions`."""
    return decay / Float64(dimensions)


# ------------------------------------------------- legacy MT19937


def init_genrand(state: MT, s: UInt32):
    """`init_genrand` from mt19937ar.c, which is also numpy's scalar seeding."""
    state.unsafe_offset(0)[] = s
    var i = 1
    while i < MT_N:
        var prev = state.unsafe_load(i - 1)
        state.unsafe_offset(i)[] = (
            UInt32(1812433253) * (prev ^ (prev >> UInt32(30))) + UInt32(i)
        )
        i += 1
    state.unsafe_offset(MT_N)[] = UInt32(MT_N)


def genrand_uint32(state: MT) -> UInt32:
    """One tempered 32-bit word, advancing the state in place."""
    if Int(state.unsafe_load(MT_N)) >= MT_N:
        var kk = 0
        while kk < MT_N - MT_M:
            var y = (state.unsafe_load(kk) & MT_UPPER_MASK) | (
                state.unsafe_load(kk + 1) & MT_LOWER_MASK
            )
            var next = state.unsafe_load(kk + MT_M) ^ (y >> UInt32(1))
            if (y & UInt32(1)) != UInt32(0):
                next ^= MT_MATRIX_A
            state.unsafe_offset(kk)[] = next
            kk += 1
        while kk < MT_N - 1:
            var y = (state.unsafe_load(kk) & MT_UPPER_MASK) | (
                state.unsafe_load(kk + 1) & MT_LOWER_MASK
            )
            var next = state.unsafe_load(kk + (MT_M - MT_N)) ^ (y >> UInt32(1))
            if (y & UInt32(1)) != UInt32(0):
                next ^= MT_MATRIX_A
            state.unsafe_offset(kk)[] = next
            kk += 1
        # mt[0] is read back already overwritten by the first loop, as the
        # reference does.
        var y = (state.unsafe_load(MT_N - 1) & MT_UPPER_MASK) | (
            state.unsafe_load(0) & MT_LOWER_MASK
        )
        var next = state.unsafe_load(MT_M - 1) ^ (y >> UInt32(1))
        if (y & UInt32(1)) != UInt32(0):
            next ^= MT_MATRIX_A
        state.unsafe_offset(MT_N - 1)[] = next
        state.unsafe_offset(MT_N)[] = UInt32(0)

    var y = state.unsafe_load(Int(state.unsafe_load(MT_N)))
    state.unsafe_offset(MT_N)[] = UInt32(Int(state.unsafe_load(MT_N)) + 1)
    y ^= y >> UInt32(11)
    y ^= (y << UInt32(7)) & UInt32(0x9d2c5680)
    y ^= (y << UInt32(15)) & UInt32(0xefc60000)
    y ^= y >> UInt32(18)
    return y


def random_sample(state: MT) -> Float64:
    """Two tempered words as a double in [0, 1): the draw `np.random.rand`
    makes, which consumes the stream in this exact order."""
    var a = genrand_uint32(state) >> UInt32(5)
    var b = genrand_uint32(state) >> UInt32(6)
    return (Float64(a) * 67108864.0 + Float64(b)) / 9007199254740992.0


# ------------------------------------------------- NodeSketch


def _generate_hash_values(
    num_nodes: Int, dimensions: Int, seed: UInt32, hashes: Ptr, state: MT
) -> None:
    """`hashes = -np.log(np.random.rand(dimensions, num_nodes))`, filled in C
    order, after `init_genrand(seed)`."""
    init_genrand(state, seed)
    for d in range(dimensions):
        for i in range(num_nodes):
            hashes.unsafe_store(
                d * num_nodes + i, -mathfn.natural_log(random_sample(state))
            )


def _do_single_sketch(
    num_nodes: Int,
    dimensions: Int,
    sla_rows: IPtr,
    sla_cols: IPtr,
    sla_vals: Ptr,
    sla_nnz: Int,
    hashes: Ptr,
    sketch: IPtr,
    min_val: Ptr,
) -> None:
    """`for iter in range(self.dimensions)`: one round of sketching.

    Every entry becomes `hashes[iter, col] / value`, and the sketch keeps, per
    node, the column of the smallest one. The comparison is `<`, so on a tie
    the first entry in sla order wins, which is why `_augment_sla` leaves the
    sla sorted the way `sum_duplicates()` does. A node with no entry at all
    keeps the `-1` written here; upstream keeps `None`, which the self-loops
    `_ensure_integrity` adds keep unreachable.
    """
    var big = inf[DType.float64]()
    for d in range(dimensions):
        var i = 0
        while i < num_nodes:
            min_val.unsafe_store(i, big)
            sketch.unsafe_store(d * num_nodes + i, Int32(-1))
            i += 1
        var e = 0
        while e < sla_nnz:
            var node = Int(sla_rows.unsafe_load(e))
            var col = Int(sla_cols.unsafe_load(e))
            var v = hashes.unsafe_load(d * num_nodes + col) / sla_vals.unsafe_load(e)
            if v < min_val.unsafe_load(node):
                min_val.unsafe_store(node, v)
                sketch.unsafe_store(d * num_nodes + node, Int32(col))
            e += 1


def _accumulate(
    cols: IPtr,
    vals: Ptr,
    e: Int,
    marker: Int32,
    stamp: IPtr,
    touched: IPtr,
    acc: Ptr,
    merged: Int,
) -> Int:
    """Fold entry `e` of the pending sla into the per-column running sum,
    recording the column in `touched` the first time this row sees it. Returns
    the new number of distinct columns."""
    var col = Int(cols.unsafe_load(e))
    if stamp.unsafe_load(col) != marker:
        stamp.unsafe_store(col, marker)
        touched.unsafe_store(merged, Int32(col))
        acc.unsafe_store(col, vals.unsafe_load(e))
        return merged + 1
    acc.unsafe_store(col, acc.unsafe_load(col) + vals.unsafe_load(e))
    return merged


def _augment_sla(
    num_nodes: Int,
    dimensions: Int,
    weight: Float64,
    indptr: IPtr,
    indices: IPtr,
    orig_rows: IPtr,
    orig_cols: IPtr,
    orig_vals: Ptr,
    orig_nnz: Int,
    sla_rows: IPtr,
    sla_cols: IPtr,
    sla_vals: Ptr,
    sla_cap: Int,
    sketch: IPtr,
    stamp: IPtr,
    counts: IPtr,
    touched: IPtr,
    acc: Ptr,
) -> Int:
    """Append one `(node, target, count * weight)` entry per distinct target
    the previous sketch chose inside `node`'s neighbourhood, then
    `sum_duplicates()`. Returns the new nnz, or `-1` if `sla_cap` is short.

    `sum_duplicates()` really does merge: an entry the original sla already had
    (data `1.0`) and one appended for the same `(row, col)` come out as a
    single entry worth `1.0 + count * weight`, and the triple is left sorted.
    Skipping the merge would leave two entries per pair, and the smaller of the
    two is the `1.0` one, so a sketch that upstream hands to an appended entry
    would land on the original instead whenever the two are within `weight` of
    each other. `weight` is `decay / dimensions`, so that window is small, but
    it is not zero.

    scipy leaves that sorted triple column-major and this writes it row-major.
    The order inside a row is the same either way, ascending by column, and a
    sketch only ever compares entries of one row against each other, so the
    embeddings agree; the triples themselves compare equal as multisets.

    The merged triple is written into the working sla from position zero while
    it is read from the pristine original and from the appended tail, so the
    whole merge costs no extra buffer. That is safe because a node's merged
    row is never longer than the entries it replaces: the write pointer after
    node `i` is at most `sum_{j<=i} (original_j + appended_j)`, which is past
    the last appended entry node `i` has read and so never overwrites one it
    has not.

    The per-node `Counter` is `stamp` plus `counts`: the marker `2 * node + 1`
    picks the current node's targets out of the array, so nothing has to be
    cleared between nodes. The merge reuses `stamp` and `touched` under the
    disjoint marker `2 * node + 2`, so `stamp` is zeroed here rather than by
    the caller.
    """
    # self._sla = self._sla_original.copy(). The merge below rewrites the whole
    # triple from scratch, so nothing has to be carried over: the entries that
    # survive are read straight out of the pristine original, and only the
    # appended tail has to be laid down first.
    for i in range(num_nodes):
        stamp.unsafe_store(i, Int32(0))

    var out = orig_nnz
    var o = 0
    var write = 0
    for node in range(num_nodes):
        var count_marker = Int32(2 * node + 1)
        var merge_marker = Int32(2 * node + 2)
        # frequencies = [Counter([dim[neighbor] for dim in self._sketch]) ...]
        var k = 0
        var e2 = Int(indptr.unsafe_load(node))
        var stop = Int(indptr.unsafe_load(node + 1))
        while e2 < stop:
            var neighbor = Int(indices.unsafe_load(e2))
            e2 += 1
            var d = 0
            while d < dimensions:
                var target = Int(sketch.unsafe_load(d * num_nodes + neighbor))
                if stamp.unsafe_load(target) != count_marker:
                    stamp.unsafe_store(target, count_marker)
                    touched.unsafe_store(k, Int32(target))
                    counts.unsafe_store(target, Int32(0))
                    k += 1
                counts.unsafe_store(
                    target, Int32(Int(counts.unsafe_load(target)) + 1)
                )
                d += 1
        var appended = out
        var j = 0
        while j < k:
            if out >= sla_cap:
                return -1
            var target = Int(touched.unsafe_load(j))
            sla_rows.unsafe_store(out, Int32(node))
            sla_cols.unsafe_store(out, Int32(target))
            sla_vals.unsafe_store(
                out, Float64(Int(counts.unsafe_load(target))) * weight
            )
            out += 1
            j += 1

        # self._sla.sum_duplicates(), one row at a time.
        var ostart = o
        while o < orig_nnz and Int(orig_rows.unsafe_load(o)) == node:
            o += 1
        var merged = 0
        var p = ostart
        while p < o:
            merged = _accumulate(
                orig_cols, orig_vals, p, merge_marker, stamp, touched, acc, merged
            )
            p += 1
        p = appended
        while p < out:
            merged = _accumulate(
                sla_cols, sla_vals, p, merge_marker, stamp, touched, acc, merged
            )
            p += 1
        # scipy leaves the triple canonical, so the merged row is sorted by
        # column; that is also the order the `<` scan of the next sketch breaks
        # ties in.
        var a = 1
        while a < merged:
            var key = Int32(touched.unsafe_load(a))
            var b = a - 1
            while b >= 0 and Int(touched.unsafe_load(b)) > Int(key):
                touched.unsafe_store(b + 1, touched.unsafe_load(b))
                b -= 1
            touched.unsafe_store(b + 1, key)
            a += 1
        j = 0
        while j < merged:
            var col = Int(touched.unsafe_load(j))
            sla_rows.unsafe_store(write, Int32(node))
            sla_cols.unsafe_store(write, Int32(col))
            sla_vals.unsafe_store(write, acc.unsafe_load(col))
            write += 1
            j += 1

    return write


def fit(
    num_nodes: Int,
    dimensions: Int,
    iterations: Int,
    decay: Float64,
    seed: UInt32,
    indptr: IPtr,
    indices: IPtr,
    orig_rows: IPtr,
    orig_cols: IPtr,
    orig_vals: Ptr,
    sla_rows: IPtr,
    sla_cols: IPtr,
    sla_vals: Ptr,
    sla_cap: Int,
    hashes: Ptr,
    sketch: IPtr,
    state: MT,
    stamp: IPtr,
    counts: IPtr,
    touched: IPtr,
    acc: Ptr,
    min_val: Ptr,
) -> Int:
    """`fit`: seed, hash matrix, the self-loop-augmented adjacency, one sketch,
    then `iterations - 1` rounds of augment-and-sketch. Returns 0, or 1 when
    `sla_cap` is too small.

    The adjacency arrives as CSR and is expanded to COO here, which is
    `nx.adjacency_matrix(..., nodelist=range(num_nodes)).tocoo()` with every
    `data` set to `1`. That expansion is `_sla_original`; the working sla need
    not be initialised, because `_augment_sla` opens by copying
    `_sla_original` back into it.

    Scratch: `hashes` (`dimensions * num_nodes` Float64), `sketch`
    (`dimensions * num_nodes` Int32), `state` (`MT_STATE_WORDS` UInt32),
    `stamp`, `counts`, `touched` (Int32) and `acc`, `min_val` (Float64), each
    `num_nodes` long, and the two COO triples, `orig_*` of exactly `nnz`
    entries and `sla_*` of `sla_cap`. A node contributes at most
    `min(dimensions, degree(node))` appended entries, so
    `sla_cap = nnz + sum_i min(dimensions, degree(i))` always suffices; that
    sum is at most `min(dimensions * num_nodes, total_degree)`, which is what
    the caller allocates.
    """
    var weight = _init_weight(decay, dimensions)
    _generate_hash_values(num_nodes, dimensions, seed, hashes, state)

    var nnz = 0
    for node in range(num_nodes):
        var e = Int(indptr.unsafe_load(node))
        var stop = Int(indptr.unsafe_load(node + 1))
        while e < stop:
            orig_rows.unsafe_store(nnz, Int32(node))
            orig_cols.unsafe_store(nnz, indices.unsafe_load(e))
            orig_vals.unsafe_store(nnz, 1.0)
            nnz += 1
            e += 1

    _do_single_sketch(
        num_nodes,
        dimensions,
        orig_rows,
        orig_cols,
        orig_vals,
        nnz,
        hashes,
        sketch,
        min_val,
    )
    var it = 0
    while it < iterations - 1:
        var sla_nnz = _augment_sla(
            num_nodes,
            dimensions,
            weight,
            indptr,
            indices,
            orig_rows,
            orig_cols,
            orig_vals,
            nnz,
            sla_rows,
            sla_cols,
            sla_vals,
            sla_cap,
            sketch,
            stamp,
            counts,
            touched,
            acc,
        )
        if sla_nnz < 0:
            return 1
        _do_single_sketch(
            num_nodes,
            dimensions,
            sla_rows,
            sla_cols,
            sla_vals,
            sla_nnz,
            hashes,
            sketch,
            min_val,
        )
        it += 1
    return 0


def get_embedding(
    num_nodes: Int, dimensions: Int, sketch: IPtr, embedding: Ptr
) -> None:
    """`np.transpose(self._sketch_to_np_array())`.

    `_sketch_to_np_array` is upstream's `np.array(self._sketch)`, a copy of a
    list of lists; the sketch is already a buffer, so the transpose is all that
    is left of it. The ids go out as Float64, the dtype the Python layer hands
    out embeddings in, and they are exact in it; the Int32 `sketch` buffer stays
    readable for callers that want the ids typed as they are upstream.
    """
    for node in range(num_nodes):
        for d in range(dimensions):
            embedding.unsafe_store(
                node * dimensions + d,
                Float64(sketch.unsafe_load(d * num_nodes + node)),
            )
