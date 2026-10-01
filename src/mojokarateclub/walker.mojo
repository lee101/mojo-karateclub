"""First- and second-order random walks, the PRNG they draw from, and the CSR
row each step reads.

Port of `karateclub/utils/walker.py`. Upstream walks a live `networkx.Graph`
and returns lists of `str`; a Mojo kernel holds neither, so the graph arrives as
a CSR `(indptr, indices[, values])`, the per-class configuration arrives as
explicit arguments, and the walk comes back as a flat `Int32` of CSR row
indices which the caller formats with `str(...)`, as upstream's
`[str(w) for w in walk]` does. That is the only divergence besides the draws.

The draws are a divergence too. Upstream mixes `random.sample` (CPython's
`_random`: MT19937, `init_by_array` seeding, rejection sampling on getrandbits)
with `np.random.choice` (numpy's legacy `RandomState`: MT19937 with a different
array seeding and a CDF search), and matching either bit for bit would buy
nothing but a dependency on CPython's private generator. This module carries
MT19937 with `init_genrand` seeding instead, laid out the way `mt19937ar.c`
and numpy's legacy state both lay it out, so the Python layer can mirror it in
NumPy and diff the walks. `mt_init_genrand` and `mt_genrand_uint32` exist for
that mirror; the layout is documented on `mt_init_genrand`.

One thing to know before reading `biased_random_walker_do_walk`: numpy's
`piecewise` assigns rather than accumulates, so the LAST matching condition
wins, not the first. On any graph carrying the self loop every estimator adds,
that makes upstream's `p` inert. It is reproduced, not corrected.

Nothing allocates. The walk, the PRNG state and every scratch buffer are the
caller's.
"""

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]

comptime MT_N = 624
comptime MT_M = 397
comptime MT_MATRIX_A: UInt32 = UInt32(0x9908b0df)
comptime MT_UPPER_MASK: UInt32 = UInt32(0x80000000)
comptime MT_LOWER_MASK: UInt32 = UInt32(0x7fffffff)
# mt[0:624] then the index, as `mt19937ar.c` numbers them.
comptime MT_STATE_WORDS = MT_N + 1


# ------------------------------------------------- random / np.random

def bit_length(n: Int) -> Int:
    """`int.bit_length`; this dialect's stdlib has no such spelling."""
    var k = 0
    var v = n
    if v >= 65536:
        k += 16
        v = v >> 16
    if v >= 256:
        k += 8
        v = v >> 8
    if v >= 16:
        k += 4
        v = v >> 4
    if v >= 4:
        k += 2
        v = v >> 2
    if v >= 2:
        k += 1
    return k + 1


def mt_init_genrand(state: U32Ptr, s: UInt32):
    """`init_genrand` from mt19937ar.c, which is also numpy's scalar seeding.

    The state is `MT_STATE_WORDS` words, laid out as `mt[0 .. MT_N - 1]`
    followed by the draw index `mti`, so a mirror is a transcription: word `i`
    is `state[i]` and the index is `state[624]`. The index starts saturated at
    `MT_N`, which is what forces the twist on the first draw.
    """
    state.unsafe_offset(0)[] = s
    var i = 1
    while i < MT_N:
        var prev = state.unsafe_load(i - 1)
        state.unsafe_offset(i)[] = (
            UInt32(1812433253) * (prev ^ (prev >> UInt32(30))) + UInt32(i)
        )
        i += 1
    state.unsafe_offset(MT_N)[] = UInt32(MT_N)


def mt_genrand_uint32(state: U32Ptr) -> UInt32:
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
        # mt[0] has already been overwritten by the first loop, which is what
        # the reference does.
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


def mt_genrand_res53(state: U32Ptr) -> Float64:
    """Two tempered words as a double in [0, 1), the draw `np.random.choice`
    makes with `RandomState.random_sample`."""
    var a = mt_genrand_uint32(state) >> UInt32(5)
    var b = mt_genrand_uint32(state) >> UInt32(6)
    return (Float64(a) * 67108864.0 + Float64(b)) * (1.0 / 9007199254740992.0)


def randbelow(state: U32Ptr, n: Int) -> Int:
    """`random._randbelow`: the number of bits is rounded up to a whole byte
    and the draw is rejected until it fits, so even `n == 1` consumes draws."""
    if n <= 0:
        return 0
    var k = bit_length(n)
    if k % 8 == 0:
        k += 8
    while True:
        var r = mt_genrand_uint32(state) >> UInt32(32 - k)
        if Int(r) < n:
            return Int(r)


def sample_one(state: U32Ptr, indices: IPtr, start: Int, n: Int) -> Int32:
    """`random.sample(seq, 1)`.

    CPython pools the population and swaps the picked element with the tail
    before the next pick; at `k == 1` that swap has no second pick to hide
    behind, so the call is `pool[randbelow(n)]` and the sample is uniform over
    the row. The rejection sampling is `randbelow`'s, so a row of two can burn
    a variable number of words exactly as upstream does.
    """
    return indices.unsafe_load(start + randbelow(state, n))


# ------------------------------------------------------- RandomWalker

def random_walker_do_walk(
    indptr: IPtr,
    indices: IPtr,
    node: Int,
    walk_length: Int,
    walk: IPtr,
    state: U32Ptr,
) -> Int:
    """One truncated first-order walk from `node`, into the caller's
    `walk[0:walk_length]`, returning how many ids it wrote.

    A step from a node with no neighbours appends nothing, so the walk comes
    back shorter than `walk_length` rather than failing; upstream has the same
    `if len(nebs) > 0` guard.
    """
    var walk_size = 1
    walk.unsafe_offset(0)[] = Int32(node)
    for _ in range(walk_length - 1):
        var current = Int(walk.unsafe_load(walk_size - 1))
        var nebs = neighbors(indptr, indices, current)
        if nebs[1] > nebs[0]:
            walk.unsafe_offset(walk_size)[] = sample_one(
                state, indices, nebs[0], nebs[1] - nebs[0]
            )
            walk_size += 1
    return walk_size


def random_walker_do_walks(
    indptr: IPtr,
    indices: IPtr,
    n: Int,
    walk_length: Int,
    walk_number: Int,
    walk: IPtr,
    state: U32Ptr,
) -> Int:
    """`walk_number` walks from every node in node order, concatenated into the
    caller's `walk[0: n * walk_number * walk_length]`, returning the number of
    ids written."""
    var total = 0
    for node in range(n):
        for _ in range(walk_number):
            total += random_walker_do_walk(
                indptr, indices, node, walk_length, walk.unsafe_offset(total), state
            )
    return total


# ------------------------------------------------------- _check_value

def check_value(value: Float64) -> Bool:
    """`_check_value`, without the exception: upstream tries `1 / value` and
    re-raises `ValueError`, so the check is exactly "is it zero"."""
    if value == 0.0:
        return False
    return True


# ------------------------------------- _undirected / _directed / _weight

def neighbors(indptr: IPtr, indices: IPtr, node: Int) -> Tuple[Int, Int]:
    """`_undirected` and `_directed` collapsed onto one CSR row.

    Upstream picks the edge function with `_get_edge_fn` and then reads
    `edge[1]`: for an undirected graph that is the other end, for a directed
    one the out-edge target. Both are the out-neighbours, which is what a CSR
    row is, so the branch disappears here. The caller only wants the ends, so
    this hands back the row as a half-open range instead of materialising it.
    """
    return (Int(indptr.unsafe_load(node)), Int(indptr.unsafe_load(node + 1)))


def edge_weight(values: Ptr, k: Int, weighted: Bool) -> Float64:
    """`_weighted` and `_unweighted` collapsed onto the CSR value array.

    `weighted` is `_get_weight_fn`'s dispatch. `values` still has to be a valid
    non-null address when it is false, because a Mojo pointer cannot be null;
    only `k` positions are read when it is true.
    """
    if weighted:
        return values.unsafe_load(k)
    return Float64(1)


# --------------------------------------------------- BiasedRandomWalker

def biased_random_walker_do_walk(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    weighted: Bool,
    node: Int,
    walk_length: Int,
    p: Float64,
    q: Float64,
    walk: IPtr,
    state: U32Ptr,
    weights: Ptr,
    mark: IPtr,
) -> Int:
    """One truncated second-order walk from `node`, into the caller's
    `walk[0:walk_length]`, returning how many ids it wrote.

    `weights` is scratch of at least the widest row in `Float64`, `mark` is
    scratch of at least `n` in `Int32` and must be all zero on entry; this
    restores it to all zero before it returns, so a caller may chain walks.
    Note that `p` is dead upstream on any graph carrying the self loop
    `_ensure_integrity` adds, because `np.piecewise` lets the last matching
    condition win and the previous node is always one of the previous node's
    own neighbours. That is reproduced, not fixed.
    """
    var walk_size = 1
    walk.unsafe_offset(0)[] = Int32(node)
    # `previous_node` is upstream's None: no CSR row index equals -1, so the
    # first condition cannot fire on the first step. `previous_node_neighbors`
    # is upstream's empty list, represented by the empty row [0, 0).
    var previous_node = -1
    var previous_row = 0
    var previous_stop = 0

    var k = 0
    for _ in range(walk_length - 1):
        var current = Int(walk.unsafe_load(walk_size - 1))
        var row = neighbors(indptr, indices, current)
        var start = row[0]
        var stop = row[1]

        if stop > start:
            var total = Float64(0)
            var i = 0
            while i < stop - start:
                var nb = Int(indices.unsafe_load(start + i))
                var weight = edge_weight(values, start + i, weighted)
                # np.piecewise ASSIGNS, `y[cond] = func(x[cond])`, and its
                # third function is the "otherwise" branch
                # `~any(condlist)`, so the LAST matching condition is the one
                # that survives, not the first: a neighbour of the previous
                # node is taken with `w / 1` even when it also is the previous
                # node, and `w / p` only survives for a previous node that is
                # not among the previous node's own neighbours. With the self
                # loop every estimator adds, the previous node is always among
                # them, so `p` never fires upstream either. Keeping the branch
                # order the same way round makes that visible here instead of
                # hiding it.
                if mark.unsafe_load(nb) != 0:
                    weight = weight / 1.0
                elif nb == previous_node:
                    weight = weight / p
                else:
                    weight = weight / q
                weights.unsafe_offset(i)[] = weight
                total += weight
                i += 1
            if total > 0.0:
                # `np.random.choice(a, 1, p=norm)`: one uniform double, then
                # the first index whose cumulative share is above it. `side`
                # is "right", hence the strict `>` and the off-the-end clamp.
                var u = mt_genrand_res53(state)
                var running = Float64(0)
                var selected = indices.unsafe_load(stop - 1)
                i = 0
                while i < stop - start:
                    running += weights.unsafe_load(i)
                    if running / total > u:
                        selected = indices.unsafe_load(start + i)
                        break
                    i += 1
                walk.unsafe_offset(walk_size)[] = selected
                walk_size += 1

        # previous_node_neighbors = current_node_neighbors
        # previous_node = current_node
        # The window is the row, so it is marked rather than copied. Clearing
        # before marking is what makes a node in both windows stay marked.
        var k = previous_row
        while k < previous_stop:
            mark.unsafe_offset(indices.unsafe_load(k))[] = Int32(0)
            k += 1
        k = start
        while k < stop:
            mark.unsafe_offset(indices.unsafe_load(k))[] = Int32(1)
            k += 1
        previous_node = current
        previous_row = start
        previous_stop = stop

    k = previous_row
    while k < previous_stop:
        mark.unsafe_offset(indices.unsafe_load(k))[] = Int32(0)
        k += 1
    return walk_size


def biased_random_walker_do_walks(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    weighted: Bool,
    n: Int,
    walk_length: Int,
    walk_number: Int,
    p: Float64,
    q: Float64,
    walk: IPtr,
    state: U32Ptr,
    weights: Ptr,
    mark: IPtr,
) -> Int:
    """`walk_number` walks from every node in node order, concatenated into the
    caller's `walk[0: n * walk_number * walk_length]`, returning the number of
    ids written."""
    var total = 0
    for node in range(n):
        for _ in range(walk_number):
            total += biased_random_walker_do_walk(
                indptr,
                indices,
                values,
                weighted,
                node,
                walk_length,
                p,
                q,
                walk.unsafe_offset(total),
                state,
                weights,
                mark,
            )
    return total
