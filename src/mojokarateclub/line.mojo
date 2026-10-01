"""First-order and second-order LINE, from `node_embedding/neighbourhood/`.

Ported from `first_order_line.py` and `second_order_line.py`. The graph arrives
already as the edge list upstream builds with `np.array(graph.edges(data=False))`:
an `Int32[number_of_edges, 2]` block in the same order, with both directions
present. `number_of_nodes` and `number_of_edges` are the arguments upstream
reads off the graph.

Two things here are not obvious and are not to be "fixed":

* `gradient[edges[:, 0]] += activations.reshape(-1, 1) * dst_embedding` is a
  fancy-index augmented assignment, which numpy evaluates as
  getitem-copy, iadd, setitem, and the copy is load bearing. A repeated index
  takes only the LAST of the batch's deltas instead of summing them, while a
  row touched by two different statements does add up. Verified on the pinned
  numpy 1.22.4: `a[[0, 0, 1]] += [1, 2, 3]` is `[2, 3, 0]`, not `[3, 3, 0]`,
  and a second statement over a row the first one wrote adds to it. See
  `_scatter_add`. Replacing it with an accumulate (what `np.add.at` does, and
  what the syntax suggests) silently breaks parity with every upstream
  document, so the positive half of the mini-batch is written first and the
  negative half second, in that order.
* The initial embedding and both batches come from `np.random.*` after
  `np.random.seed(seed)`, so the legacy MT19937 stream is part of the model's
  output. It is implemented here rather than left to the caller: `init_genrand`,
  `random_sample` and `randint` reproduce `numpy.random.RandomState(seed)` draw
  for draw, with `randint` using numpy's masked rejection loop
  (`gen_mask(high - 1)`, reject while `value > high - 1`).

The two upstream classes both name their methods `_update` and `fit`, so the
first-order pair keeps the bare names and the second-order pair carries a
`_second_order` suffix.

The sigmoid goes through `mathfn.natural_exp` rather than `std.math.exp`, which
is SLEEF and a 1e-10 relative error there would land straight in the gradient.

Nothing here allocates. Scratch is caller-owned and passed as trailing
arguments; the sizes each `fit` needs are spelled out in its docstring.
"""

import mathfn

import kclinalg as linalg

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime MT = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U32 = UInt32

# The state is 624 words plus the index into it, so that a caller can carve one
# UInt32 block and never has to thread an extra scalar through the ABI.
comptime MT_N = 624
comptime MT_M = 397
comptime MT_SCRATCH_WORDS = 625


# --------------------------------------------------------------------------
# numpy's legacy MT19937 (`numpy/random/src/mt19937/mt19937.c`)


def init_genrand(state: MT, seed: UInt32):
    """`init_genrand` of the MT19937 reference, which is what `np.random.seed`
    calls for an integer seed."""
    state.unsafe_store(0, seed)
    for i in range(1, MT_N):
        var prev = state.unsafe_load(i - 1)
        state.unsafe_store(
            i, U32(1812433253) * (prev ^ (prev >> U32(30))) + U32(i)
        )
    state.unsafe_store(MT_N, U32(MT_N))


def genrand_uint32(state: MT) -> U32:
    """One tempered 32-bit word, regenerating the 624-word block when needed."""
    var idx = Int(state.unsafe_load(MT_N))
    if idx >= MT_N:
        for i in range(MT_N):
            var j = i + 1
            if j == MT_N:
                j = 0
            var k = i + MT_M
            if k >= MT_N:
                k = k - MT_N
            var y = (state.unsafe_load(i) & U32(0x80000000)) | (
                state.unsafe_load(j) & U32(0x7fffffff)
            )
            var next = state.unsafe_load(k) ^ (y >> U32(1))
            if y & U32(1) != U32(0):
                next = next ^ U32(0x9908b0df)
            state.unsafe_store(i, next)
        idx = 0
    var y = state.unsafe_load(idx)
    state.unsafe_store(MT_N, U32(idx + 1))
    y = y ^ (y >> U32(11))
    y = y ^ ((y << U32(7)) & U32(0x9d2c5680))
    y = y ^ ((y << U32(15)) & U32(0xefc60000))
    return y ^ (y >> U32(18))


def random_sample(state: MT) -> Float64:
    """`random_sample()`: the 53-bit double built out of two tempered words."""
    var a = Float64(genrand_uint32(state) >> U32(5))
    var b = Float64(genrand_uint32(state) >> U32(6))
    return (a * 67108864.0 + b) / 9007199254740992.0


def uniform(state: MT, dst: Ptr, n: Int):
    """`np.random.uniform(size=...)` into a flat `dst[n]`, in C order.

    `low=0, high=1` makes the affine transform the identity, so this is
    `random_sample` element for element.
    """
    for i in range(n):
        dst.unsafe_store(i, random_sample(state))


def gen_mask(rng: U32) -> U32:
    """`gen_mask`: the smallest bit mask that covers `rng`."""
    var mask = rng | (rng >> U32(1))
    mask = mask | (mask >> U32(2))
    mask = mask | (mask >> U32(4))
    mask = mask | (mask >> U32(8))
    return mask | (mask >> U32(16))


def randint(state: MT, high: Int, dst: IPtr, count: Int):
    """`np.random.randint(high, size=count)` for the legacy generator.

    `low` is 0 and `high` is exclusive, so the interval is `[0, high)` and the
    bound numpy masks against is `high - 1`. Rejections consume a word and
    retry, which is why the stream position after a batch is not predictable
    from the batch size alone.
    """
    var rng = U32(high - 1)
    var mask = gen_mask(rng)
    for i in range(count):
        var value = genrand_uint32(state) & mask
        while value > rng:
            value = genrand_uint32(state) & mask
        dst.unsafe_store(i, Int32(value))


# --------------------------------------------------------------------------
# FirstOrderLINE._update / fit


def _scatter_add(
    gradient: Ptr,
    edges: IPtr,
    column: Int,
    batch: Int,
    dimensions: Int,
    activations: Ptr,
    gathered: Ptr,
    base: Ptr,
):
    """`gradient[edges[:, column]] += activations.reshape(-1, 1) * gathered`.

    numpy runs a fancy-index `+=` as copy, in-place add, copy back, and the
    copy is load bearing. A row keeps whatever the gradient held *before* this
    statement and takes only the last delta of the batch that repeats it, so
    two statements over the same row add up while a repeated row within one
    statement does not. `np.add.at` is the accumulating one; it is not what
    upstream calls. `base` is that copy and `gathered` the rows the deltas are
    read from.
    """
    for i in range(batch):
        var row = Int(edges.unsafe_load(2 * i + column)) * dimensions
        for j in range(dimensions):
            base.unsafe_store(i * dimensions + j, gradient.unsafe_load(row + j))

    for i in range(batch):
        var row = Int(edges.unsafe_load(2 * i + column)) * dimensions
        var activation = activations.unsafe_load(i)
        for j in range(dimensions):
            gradient.unsafe_store(
                row + j,
                base.unsafe_load(i * dimensions + j)
                + activation * gathered.unsafe_load(i * dimensions + j),
            )


def _update(
    embedding: Ptr,
    edges: IPtr,
    gradient: Ptr,
    label: Int,
    batch: Int,
    dimensions: Int,
    src_embedding: Ptr,
    dst_embedding: Ptr,
    activations: Ptr,
    base: Ptr,
):
    """`FirstOrderLINE._update`: one batch against `gradient`.

    The two scatters go through `_scatter_add` because `gradient[edges[:, k]]
    += ...` is not an accumulating scatter in numpy; read that first.

    `edges` is `batch x 2`. Scratch: `src_embedding[batch, dimensions]`,
    `dst_embedding[batch, dimensions]`, `activations[batch]` and
    `base[batch, dimensions]`, all row-major.
    """
    for i in range(batch):
        var src = Int(edges.unsafe_load(2 * i)) * dimensions
        var dst = Int(edges.unsafe_load(2 * i + 1)) * dimensions
        for j in range(dimensions):
            src_embedding.unsafe_store(
                i * dimensions + j, embedding.unsafe_load(src + j)
            )
            dst_embedding.unsafe_store(
                i * dimensions + j, embedding.unsafe_load(dst + j)
            )

    for i in range(batch):
        var dots = linalg.dot(
            src_embedding.unsafe_offset(i * dimensions),
            dst_embedding.unsafe_offset(i * dimensions),
            dimensions,
        )
        activations.unsafe_store(
            i, 1.0 / (1.0 + mathfn.natural_exp(-dots)) - Float64(label)
        )

    _scatter_add(
        gradient, edges, 0, batch, dimensions, activations, dst_embedding, base
    )
    _scatter_add(
        gradient, edges, 1, batch, dimensions, activations, src_embedding, base
    )


def fit_first_order(
    number_of_nodes: Int,
    dimensions: Int,
    epochs: Int,
    mini_batch_size: Int,
    learning_rate: Float64,
    learning_rate_decay: Float64,
    number_of_edges: Int,
    edges: IPtr,
    seed: Int,
    embedding: Ptr,
    gradient: Ptr,
    positive_index: IPtr,
    positive_batch: IPtr,
    negative_batch: IPtr,
    src_embedding: Ptr,
    dst_embedding: Ptr,
    activations: Ptr,
    base: Ptr,
    state: MT,
):
    """`FirstOrderLINE.fit`. The embedding is written in place, not returned.

    Scratch: `gradient[number_of_nodes, dimensions]`,
    `positive_index[mini_batch_size // 2]`, `positive_batch[half, 2]` and
    `negative_batch[half, 2]`, `src_embedding[half, dimensions]`,
    `dst_embedding[half, dimensions]`, `activations[half]`, `base[half,
    dimensions]`, and the `MT_SCRATCH_WORDS` generator state, where
    `half = mini_batch_size // 2`.
    """
    init_genrand(state, U32(seed))
    uniform(state, embedding, number_of_nodes * dimensions)

    var half = mini_batch_size // 2
    for epoch in range(epochs):
        for _ in range(number_of_edges // mini_batch_size):
            linalg.fill(gradient, 0.0, number_of_nodes * dimensions)

            randint(state, number_of_edges, positive_index, half)
            for i in range(half):
                var edge = Int(positive_index.unsafe_load(i))
                positive_batch.unsafe_store(2 * i, edges.unsafe_load(2 * edge))
                positive_batch.unsafe_store(2 * i + 1, edges.unsafe_load(2 * edge + 1))
            _update(
                embedding,
                positive_batch,
                gradient,
                1,
                half,
                dimensions,
                src_embedding,
                dst_embedding,
                activations,
                base,
            )

            randint(state, number_of_nodes, negative_batch, half * 2)
            _update(
                embedding,
                negative_batch,
                gradient,
                0,
                half,
                dimensions,
                src_embedding,
                dst_embedding,
                activations,
                base,
            )

            var rate = learning_rate * (learning_rate_decay ** Float64(epoch))
            linalg.axpy(-rate, gradient, embedding, number_of_nodes * dimensions)


# --------------------------------------------------------------------------
# SecondOrderLINE._update / fit


def _update_second_order(
    src_embedding: Ptr,
    dst_embedding: Ptr,
    edges: IPtr,
    src_gradient: Ptr,
    dst_gradient: Ptr,
    label: Int,
    batch: Int,
    dimensions: Int,
    src_batch: Ptr,
    dst_batch: Ptr,
    activations: Ptr,
    base: Ptr,
):
    """`SecondOrderLINE._update`: the same batch, against two tables.

    Upstream shadows its two arguments with the gathered rows, so the gather
    happens before the dots and the scatter reads the gathered rows; the two
    scatters go through `_scatter_add` for the reason given there. `edges` is
    `batch x 2`. Scratch: `src_batch[batch, dimensions]`,
    `dst_batch[batch, dimensions]`, `activations[batch]`, `base[batch,
    dimensions]`.
    """
    for i in range(batch):
        var src = Int(edges.unsafe_load(2 * i)) * dimensions
        var dst = Int(edges.unsafe_load(2 * i + 1)) * dimensions
        for j in range(dimensions):
            src_batch.unsafe_store(
                i * dimensions + j, src_embedding.unsafe_load(src + j)
            )
            dst_batch.unsafe_store(
                i * dimensions + j, dst_embedding.unsafe_load(dst + j)
            )

    for i in range(batch):
        var dots = linalg.dot(
            src_batch.unsafe_offset(i * dimensions),
            dst_batch.unsafe_offset(i * dimensions),
            dimensions,
        )
        activations.unsafe_store(
            i, 1.0 / (1.0 + mathfn.natural_exp(-dots)) - Float64(label)
        )

    _scatter_add(
        src_gradient, edges, 0, batch, dimensions, activations, dst_batch, base
    )
    _scatter_add(
        dst_gradient, edges, 1, batch, dimensions, activations, src_batch, base
    )


def fit_second_order(
    number_of_nodes: Int,
    dimensions: Int,
    epochs: Int,
    mini_batch_size: Int,
    learning_rate: Float64,
    learning_rate_decay: Float64,
    number_of_edges: Int,
    edges: IPtr,
    seed: Int,
    src_embedding: Ptr,
    dst_embedding: Ptr,
    src_gradient: Ptr,
    dst_gradient: Ptr,
    positive_index: IPtr,
    positive_batch: IPtr,
    negative_batch: IPtr,
    src_batch: Ptr,
    dst_batch: Ptr,
    activations: Ptr,
    base: Ptr,
    state: MT,
):
    """`SecondOrderLINE.fit`. Both tables are written in place, not returned.

    Each table is `number_of_nodes, dimensions // 2`, as upstream has it, so an
    odd `dimensions` loses a column exactly as it does there. Scratch: the two
    gradients, `positive_index[half]`, `positive_batch[half, 2]`,
    `negative_batch[half, 2]`, `src_batch[half, dimensions // 2]`,
    `dst_batch[half, dimensions // 2]`, `activations[half]`,
    `base[half, dimensions // 2]` and the generator state, where
    `half = mini_batch_size // 2`.
    """
    init_genrand(state, U32(seed))
    var half_dimensions = dimensions // 2
    var cells = number_of_nodes * half_dimensions
    uniform(state, src_embedding, cells)
    uniform(state, dst_embedding, cells)

    var half = mini_batch_size // 2
    for epoch in range(epochs):
        for _ in range(number_of_edges // mini_batch_size):
            linalg.fill(src_gradient, 0.0, cells)
            linalg.fill(dst_gradient, 0.0, cells)

            randint(state, number_of_edges, positive_index, half)
            for i in range(half):
                var edge = Int(positive_index.unsafe_load(i))
                positive_batch.unsafe_store(2 * i, edges.unsafe_load(2 * edge))
                positive_batch.unsafe_store(2 * i + 1, edges.unsafe_load(2 * edge + 1))
            _update_second_order(
                src_embedding,
                dst_embedding,
                positive_batch,
                src_gradient,
                dst_gradient,
                1,
                half,
                half_dimensions,
                src_batch,
                dst_batch,
                activations,
                base,
            )

            randint(state, number_of_nodes, negative_batch, half * 2)
            _update_second_order(
                src_embedding,
                dst_embedding,
                negative_batch,
                src_gradient,
                dst_gradient,
                0,
                half,
                half_dimensions,
                src_batch,
                dst_batch,
                activations,
                base,
            )

            var rate = learning_rate * (learning_rate_decay ** Float64(epoch))
            linalg.axpy(-rate, src_gradient, src_embedding, cells)
            linalg.axpy(-rate, dst_gradient, dst_embedding, cells)
