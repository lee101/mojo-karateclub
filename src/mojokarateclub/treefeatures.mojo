"""`WeisfeilerLehmanHashing` from `karateclub/utils/treefeatures.py`.

Upstream keeps a `dict` of per-node lists of strings and rebuilds every label
from a Python string concatenation and an `hashlib.md5` on every node of every
iteration. Here the strings are byte ranges and the digests are the four words
`hashing.md5` writes, so a recursion is a sort, a join into a caller-owned
staging buffer, and one MD5 per node. The digests come back as raw `UInt32`;
the caller hexlifies them and gets the same 32 characters `hexdigest()` does.

One divergence, confined to the signature: a label cannot cross the boundary as
a Python string, so `_set_features` is the caller's job. It passes the base
labels already `[str(v)]`-ed -- the `"feature"` node attributes, or
`{node: graph.degree(node)}` -- as one `UInt8` buffer of concatenated UTF-8
bytes plus an `Int32[n+1]` offset table, which is those `n` base labels
concatenated. Everything after that, including the `str()` of the label, the
`+` of the node's own label, the `sorted()` of the neighbour labels, the
`"_".join`, the md5, the erasure order and the node-major flattening order, is
here.

CSR row `k` must be exactly `graph.neighbors(k)`, so the Python layer feeds the
graph as it stands, before `Estimator._ensure_integrity` adds self-loops.

Layout of the result: node-major, one run of `wl_iterations + 1` slots per node,
slot 0 reserved for the base label (which the caller already holds) and slot
`1 + it` holding the four words of the digest produced by iteration `it`. The
table read in index order is `get_graph_features()`, and run `k` is
`get_node_features()[k]`. `erase_base_features` drops slot 0 of every run, so
afterwards the run is `wl_iterations` slots long and the leading words are
dead.

A label is carried as an address plus a length, and the erase flag is an
`Int`, not a struct: a `--emit shared-lib` build of this toolchain hands a `def`
callee the wrong bytes for a by-value struct parameter, which is silent and
reads as a numerical bug. (A `Bool` parameter was checked and crosses the
boundary correctly; the struct was the problem.)
"""

import hashing as hashing

comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

# `hashlib.md5(...).hexdigest()`: 32 lowercase hex chars over 16 digest bytes.
comptime WL_HEX_CHARS = 32
comptime WL_DIGEST_BYTES = 16

# A digest is four words, so a feature slot is four words, sixteen bytes.
comptime WL_SLOT_WORDS = 4
comptime WL_SLOT_BYTES = 16


def wl_label_data(
    base_bytes: Int,
    base_offsets: I32Ptr,
    extracted: Int,
    stride: Int,
    it: Int,
    node: Int,
) -> Int:
    """Address of the label of `node` going into iteration `it`: the base
    feature when `it` is 0, otherwise the digest the previous recursion left in
    slot `node * stride + it`, which is slot `1 + (it - 1)`.
    """
    if it == 0:
        return base_bytes + Int(base_offsets.unsafe_load(node))
    return extracted + (node * stride + it) * WL_SLOT_BYTES


def wl_label_len(base_offsets: I32Ptr, it: Int, node: Int) -> Int:
    """Length of that label's `str()` form: the base label's own byte count, or
    the 32 hex characters a digest stands for.
    """
    if it == 0:
        return Int(base_offsets.unsafe_load(node + 1)) - Int(
            base_offsets.unsafe_load(node)
        )
    return WL_HEX_CHARS


def wl_label_cmp(a_data: Int, a_len: Int, b_data: Int, b_len: Int, it: Int) -> Int:
    """`sorted` on the label strings, without materialising them.

    UTF-8 byte order is code point order, so a byte comparison of the base
    labels is the `str` comparison upstream does. Two digests are compared over
    their 16 bytes: `hexdigest` writes each digest byte as two hex chars in
    that same order, so the first differing byte is the first differing hex
    char, and equal byte prefixes stay equal string prefixes.
    """
    var common = min(a_len, b_len)
    if it > 0:
        common = min(common, WL_DIGEST_BYTES)
    var pa = U8Ptr(unsafe_from_address=a_data)
    var pb = U8Ptr(unsafe_from_address=b_data)
    for i in range(common):
        var x = Int(pa.unsafe_load(i))
        var y = Int(pb.unsafe_load(i))
        if x < y:
            return -1
        elif x > y:
            return 1
    if a_len < b_len:
        return -1
    elif a_len > b_len:
        return 1
    return 0


def wl_hex_digit(nibble: UInt32) -> UInt8:
    var d = Int(nibble & UInt32(0x0f))
    if d < 10:
        return UInt8(0x30 + d)
    return UInt8(0x61 + d - 10)


def wl_write_label(dst_addr: Int, data: Int, nchars: Int, it: Int) -> Int:
    """The `str()` form of a label at the absolute address `dst_addr`; returns
    bytes written.

    A digest is hexlified on the way, because the message upstream hashes is a
    concatenation of the 32-character strings, not of the 16 digest bytes.
    """
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    var src = U8Ptr(unsafe_from_address=data)
    if it > 0:
        for i in range(WL_DIGEST_BYTES):
            var b = UInt32(src.unsafe_load(i))
            dst.unsafe_offset(2 * i)[] = wl_hex_digit(b >> UInt32(4))
            dst.unsafe_offset(2 * i + 1)[] = wl_hex_digit(b)
        return WL_HEX_CHARS
    for i in range(nchars):
        dst.unsafe_offset(i)[] = src.unsafe_load(i)
    return nchars


def wl_merge(
    order: I32Ptr,
    order_tmp: I32Ptr,
    base_bytes: Int,
    base_offsets: I32Ptr,
    extracted: Int,
    stride: Int,
    it: Int,
    lo: Int,
    mid: Int,
    hi: Int,
):
    """Merge the sorted runs `order[lo:mid]` and `order[mid:hi]`, stably."""
    var i = lo
    var j = mid
    var k = lo
    while i < mid and j < hi:
        var a = Int(order.unsafe_load(i))
        var b = Int(order.unsafe_load(j))
        var cmp = wl_label_cmp(
            wl_label_data(base_bytes, base_offsets, extracted, stride, it, a),
            wl_label_len(base_offsets, it, a),
            wl_label_data(base_bytes, base_offsets, extracted, stride, it, b),
            wl_label_len(base_offsets, it, b),
            it,
        )
        if cmp <= 0:
            order_tmp.unsafe_offset(k)[] = order.unsafe_load(i)
            i += 1
        else:
            order_tmp.unsafe_offset(k)[] = order.unsafe_load(j)
            j += 1
        k += 1
    while i < mid:
        order_tmp.unsafe_offset(k)[] = order.unsafe_load(i)
        i += 1
        k += 1
    while j < hi:
        order_tmp.unsafe_offset(k)[] = order.unsafe_load(j)
        j += 1
        k += 1
    for p in range(lo, hi):
        order.unsafe_offset(p)[] = order_tmp.unsafe_load(p)


def wl_sort_labels(
    indptr: I32Ptr,
    indices: I32Ptr,
    order: I32Ptr,
    order_tmp: I32Ptr,
    base_bytes: Int,
    base_offsets: I32Ptr,
    extracted: Int,
    stride: Int,
    it: Int,
    node: Int,
    deg: Int,
):
    """`sorted([str(...) for neb in graph.neighbors(node)])`, as a permuted
    index list. Bottom-up merge sort over `deg` entries, so the comparison goes
    through the label accessors and no string is ever built.
    """
    var start = Int(indptr.unsafe_load(node))
    for i in range(deg):
        order.unsafe_offset(i)[] = indices.unsafe_load(start + i)
    var width = 1
    while width < deg:
        var lo = 0
        while lo < deg:
            wl_merge(
                order,
                order_tmp,
                base_bytes,
                base_offsets,
                extracted,
                stride,
                it,
                lo,
                min(lo + width, deg),
                min(lo + 2 * width, deg),
            )
            lo += 2 * width
        width *= 2


def wl_max_message_bytes(max_degree: Int, max_label_len: Int) -> Int:
    """Size of the `msg` staging buffer.

    A node's message is its own label, then one `"_"` and one neighbour label
    per neighbour, so the longest is `(max_degree + 1) * label + max_degree`.
    Every label from the first recursion on is a 32-character digest, so the
    bound never drops below that.
    """
    var chars = max(max_label_len, WL_HEX_CHARS)
    return (max_degree + 1) * chars + max_degree


def wl_digest_buffer_words(n: Int, wl_iterations: Int) -> Int:
    """Size of the `extracted` buffer: `n` runs of `wl_iterations + 1` slots.

    The base slot is written by nobody, but it is part of a run for as long as
    the recursions go, so the erased case still needs the full stride.
    """
    return n * (wl_iterations + 1) * WL_SLOT_WORDS


def wl_feature_stride(wl_iterations: Int, erase_base: Int) -> Int:
    """Slots per node in the result, after any erasure."""
    if erase_base != 0:
        return wl_iterations
    return wl_iterations + 1


def do_a_recursion(
    n: Int,
    indptr: I32Ptr,
    indices: I32Ptr,
    base_bytes: U8Ptr,
    base_offsets: I32Ptr,
    stride: Int,
    extracted: U32Ptr,
    msg: U8Ptr,
    order: I32Ptr,
    order_tmp: I32Ptr,
    m: U32Ptr,
    state: U32Ptr,
    tail: U8Ptr,
    it: Int,
):
    """The method does a single WL recursion.

    `degs = [features[neb] for neb in nebs]`, then
    `features = [str(features[node])] + sorted([str(d) for d in degs])`,
    `"_".join(features)`, `hashlib.md5(...).hexdigest()` -- with the joined
    string in `msg` and the four words of the digest in slot `1 + it` of node
    `k`'s run. `self.features = new_features` is the shift
    `wl_label_data` makes when `it` is not 0.
    """
    var base = Int(base_bytes)
    var table = Int(extracted)
    for node in range(n):
        var deg = Int(indptr.unsafe_load(node + 1)) - Int(indptr.unsafe_load(node))
        wl_sort_labels(
            indptr,
            indices,
            order,
            order_tmp,
            base,
            base_offsets,
            table,
            stride,
            it,
            node,
            deg,
        )

        var at = wl_write_label(
            Int(msg),
            wl_label_data(base, base_offsets, table, stride, it, node),
            wl_label_len(base_offsets, it, node),
            it,
        )
        for i in range(deg):
            msg.unsafe_offset(at)[] = UInt8(0x5f)
            at += 1
            var neb = Int(order.unsafe_load(i))
            at += wl_write_label(
                Int(msg) + at,
                wl_label_data(base, base_offsets, table, stride, it, neb),
                wl_label_len(base_offsets, it, neb),
                it,
            )

        hashing.md5(
            msg,
            at,
            U32Ptr(
                unsafe_from_address=table + (node * stride + 1 + it) * WL_SLOT_BYTES
            ),
            m,
            state,
            tail,
        )


def erase_base_features(extracted: U32Ptr, n: Int, stride: Int):
    """Erasing the base features: `del extracted_features[k][0]`, for every
    node. The base label is the caller's, so dropping element 0 of every run is
    a shift of the digest slots down by one.
    """
    var new_stride = stride - 1
    var new_words = new_stride * WL_SLOT_WORDS
    for node in range(n):
        # element 0 of the run is `stride` slots in; everything after it moves
        # down by one slot
        var src = node * stride * WL_SLOT_WORDS + WL_SLOT_WORDS
        var dst = node * new_words
        for word in range(new_words):
            extracted.unsafe_offset(dst + word)[] = extracted.unsafe_load(src + word)


def do_recursions(
    n: Int,
    indptr: I32Ptr,
    indices: I32Ptr,
    base_bytes: U8Ptr,
    base_offsets: I32Ptr,
    wl_iterations: Int,
    extracted: U32Ptr,
    msg: U8Ptr,
    order: I32Ptr,
    order_tmp: I32Ptr,
    m: U32Ptr,
    state: U32Ptr,
    tail: U8Ptr,
    erase_base: Int,
):
    """The method does a series of WL recursions."""
    var stride = wl_iterations + 1
    for it in range(wl_iterations):
        do_a_recursion(
            n,
            indptr,
            indices,
            base_bytes,
            base_offsets,
            stride,
            extracted,
            msg,
            order,
            order_tmp,
            m,
            state,
            tail,
            it,
        )
    if erase_base != 0:
        erase_base_features(extracted, n, stride)


def wl_hash(
    n: Int,
    indptr: I32Ptr,
    indices: I32Ptr,
    base_bytes: U8Ptr,
    base_offsets: I32Ptr,
    wl_iterations: Int,
    erase_base: Int,
    extracted: U32Ptr,
    msg: U8Ptr,
    order: I32Ptr,
    order_tmp: I32Ptr,
    m: U32Ptr,
    state: U32Ptr,
    tail: U8Ptr,
):
    """`__init__`: `_set_features()` then `_do_recursions()`.

    `indptr` / `indices` are the CSR of the graph, row `k` being exactly
    `graph.neighbors(k)`. `base_bytes` / `base_offsets` are what `_set_features`
    produced, `base_offsets[n]` being the length of `base_bytes`.
    `erase_base` is `erase_base_features` as 0 or 1.

    `extracted` is the node-major feature table of `wl_digest_buffer_words`
    words: read it in index order for `get_graph_features()`, and read run `k`
    of `wl_feature_stride(wl_iterations, erase_base)` slots for
    `get_node_features()[k]`, hexlifying a digest slot from its four words and
    taking a base slot from `base_bytes` at `base_offsets[k]`.

    `msg`, `order` and `order_tmp` are scratch: `wl_max_message_bytes(max_degree,
    longest_base_label)` bytes, and `max_degree` `Int32` each. `m`, `state` and
    `tail` are the `hashing` MD5 scratch.
    """
    do_recursions(
        n,
        indptr,
        indices,
        base_bytes,
        base_offsets,
        wl_iterations,
        extracted,
        msg,
        order,
        order_tmp,
        m,
        state,
        tail,
        erase_base,
    )
