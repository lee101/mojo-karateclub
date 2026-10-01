"""RFC 1321 MD5, the digest `hashlib.md5` produces.

Not upstream code: karateclub reaches for `hashlib` inside
`WeisfeilerLehmanHashing._do_a_recursion`, once per node per iteration. That
call is the whole cost of the Weisfeiler-Lehman feature extractor, so it is
worth doing here. The output is bit-identical to Python's, which is what makes
`Graph2Vec` documents comparable with upstream's.

`md5_k` and `md5_s` are the RFC's `T[i] = floor(2**32 * abs(sin(i + 1)))` and
per-round rotation tables, spelled out because this toolchain has no comptime
array literal. The message buffer is never copied: the tail is staged in a
128-byte stack buffer so the one and two block cases share one code path.
"""

comptime U32 = UInt32
comptime W32Ptr = Pointer[U32, AnyOrigin[mut=True]]
comptime SU8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

comptime MD5_SCRATCH_WORDS = 64
comptime MD5_SCRATCH_BYTES = 128


def md5_k(i: Int) -> U32:
    if i == 0:
        return U32(0xd76aa478)
    elif i == 1:
        return U32(0xe8c7b756)
    elif i == 2:
        return U32(0x242070db)
    elif i == 3:
        return U32(0xc1bdceee)

    elif i == 4:
        return U32(0xf57c0faf)
    elif i == 5:
        return U32(0x4787c62a)
    elif i == 6:
        return U32(0xa8304613)
    elif i == 7:
        return U32(0xfd469501)
    elif i == 8:
        return U32(0x698098d8)
    elif i == 9:
        return U32(0x8b44f7af)
    elif i == 10:
        return U32(0xffff5bb1)
    elif i == 11:
        return U32(0x895cd7be)
    elif i == 12:
        return U32(0x6b901122)
    elif i == 13:
        return U32(0xfd987193)
    elif i == 14:
        return U32(0xa679438e)
    elif i == 15:
        return U32(0x49b40821)
    elif i == 16:
        return U32(0xf61e2562)
    elif i == 17:
        return U32(0xc040b340)
    elif i == 18:
        return U32(0x265e5a51)
    elif i == 19:
        return U32(0xe9b6c7aa)
    elif i == 20:
        return U32(0xd62f105d)
    elif i == 21:
        return U32(0x02441453)
    elif i == 22:
        return U32(0xd8a1e681)
    elif i == 23:
        return U32(0xe7d3fbc8)
    elif i == 24:
        return U32(0x21e1cde6)
    elif i == 25:
        return U32(0xc33707d6)
    elif i == 26:
        return U32(0xf4d50d87)
    elif i == 27:
        return U32(0x455a14ed)
    elif i == 28:
        return U32(0xa9e3e905)
    elif i == 29:
        return U32(0xfcefa3f8)
    elif i == 30:
        return U32(0x676f02d9)
    elif i == 31:
        return U32(0x8d2a4c8a)
    elif i == 32:
        return U32(0xfffa3942)
    elif i == 33:
        return U32(0x8771f681)
    elif i == 34:
        return U32(0x6d9d6122)
    elif i == 35:
        return U32(0xfde5380c)
    elif i == 36:
        return U32(0xa4beea44)
    elif i == 37:
        return U32(0x4bdecfa9)
    elif i == 38:
        return U32(0xf6bb4b60)
    elif i == 39:
        return U32(0xbebfbc70)
    elif i == 40:
        return U32(0x289b7ec6)
    elif i == 41:
        return U32(0xeaa127fa)
    elif i == 42:
        return U32(0xd4ef3085)
    elif i == 43:
        return U32(0x04881d05)
    elif i == 44:
        return U32(0xd9d4d039)
    elif i == 45:
        return U32(0xe6db99e5)
    elif i == 46:
        return U32(0x1fa27cf8)
    elif i == 47:
        return U32(0xc4ac5665)
    elif i == 48:
        return U32(0xf4292244)
    elif i == 49:
        return U32(0x432aff97)
    elif i == 50:
        return U32(0xab9423a7)
    elif i == 51:
        return U32(0xfc93a039)
    elif i == 52:
        return U32(0x655b59c3)
    elif i == 53:
        return U32(0x8f0ccc92)
    elif i == 54:
        return U32(0xffeff47d)
    elif i == 55:
        return U32(0x85845dd1)
    elif i == 56:
        return U32(0x6fa87e4f)
    elif i == 57:
        return U32(0xfe2ce6e0)
    elif i == 58:
        return U32(0xa3014314)
    elif i == 59:
        return U32(0x4e0811a1)
    elif i == 60:
        return U32(0xf7537e82)
    elif i == 61:
        return U32(0xbd3af235)
    elif i == 62:
        return U32(0x2ad7d2bb)
    else:
        return U32(0xeb86d391)


def md5_s(i: Int) -> Int:
    """Per-round rotation amount.

    Each of the four rounds cycles four constants, and no cycle is an
    arithmetic progression, so both the round and the slot are looked up.
    """
    var r: Int = i % 4
    if i < 16:
        if r == 0:
            return 7
        elif r == 1:
            return 12
        elif r == 2:
            return 17
        else:
            return 22
    elif i < 32:
        if r == 0:
            return 5
        elif r == 1:
            return 9
        elif r == 2:
            return 14
        else:
            return 20
    elif i < 48:
        if r == 0:
            return 4
        elif r == 1:
            return 11
        elif r == 2:
            return 16
        else:
            return 23
    else:
        if r == 0:
            return 6
        elif r == 1:
            return 10
        elif r == 2:
            return 15
        else:
            return 21


def rotl(x: U32, c: Int) -> U32:
    return (x << U32(c)) | (x >> U32(32 - c))


def md5_block(chunk: SU8Ptr, m: W32Ptr, state: W32Ptr):
    """Schedule one 64-byte block into `m[0:64]` and run the four rounds.

    `state` holds `A B C D` on entry and the same four words advanced by the
    block on exit. It cannot live in `m`: the message schedule overwrites
    `m[16:20]` on its own second half.
    """
    for i in range(16):
        var word = U32(0)
        for b in range(4):
            word |= U32(chunk.unsafe_load(i * 4 + b)) << U32(8 * b)
        m.unsafe_offset(i)[] = word
    for i in range(16, 64):
        m.unsafe_offset(i)[] = (
            m.unsafe_load(i - 16)
            + (
                rotl(m.unsafe_load(i - 15), 7)
                ^ rotl(m.unsafe_load(i - 7), 12)
                ^ m.unsafe_load(i - 3)
            )
            + rotl(m.unsafe_load(i - 2), 17)
        )

    var a0 = state.unsafe_load(0)
    var b0 = state.unsafe_load(1)
    var c0 = state.unsafe_load(2)
    var d0 = state.unsafe_load(3)
    var a = a0
    var b = b0
    var c = c0
    var d = d0

    for i in range(64):
        var f: U32
        var g: Int
        if i < 16:
            f = (b & c) | (~b & d)
            g = i
        elif i < 32:
            f = (d & b) | (~d & c)
            g = (5 * i + 1) % 16
        elif i < 48:
            f = b ^ c ^ d
            g = (3 * i + 5) % 16
        else:
            f = c ^ (b | ~d)
            g = (7 * i) % 16
        f = f + a + m.unsafe_load(g) + md5_k(i)
        a = d
        d = c
        c = b
        b = b + rotl(f, md5_s(i))

    state.unsafe_offset(0)[] = a0 + a
    state.unsafe_offset(1)[] = b0 + b
    state.unsafe_offset(2)[] = c0 + c
    state.unsafe_offset(3)[] = d0 + d


def md5(
    data: SU8Ptr,
    n: Int,
    digest: W32Ptr,
    m: W32Ptr,
    state: W32Ptr,
    tail: SU8Ptr,
):
    """MD5 of the `n` bytes at `data`, as four little-endian words in `digest`.

    Scratch is the caller's, as everywhere in this library: `m` is the
    64-word message schedule, `state` is the RFC's `A0 B0 C0 D0` (the only
    thing carried between blocks) and `tail` is the 128-byte staging area for
    the padded final block. Sizes are `MD5_SCRATCH_WORDS` and
    `MD5_SCRATCH_BYTES`.
    """
    state.unsafe_offset(0)[] = U32(0x67452301)
    state.unsafe_offset(1)[] = U32(0xefcdab89)
    state.unsafe_offset(2)[] = U32(0x98badcfe)
    state.unsafe_offset(3)[] = U32(0x10325476)

    var full = n // 64
    for block in range(full):
        md5_block(SU8Ptr(unsafe_from_address=Int(data) + block * 64), m, state)

    var rest = n - full * 64
    for i in range(rest):
        tail.unsafe_offset(i)[] = data.unsafe_load(full * 64 + i)
    tail.unsafe_offset(rest)[] = UInt8(0x80)
    # One block holds 55 payload bytes plus the 0x80 and the 8-byte length.
    var padded = 64
    if rest >= 56:
        padded = 128
    for i in range(rest + 1, padded - 8):
        tail.unsafe_offset(i)[] = UInt8(0)
    for i in range(8):
        tail.unsafe_offset(padded - 8 + i)[] = UInt8((n * 8) >> (8 * i))
    md5_block(tail, m, state)
    if padded == 128:
        md5_block(SU8Ptr(unsafe_from_address=Int(tail) + 64), m, state)

    for i in range(4):
        digest.unsafe_offset(i)[] = state.unsafe_load(i)
