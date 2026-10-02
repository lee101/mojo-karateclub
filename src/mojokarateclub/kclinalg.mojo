"""Dense float64, CSR and COO primitives over raw row-major buffers.

Not upstream code: karateclub leans on scipy/numpy for these. They live here so
the kernels can be read next to the upstream call each one replaces, without a
scipy round trip breaking the flow.

Nothing here allocates. Every routine writes into a caller-owned buffer, so
lifetimes stay the caller's problem.
"""

from std.math import sqrt

comptime W = 4
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

comptime TRANSPOSE_TILE = 64


def at(a: IPtr, i: Int) -> Int:
    """Index read out of an Int32 buffer; Mojo will not widen implicitly."""
    return Int(a.unsafe_load(i))


def row_range(indptr: IPtr, r: Int) -> Tuple[Int, Int]:
    return (Int(indptr.unsafe_load(r)), Int(indptr.unsafe_load(r + 1)))


def fill(x: Ptr, value: Float64, n: Int):
    for i in range(n):
        x.unsafe_offset(i)[] = value


def copy(src: Ptr, dst: Ptr, n: Int):
    for i in range(n):
        dst.unsafe_offset(i)[] = src.unsafe_load(i)


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.unsafe_store(i, y.unsafe_load[width=W](i) + va * x.unsafe_load[width=W](i))
        i += W
    while i < n:
        y.unsafe_offset(i)[] += alpha * x.unsafe_load(i)
        i += 1


def scale(x: Ptr, alpha: Float64, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        x.unsafe_store(i, x.unsafe_load[width=W](i) * va)
        i += W
    while i < n:
        x.unsafe_offset(i)[] *= alpha
        i += 1


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    return total


def sumf(a: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a.unsafe_load(i)
        i += 1
    return total


def sum_squares(a: Ptr, n: Int) -> Float64:
    """`numpy.linalg.norm(x) ** 2` over a dense block."""
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var v = a.unsafe_load[width=W](i)
        acc += v * v
        i += W
    var total = acc.reduce_add()
    while i < n:
        var v = a.unsafe_load(i)
        total += v * v
        i += 1
    return total


def frobenius(a: Ptr, n: Int) -> Float64:
    return sqrt(sum_squares(a, n))


def gemm(a: Ptr, b: Ptr, dst: Ptr, m: Int, k: Int, n: Int):
    """`dst[m, n] = a[m, k] @ b[k, n]`, all row-major."""
    for i in range(m * n):
        dst.unsafe_offset(i)[] = 0.0
    for i in range(m):
        for t in range(k):
            var v = a.unsafe_load(i * k + t)
            if v != 0.0:
                axpy(v, b.unsafe_offset(t * n), dst.unsafe_offset(i * n), n)


def gemm_nt(a: Ptr, b: Ptr, dst: Ptr, m: Int, k: Int, n: Int):
    """`dst[m, n] = a[m, k] @ b[n, k].T`, both operands row-major.

    `axpy` cannot carry this one: the column `b[:, t]` is strided by `k` in a
    row-major `b`, so the inner loop walks it by hand.
    """
    for i in range(m * n):
        dst.unsafe_offset(i)[] = 0.0
    for i in range(m):
        for t in range(k):
            var v = a.unsafe_load(i * k + t)
            if v == 0.0:
                continue
            for j in range(n):
                dst.unsafe_offset(i * n + j)[] += v * b.unsafe_load(j * k + t)


def matvec(a: Ptr, x: Ptr, dst: Ptr, rows: Int, cols: Int):
    for r in range(rows):
        dst.unsafe_offset(r)[] = dot(a.unsafe_offset(r * cols), x, cols)



def jacobi_rows(
    row_p: Ptr,
    row_q: Ptr,
    row_vp: Ptr,
    row_vq: Ptr,
    lo: Int,
    hi: Int,
    c: Float64,
    s: Float64,
):
    """One Jacobi rotation over the `k` in `[lo, hi)` of the pair `(p, q)`,
    for the span where both matrix entries are row elements of the packed
    layout: `A[k, p]` is `row_p[k]` and `A[k, q]` is `row_q[k]`. All four
    streams are contiguous, so the whole span vectorises.
    """
    var vc = SIMD[DType.float64, W](c)
    var vs = SIMD[DType.float64, W](s)
    var i = lo
    while i + W <= hi:
        var ap = row_p.unsafe_load[width=W](i)
        var aq = row_q.unsafe_load[width=W](i)
        var bp = row_vp.unsafe_load[width=W](i)
        var bq = row_vq.unsafe_load[width=W](i)
        row_p.unsafe_store(i, vc * ap - vs * aq)
        row_q.unsafe_store(i, vs * ap + vc * aq)
        row_vp.unsafe_store(i, vc * bp - vs * bq)
        row_vq.unsafe_store(i, vs * bp + vc * bq)
        i += W
    while i < hi:
        var ap = row_p.unsafe_load(i)
        var aq = row_q.unsafe_load(i)
        var bp = row_vp.unsafe_load(i)
        var bq = row_vq.unsafe_load(i)
        row_p.unsafe_offset(i)[] = c * ap - s * aq
        row_q.unsafe_offset(i)[] = s * ap + c * aq
        row_vp.unsafe_offset(i)[] = c * bp - s * bq
        row_vq.unsafe_offset(i)[] = s * bp + c * bq
        i += 1


def jacobi_pairs(
    col_p: Ptr,
    stride_p: Int,
    col_q: Ptr,
    stride_q: Int,
    row_vp: Ptr,
    row_vq: Ptr,
    lo: Int,
    hi: Int,
    c: Float64,
    s: Float64,
):
    """One Jacobi rotation over the `k` in `[lo, hi)` of the pair `(p, q)`,
    for the spans where at least one matrix entry is a column element.

    `col_p[i * stride_p]` holds `A[i, p]` and `col_q[i * stride_q]` holds
    `A[i, q]`: the packed layout puts the first in row `p` when `i < p` and in
    column `p` when `i > p`, so a column span passes a base and a stride rather
    than a row. The eigenvector streams stay unit stride, but the matrix ones
    do not, so this one stays scalar.
    """

    var i = lo
    while i < hi:
        var ap = col_p.unsafe_load(i * stride_p)
        var aq = col_q.unsafe_load(i * stride_q)
        var bp = row_vp.unsafe_load(i)
        var bq = row_vq.unsafe_load(i)
        col_p.unsafe_offset(i * stride_p)[] = c * ap - s * aq
        col_q.unsafe_offset(i * stride_q)[] = s * ap + c * aq
        row_vp.unsafe_offset(i)[] = c * bp - s * bq
        row_vq.unsafe_offset(i)[] = s * bp + c * bq
        i += 1


def transpose(src: Ptr, dst: Ptr, d: Int):
    """`dst[i, j] = src[j, i]` for a square `d x d`, in `TRANSPOSE_TILE`
    squares so neither side leaves L1 mid-tile."""
    var jb = 0
    while jb < d:
        var jend = min(jb + TRANSPOSE_TILE, d)
        var ib = 0
        while ib < d:
            var iend = min(ib + TRANSPOSE_TILE, d)
            for j in range(jb, jend):
                var row = src.unsafe_offset(j * d)
                for i in range(ib, iend):
                    dst.unsafe_offset(i * d + j)[] = row.unsafe_load(i)
            ib += TRANSPOSE_TILE
        jb += TRANSPOSE_TILE


def jacobi_eigh(
    a: Ptr, vectors: Ptr, vt: Ptr, d: Int, sweeps: Int, tol: Float64
) -> Int:
    """Diagonalise a symmetric `a[d, d]` in place; eigenvectors land in the
    columns of `vectors`. Returns the number of sweeps actually used.

    `numpy.linalg.eigh` is LAPACK's divide and conquer, which fixes the sign of
    each eigenvector and its ordering. The caller only ever consumes
    basis-invariant combinations here (`U @ diag(f) @ U.T`, and the spectrum
    itself), so the sweep order is free; anything that needs a specific sign
    has to pin it, as `mojo-sklearn`'s PCA does.

    The sweep runs on the packed upper triangle of `a`: `A[i, j]` lives at
    `a[min(i, j), max(i, j)]`, so the mirror `A[p, k] = A[k, p]` never has to
    be written during the sweep - and, more to the point, the span of `k` above
    `q` becomes two contiguous rows and vectorises, where the unfused sweep
    read that same span as two strided columns. The lower triangle is filled in
    once the sweep is done, so `a` is the symmetric matrix again on return.
    """
    for j in range(d):
        var row = vt.unsafe_offset(j * d)
        fill(row, 0.0, j)
        row.unsafe_offset(j)[] = 1.0
        fill(row.unsafe_offset(j + 1), 0.0, d - j - 1)
    var used = sweeps
    for sweep in range(sweeps):
        var off = 0.0
        for i in range(d):
            var row = a.unsafe_offset(i * d)
            for j in range(i + 1, d):
                var v = row.unsafe_load(j)
                off += v * v
        if off <= tol:
            used = sweep
            break
        for p in range(d):
            var row_p = a.unsafe_offset(p * d)
            var row_vp = vt.unsafe_offset(p * d)
            for q in range(p + 1, d):
                var apq = row_p.unsafe_load(q)
                if apq == 0.0:
                    continue
                var row_q = a.unsafe_offset(q * d)
                var row_vq = vt.unsafe_offset(q * d)
                var app = row_p.unsafe_load(p)
                var aqq = row_q.unsafe_load(q)
                var theta = (aqq - app) / (2.0 * apq)
                var sign = 1.0 if theta >= 0.0 else -1.0
                var t = sign / (abs(theta) + sqrt(theta * theta + 1.0))
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                jacobi_pairs(
                    a.unsafe_offset(p), d, a.unsafe_offset(q), d, row_vp, row_vq, 0, p, c, s
                )
                jacobi_pairs(
                    row_p, 1, a.unsafe_offset(q), d, row_vp, row_vq, p + 1, q, c, s
                )
                jacobi_rows(row_p, row_q, row_vp, row_vq, q + 1, d, c, s)
                # The 2 x 2 corner the three spans skip. Only the upper cell is
                # kept; the sweep reads no lower one until the mirror pass.
                var t1 = c * app - s * apq
                var t2 = s * app + c * apq
                var t3 = c * apq - s * aqq
                var t4 = s * apq + c * aqq
                row_p.unsafe_offset(p)[] = c * t1 - s * t3
                row_p.unsafe_offset(q)[] = c * t2 - s * t4
                row_q.unsafe_offset(q)[] = s * t2 + c * t4
                var m = row_vp.unsafe_load(p)
                var w = row_vq.unsafe_load(p)
                row_vp.unsafe_offset(p)[] = c * m - s * w
                row_vq.unsafe_offset(p)[] = s * m + c * w
                m = row_vp.unsafe_load(q)
                w = row_vq.unsafe_load(q)
                row_vp.unsafe_offset(q)[] = c * m - s * w
                row_vq.unsafe_offset(q)[] = s * m + c * w
    for i in range(d):
        var row = a.unsafe_offset(i * d)
        for j in range(i + 1, d):
            a.unsafe_offset(j * d + i)[] = row.unsafe_load(j)
    transpose(vt, vectors, d)
    return used



def pinv_spectrum(
    vectors: Ptr,
    values: Ptr,
    dst: Ptr,
    work: Ptr,
    inv: Ptr,
    d: Int,
    cut: Float64,
):
    """`U @ diag(1 / s) @ U.T` for a symmetric `U` with eigenvalues `s`.

    `values` is a `d x d` block whose diagonal holds the eigenvalues and
    whose rest is read never; `work` is a `d x d` scratch and `inv` is `d`
    doubles. The masked reciprocal is built once and applied to whole rows, so
    the `O(d^3)` reconstruction is a run of contiguous dot products rather
    than `d^2` passes each dividing by `s[k]` on every element. The result is
    symmetric, so only the upper triangle is accumulated and then mirrored.
    """
    for k in range(d):
        var s = values.unsafe_load(k * d + k)
        inv.unsafe_offset(k)[] = 0.0 if abs(s) <= cut else 1.0 / s
    for r in range(d):
        var src = vectors.unsafe_offset(r * d)
        var scaled = work.unsafe_offset(r * d)
        var k = 0
        while k + W <= d:
            scaled.unsafe_store(
                k, src.unsafe_load[width=W](k) * inv.unsafe_load[width=W](k)
            )
            k += W
        while k < d:
            scaled.unsafe_offset(k)[] = src.unsafe_load(k) * inv.unsafe_load(k)
            k += 1
    for r in range(d):
        var row = work.unsafe_offset(r * d)
        var out = dst.unsafe_offset(r * d)
        for c in range(r, d):
            var total = dot(row, vectors.unsafe_offset(c * d), d)
            out.unsafe_offset(c)[] = total
            dst.unsafe_offset(c * d + r)[] = total


def dense_to_coo(a: Ptr, n: Int, rows: IPtr, cols: IPtr, values: Ptr) -> Int:
    """Nonzeros of a dense `n x n` block, in row-major order. Returns the count."""
    var count = 0
    for i in range(n):
        for j in range(n):
            var v = a.unsafe_load(i * n + j)
            if v != 0.0:
                rows.unsafe_offset(count)[] = Int32(i)
                cols.unsafe_offset(count)[] = Int32(j)
                values.unsafe_offset(count)[] = v
                count += 1
    return count


def linspace(dst: Ptr, lo: Float64, hi: Float64, n: Int):
    """`numpy.linspace(lo, hi, n)`; the final element is the stop value exactly."""
    if n == 0:
        return
    if n == 1:
        dst.unsafe_offset(0)[] = lo
        return
    var step = (hi - lo) / Float64(n - 1)
    for i in range(n):
        dst.unsafe_offset(i)[] = lo + step * Float64(i)
    dst.unsafe_offset(n - 1)[] = hi

def invert(src: Ptr, work: Ptr, dst: Ptr, d: Int) -> Bool:
    """`numpy.linalg.inv` of a square `src[d, d]`, written into `dst[d, d]`.

    Gauss-Jordan with partial pivoting. `numpy.linalg.inv` is LAPACK LU with
    partial pivoting; for the small well-conditioned `d x d` systems
    karateclub inverts this agrees with it to rounding. Returns False on an
    exactly singular pivot, leaving `dst` undefined.

    `work` is a caller-owned `d x 2d` row-major scratch holding the augmented
    system `[src | I]`; its content on entry is irrelevant. It cannot be elided:
    the elimination is Gauss-Jordan on the augmented matrix, so the matrix and
    the running inverse have to be live at the same time, and this file does
    not allocate.
    """
    if d == 0:
        return True
    var wide = 2 * d
    for r in range(d):
        for c in range(wide):
            work.unsafe_offset(r * wide + c)[] = (
                src.unsafe_load(r * d + c) if c < d else 0.0
            )
        work.unsafe_offset(r * wide + (d + r))[] = 1.0
    for col in range(d):
        var pivot = col
        var best = abs(work.unsafe_load(col * wide + col))
        for r in range(col + 1, d):
            var candidate = abs(work.unsafe_load(r * wide + col))
            if candidate > best:
                best = candidate
                pivot = r
        if best == 0.0:
            return False
        if pivot != col:
            for c in range(wide):
                var tmp = work.unsafe_load(col * wide + c)
                work.unsafe_offset(col * wide + c)[] = work.unsafe_load(pivot * wide + c)
                work.unsafe_offset(pivot * wide + c)[] = tmp
        var diagonal = work.unsafe_load(col * wide + col)
        for c in range(wide):
            work.unsafe_offset(col * wide + c)[] = (
                work.unsafe_load(col * wide + c) / diagonal
            )
        for r in range(d):
            if r == col:
                continue
            var factor = work.unsafe_load(r * wide + col)
            if factor == 0.0:
                continue
            for c in range(wide):
                work.unsafe_offset(r * wide + c)[] = (
                    work.unsafe_load(r * wide + c) - factor * work.unsafe_load(col * wide + c)
                )
    for r in range(d):
        for c in range(d):
            dst.unsafe_offset(r * d + c)[] = work.unsafe_load(r * wide + (d + c))
    return True


def l1_normalize_rows(x: Ptr, n: Int, cols: Int):
    """`sklearn.preprocessing.normalize(x, axis=1, norm="l1")`.

    A zero row is left alone, which is what sklearn does by mapping a zero norm
    to one before dividing.
    """
    for r in range(n):
        var total = 0.0
        for c in range(cols):
            total += abs(x.unsafe_load(r * cols + c))
        if total == 0.0:
            continue
        for c in range(cols):
            x.unsafe_offset(r * cols + c)[] = x.unsafe_load(r * cols + c) / total


def histogram(x: Ptr, dst: Ptr, n: Int, bins: Int, lo: Float64, hi: Float64):
    """`numpy.histogram(x, bins=bins, range=(lo, hi))[0]`.

    Bins are half open `[edge[i], edge[i+1])` except the last, which includes
    its right edge; values outside the range are dropped. A degenerate range
    (`lo == hi`) is widened by half a unit each way first, as numpy does,
    rather than left to divide by zero.
    """
    if bins < 1:
        # `numpy.histogram` raises here; the kernel has nowhere to report it, so
        # it writes nothing rather than clamping a bin index to -1.
        return
    for b in range(bins):
        dst.unsafe_offset(b)[] = 0.0
    var low = lo
    var high = hi
    if high == low:
        low -= 0.5
        high += 0.5
    var width = (high - low) / Float64(bins)
    for i in range(n):
        var v = x.unsafe_load(i)
        if v < low or v > high:
            continue
        var b = Int((v - low) / width)
        if b < 0:
            b = 0
        if b > bins - 1:
            b = bins - 1
        dst.unsafe_offset(b)[] += 1.0


def clustering(indptr: IPtr, indices: IPtr, dst: Ptr, n: Int, marker: Ptr):
    """`networkx.clustering(G, node)` for an unweighted undirected CSR.

    Self loops are dropped, matching `set(v_nbrs) - {v}`, and a node with no
    triangles scores zero rather than a 0/0.
    """
    for v in range(n):
        var bounds = row_range(indptr, v)
        var degree = 0
        for e in range(bounds[0], bounds[1]):
            var w = at(indices, e)
            marker.unsafe_offset(w)[] = 1.0
            if w != v:
                degree += 1
        marker.unsafe_offset(v)[] = 0.0
        var triangles = 0
        for e in range(bounds[0], bounds[1]):
            var w = at(indices, e)
            if w == v:
                continue
            var inner = row_range(indptr, w)
            for f in range(inner[0], inner[1]):
                var u = at(indices, f)
                if u != w and marker.unsafe_load(u) != 0.0:
                    triangles += 1
        for e in range(bounds[0], bounds[1]):
            marker.unsafe_offset(at(indices, e))[] = 0.0
        dst.unsafe_offset(v)[] = (
            0.0
            if triangles == 0
            else Float64(triangles) / Float64(degree * (degree - 1))
        )


def eccentricity(indptr: IPtr, indices: IPtr, dst: Ptr, n: Int, queue: IPtr, seen: Ptr):
    """`networkx.eccentricity(G, node)` for an undirected CSR.

    One breadth-first sweep per node; the eccentricity is the largest distance
    reached. A node that cannot reach the whole graph gets -1, which is how the
    caller sees that networkx would have raised instead.
    """
    for v in range(n):
        for i in range(n):
            seen.unsafe_offset(i)[] = 0.0
        var head = 0
        var tail = 0
        queue.unsafe_offset(tail)[] = Int32(v)
        tail += 1
        seen.unsafe_offset(v)[] = 1.0
        var level = 0
        var deepest = 0
        while head < tail:
            level += 1
            var level_end = tail
            while head < level_end:
                var node = at(queue, head)
                head += 1
                var bounds = row_range(indptr, node)
                for e in range(bounds[0], bounds[1]):
                    var w = at(indices, e)
                    if w != node and seen.unsafe_load(w) == 0.0:
                        seen.unsafe_offset(w)[] = 1.0
                        queue.unsafe_offset(tail)[] = Int32(w)
                        tail += 1
            if tail > level_end:
                deepest = level
        dst.unsafe_offset(v)[] = Float64(deepest) if tail == n else -1.0


# ------------------------------------------------------------------ CSR

def csr_row_scale(values: Ptr, scale: Ptr, indptr: IPtr, n: Int):
    """Turn an unweighted CSR into `diag(scale) @ A` in place on the values."""
    for r in range(n):
        var factor = scale.unsafe_load(r)
        var bounds = row_range(indptr, r)
        for e in range(bounds[0], bounds[1]):
            values.unsafe_offset(e)[] = values.unsafe_load(e) * factor


def csr_matvec(
    indptr: IPtr, indices: IPtr, values: Ptr, x: Ptr, dst: Ptr, n: Int, cols: Int
):
    """`dst = A @ x` for a row-major CSR `A[n, cols]`."""
    for r in range(n):
        var bounds = row_range(indptr, r)
        var acc = 0.0
        for e in range(bounds[0], bounds[1]):
            acc += values.unsafe_load(e) * x.unsafe_load(at(indices, e))
        dst.unsafe_offset(r)[] = acc


def csr_matmul(
    indptr: IPtr,
    indices: IPtr,
    values: Ptr,
    x: Ptr,
    dst: Ptr,
    n: Int,
    cols: Int,
    width: Int,
):
    """`dst[n, width] = A[n, cols] @ x[cols, width]` for a row-major CSR `A`."""
    for i in range(n * width):
        dst.unsafe_offset(i)[] = 0.0
    for r in range(n):
        var bounds = row_range(indptr, r)
        for e in range(bounds[0], bounds[1]):
            var alpha = values.unsafe_load(e)
            if alpha == 0.0:
                continue
            axpy(
                alpha,
                x.unsafe_offset(at(indices, e) * width),
                dst.unsafe_offset(r * width),
                width,
            )


def csr_dense(indptr: IPtr, indices: IPtr, values: Ptr, n: Int, dst: Ptr):
    """`dst[n, n] = A @ A` for a square row-major CSR `A`."""
    for i in range(n * n):
        dst.unsafe_offset(i)[] = 0.0
    for r in range(n):
        var row = dst.unsafe_offset(r * n)
        var bounds = row_range(indptr, r)
        for e1 in range(bounds[0], bounds[1]):
            var alpha = values.unsafe_load(e1)
            var k = at(indices, e1)
            var inner = row_range(indptr, k)
            for e2 in range(inner[0], inner[1]):
                var column = at(indices, e2)
                row.unsafe_offset(column)[] = (
                    row.unsafe_load(column) + alpha * values.unsafe_load(e2)
                )


def csr_scatter(indptr: IPtr, indices: IPtr, values: Ptr, n: Int, cols: Int, dst: Ptr):
    """`dst[n, cols] = A` for a row-major CSR `A`, including explicit zeros."""
    for i in range(n * cols):
        dst.unsafe_offset(i)[] = 0.0
    for r in range(n):
        var bounds = row_range(indptr, r)
        for e in range(bounds[0], bounds[1]):
            dst.unsafe_offset(r * cols + at(indices, e))[] = values.unsafe_load(e)


# ------------------------------------------------------------------ COO

def coo_matmul(
    rows: IPtr,
    cols: IPtr,
    values: Ptr,
    nnz: Int,
    x: Ptr,
    dst: Ptr,
    n: Int,
    width: Int,
):
    """`dst[n, width] = A @ x` for a COO `A`; duplicate entries add."""
    for i in range(n * width):
        dst.unsafe_offset(i)[] = 0.0
    for e in range(nnz):
        axpy(
            values.unsafe_load(e),
            x.unsafe_offset(at(cols, e) * width),
            dst.unsafe_offset(at(rows, e) * width),
            width,
        )


def coo_matmul_t(
    rows: IPtr,
    cols: IPtr,
    values: Ptr,
    nnz: Int,
    x: Ptr,
    dst: Ptr,
    n: Int,
    width: Int,
):
    """`dst[n, width] = A.T @ x` for a COO `A`."""
    for i in range(n * width):
        dst.unsafe_offset(i)[] = 0.0
    for e in range(nnz):
        axpy(
            values.unsafe_load(e),
            x.unsafe_offset(at(rows, e) * width),
            dst.unsafe_offset(at(cols, e) * width),
            width,
        )


def coo_scatter(rows: IPtr, cols: IPtr, values: Ptr, nnz: Int, n: Int, dst: Ptr):
    """`dst[n, n] = A` for a COO `A`; a repeated entry overwrites, as scipy does."""
    for i in range(n * n):
        dst.unsafe_offset(i)[] = 0.0
    for e in range(nnz):
        dst.unsafe_offset(at(rows, e) * n + at(cols, e))[] = values.unsafe_load(e)
