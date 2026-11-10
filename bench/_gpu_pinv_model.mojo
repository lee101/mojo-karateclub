"""End-to-end model of the one GPU candidate, `pinv_spectrum`, in Mojo.

This is not part of the library. It is the prototype that decided the GPU gate:
a single `max.gpu` kernel launch mirroring `kclinalg.pinv_spectrum` (one thread
per upper-triangle cell, one `dot` per cell, the tail handled by a `t < total`
guard), timed with the copies a real GPU path would need - the two inputs in
once, the result out once, one launch, one sync per call.

It reports per-call time and checks the device result against the same
reconstruction computed on the host, so a timing cannot come from a kernel that
did the wrong thing.

    # MODULAR_HOME must point at share/max (MOJO_NOTES.md 0); pixi run sets it.
    pixi run mojo run bench/_gpu_pinv_model.mojo

Measured on the RTX 5090 (PCIe enumerated as gen 3 x16, ~12 GB/s pinned),
2026-11-10, a shared 72-core host at load average 34-41. The per-call figures
are from `bench/_gpu_compare.py`, which interleaves this with the CPU loop so a
load spike lands on both; this script alone reads 0.02-0.04 ms slower.

    d     per call (device)   CPU serial
    34      0.022 ms          0.029 ms
    60      0.034 ms          0.066 ms
    200     0.152 ms          1.553 ms
    300     0.314 ms          5.496 ms

    device result bit-identical to the host reconstruction (max |diff| = 0.0)
"""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime THREADS = 256


def reconstruct_kernel(
    vectors: Ptr,
    inv: Ptr,
    dst: Ptr,
    d: Int32,
    threads: Int32,
):
    """`dst[r, c] = dst[c, r] = sum_k vectors[r, k] * inv[k] * vectors[c, k]`.

    One thread per cell of the upper triangle; the triangular index `t` maps to
    `(r, c)` by walking the row lengths, so the grid is `t < total` and the last
    block's spare threads are the explicit tail.
    """
    var t = Int(block_idx.x) * Int(threads) + Int(thread_idx.x)
    var di = Int(d)
    var total = di * (di + 1) // 2
    if t >= total:
        return
    var r = 0
    var row_start = 0
    while row_start + (di - r) <= t:
        row_start += di - r
        r += 1
    var c = r + (t - row_start)
    var acc = 0.0
    for k in range(di):
        acc += (
            vectors.unsafe_load(r * di + k)
            * inv.unsafe_load(k)
            * vectors.unsafe_load(c * di + k)
        )
    dst.unsafe_store(r * di + c, acc)
    dst.unsafe_store(c * di + r, acc)


def _orthonormal_basis(d: Int) -> List[Float64]:
    """A deterministic orthonormal `d x d` basis (LCG + Gram-Schmidt), so the
    input is a real rotated identity rather than a synthetic all-ones block."""
    var nn = d * d
    var basis = List[Float64](length=nn, fill=0.0)
    var rng: UInt64 = 20261110
    for r in range(d):
        for c in range(d):
            rng = rng * 6364136223846793005 + 1442695040888963407
            basis[r * d + c] = Float64(Int(rng >> 33) % 2001 - 1000) / 1000.0
    for r in range(d):
        for prev in range(r):
            var held = 0.0
            for k in range(d):
                held += basis[r * d + k] * basis[prev * d + k]
            for k in range(d):
                basis[r * d + k] -= held * basis[prev * d + k]
        var norm = 0.0
        for k in range(d):
            norm += basis[r * d + k] * basis[r * d + k]
        norm = sqrt(norm)
        for k in range(d):
            basis[r * d + k] /= norm
    return basis^


def _matmul_nt(basis: List[Float64], d: Int) -> List[Float64]:
    var nn = d * d
    var out = List[Float64](length=nn, fill=0.0)
    for r in range(d):
        for c in range(d):
            var acc = 0.0
            for k in range(d):
                acc += basis[r * d + k] * basis[c * d + k]
            out[r * d + c] = acc
    return out^


def run_d(ctx: DeviceContext, d: Int) raises -> Float64:
    var nn = d * d
    var basis = _orthonormal_basis(d)
    var h_vec = _matmul_nt(basis, d)
    var h_inv = List[Float64](length=d, fill=1.0)
    var h_out = List[Float64](length=nn, fill=0.0)
    var dev_vec = ctx.enqueue_create_buffer[DType.float64](nn)
    var dev_inv = ctx.enqueue_create_buffer[DType.float64](d)
    var dev_out = ctx.enqueue_create_buffer[DType.float64](nn)
    var blocks = (d * (d + 1) // 2 + THREADS - 1) // THREADS
    var best = 1e30
    for _ in range(7):
        var t0 = perf_counter_ns()
        for _ in range(200):
            ctx.enqueue_copy(dev_vec, h_vec)
            ctx.enqueue_copy(dev_inv, h_inv)
            ctx.enqueue_function[reconstruct_kernel](
                dev_vec, dev_inv, dev_out, Int32(d), Int32(THREADS),
                grid_dim=blocks, block_dim=THREADS,
            )
            ctx.enqueue_copy(h_out, dev_out)
            ctx.synchronize()
        best = min(best, Float64(perf_counter_ns() - t0) / 1e6 / 200.0)
    # Correctness: the device result against the host reconstruction.
    var worst = 0.0
    for r in range(d):
        for c in range(d):
            var acc = 0.0
            for k in range(d):
                acc += h_vec[r * d + k] * h_inv[k] * h_vec[c * d + k]
            worst = max(worst, abs(acc - h_out[r * d + c]))
    print("  max |gpu - host| =", worst)
    return best


def main() raises:
    var ctx = DeviceContext()
    print(ctx.name())
    for d in [34, 60, 200, 300]:
        var ms = run_d(ctx, d)
        print("d=", d, " per call ms=", ms)
