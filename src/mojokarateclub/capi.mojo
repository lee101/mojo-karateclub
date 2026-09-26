"""C ABI surface over `kclinalg`, loaded by ctypes from `python/mojokarateclub`.

Every buffer crosses the boundary as an `Int` address and is rebuilt inside the
wrapper: Mojo pointers are non-nullable, so they cannot be parameters of an
`abi("C")` function. Symbols are `kc_`-prefixed because the library is loaded
with `RTLD_LOCAL` into whatever process the tests happen to run in.
"""

import kclinalg as linalg

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ------------------------------------------------------------------ dense

@export("kc_fill")
def kc_fill(x: Int, value: Float64, n: Int) abi("C") -> None:
    linalg.fill(fptr(x), value, n)


@export("kc_copy")
def kc_copy(src: Int, dst: Int, n: Int) abi("C") -> None:
    linalg.copy(fptr(src), fptr(dst), n)


@export("kc_axpy")
def kc_axpy(alpha: Float64, x: Int, y: Int, n: Int) abi("C") -> None:
    linalg.axpy(alpha, fptr(x), fptr(y), n)


@export("kc_scale")
def kc_scale(x: Int, alpha: Float64, n: Int) abi("C") -> None:
    linalg.scale(fptr(x), alpha, n)


@export("kc_dot")
def kc_dot(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return linalg.dot(fptr(a), fptr(b), n)


@export("kc_sum")
def kc_sum(a: Int, n: Int) abi("C") -> Float64:
    return linalg.sumf(fptr(a), n)


@export("kc_sum_squares")
def kc_sum_squares(a: Int, n: Int) abi("C") -> Float64:
    return linalg.sum_squares(fptr(a), n)


@export("kc_frobenius")
def kc_frobenius(a: Int, n: Int) abi("C") -> Float64:
    return linalg.frobenius(fptr(a), n)


@export("kc_gemm")
def kc_gemm(a: Int, b: Int, dst: Int, m: Int, k: Int, n: Int) abi("C") -> None:
    linalg.gemm(fptr(a), fptr(b), fptr(dst), m, k, n)


@export("kc_gemm_nt")
def kc_gemm_nt(a: Int, b: Int, dst: Int, m: Int, k: Int, n: Int) abi("C") -> None:
    linalg.gemm_nt(fptr(a), fptr(b), fptr(dst), m, k, n)


@export("kc_matvec")
def kc_matvec(a: Int, x: Int, dst: Int, rows: Int, cols: Int) abi("C") -> None:
    linalg.matvec(fptr(a), fptr(x), fptr(dst), rows, cols)


@export("kc_jacobi_eigh")
def kc_jacobi_eigh(
    a: Int, vectors: Int, d: Int, sweeps: Int, tol: Float64
) abi("C") -> Int:
    return linalg.jacobi_eigh(fptr(a), fptr(vectors), d, sweeps, tol)


@export("kc_dense_to_coo")
def kc_dense_to_coo(
    a: Int, n: Int, rows: Int, cols: Int, values: Int
) abi("C") -> Int:
    return linalg.dense_to_coo(fptr(a), n, iptr(rows), iptr(cols), fptr(values))


@export("kc_invert")
def kc_invert(src: Int, work: Int, dst: Int, d: Int) abi("C") -> Bool:
    return linalg.invert(fptr(src), fptr(work), fptr(dst), d)

@export("kc_linspace")
def kc_linspace(dst: Int, lo: Float64, hi: Float64, n: Int) abi("C") -> None:
    linalg.linspace(fptr(dst), lo, hi, n)


@export("kc_l1_normalize_rows")
def kc_l1_normalize_rows(x: Int, n: Int, cols: Int) abi("C") -> None:
    linalg.l1_normalize_rows(fptr(x), n, cols)


@export("kc_histogram")
def kc_histogram(
    x: Int, dst: Int, n: Int, bins: Int, lo: Float64, hi: Float64
) abi("C") -> None:
    linalg.histogram(fptr(x), fptr(dst), n, bins, lo, hi)


# ------------------------------------------------------------------ graphs

@export("kc_clustering")
def kc_clustering(
    indptr: Int, indices: Int, dst: Int, n: Int, marker: Int
) abi("C") -> None:
    linalg.clustering(iptr(indptr), iptr(indices), fptr(dst), n, fptr(marker))


@export("kc_eccentricity")
def kc_eccentricity(
    indptr: Int, indices: Int, dst: Int, n: Int, queue: Int, seen: Int
) abi("C") -> None:
    linalg.eccentricity(
        iptr(indptr), iptr(indices), fptr(dst), n, iptr(queue), fptr(seen)
    )


@export("kc_csr_row_scale")
def kc_csr_row_scale(
    values: Int, scale: Int, indptr: Int, n: Int
) abi("C") -> None:
    linalg.csr_row_scale(fptr(values), fptr(scale), iptr(indptr), n)


@export("kc_csr_matvec")
def kc_csr_matvec(
    indptr: Int, indices: Int, values: Int, x: Int, dst: Int, n: Int, cols: Int
) abi("C") -> None:
    linalg.csr_matvec(
        iptr(indptr), iptr(indices), fptr(values), fptr(x), fptr(dst), n, cols
    )


@export("kc_csr_matmul")
def kc_csr_matmul(
    indptr: Int,
    indices: Int,
    values: Int,
    x: Int,
    dst: Int,
    n: Int,
    cols: Int,
    width: Int,
) abi("C") -> None:
    linalg.csr_matmul(
        iptr(indptr),
        iptr(indices),
        fptr(values),
        fptr(x),
        fptr(dst),
        n,
        cols,
        width,
    )


@export("kc_csr_dense")
def kc_csr_dense(
    indptr: Int, indices: Int, values: Int, n: Int, dst: Int
) abi("C") -> None:
    linalg.csr_dense(iptr(indptr), iptr(indices), fptr(values), n, fptr(dst))


@export("kc_csr_scatter")
def kc_csr_scatter(
    indptr: Int, indices: Int, values: Int, n: Int, cols: Int, dst: Int
) abi("C") -> None:
    linalg.csr_scatter(
        iptr(indptr), iptr(indices), fptr(values), n, cols, fptr(dst)
    )


@export("kc_coo_matmul")
def kc_coo_matmul(
    rows: Int,
    cols: Int,
    values: Int,
    nnz: Int,
    x: Int,
    dst: Int,
    n: Int,
    width: Int,
) abi("C") -> None:
    linalg.coo_matmul(
        iptr(rows), iptr(cols), fptr(values), nnz, fptr(x), fptr(dst), n, width
    )


@export("kc_coo_matmul_t")
def kc_coo_matmul_t(
    rows: Int,
    cols: Int,
    values: Int,
    nnz: Int,
    x: Int,
    dst: Int,
    n: Int,
    width: Int,
) abi("C") -> None:
    linalg.coo_matmul_t(
        iptr(rows), iptr(cols), fptr(values), nnz, fptr(x), fptr(dst), n, width
    )


@export("kc_coo_scatter")
def kc_coo_scatter(
    rows: Int, cols: Int, values: Int, nnz: Int, n: Int, dst: Int
) abi("C") -> None:
    linalg.coo_scatter(iptr(rows), iptr(cols), fptr(values), nnz, n, fptr(dst))
