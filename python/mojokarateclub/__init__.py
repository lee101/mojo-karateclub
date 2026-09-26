"""ctypes bindings for the Mojo kernels in `src/mojokarateclub`.

Every routine takes numpy arrays and returns numpy arrays, so the caller never
sees a raw address. Buffers are handed to Mojo as 64-bit `Int` addresses and
rebuilt as pointers inside the C ABI shim; Mojo `Int` is 64-bit, so a 32-bit
ctypes argument silently truncates an address into a wild pointer.

The kernels write through the caller's memory and never allocate, so a wrong
buffer size is memory corruption rather than an exception. `_data`/`_index` are
therefore strict about dtype and contiguity, and routines documented as in-place
reject anything they would otherwise have to copy.
"""

import ctypes
import os
from pathlib import Path

import numpy as np

__all__ = [
    "axpy",
    "clustering",
    "coo_matmul",
    "coo_matmul_t",
    "coo_scatter",
    "copy",
    "csr_dense",
    "csr_matmul",
    "csr_matvec",
    "csr_row_scale",
    "csr_scatter",
    "dense_to_coo",
    "dot",
    "eccentricity",
    "eigh",
    "fill",
    "frobenius",
    "gemm",
    "gemm_nt",
    "histogram",
    "invert",
    "l1_normalize_rows",
    "linspace",
    "matvec",
    "scale",
    "sum_squares",
    "total",
]

_F64 = ctypes.c_double
_I64 = ctypes.c_int64

_ROOT = Path(__file__).resolve().parents[2]


def library_path() -> Path:
    """Path of the shared library `pixi run build` produces."""
    override = os.environ.get("MOJOKARATECLUB_LIB")
    return Path(override) if override else _ROOT / "dist" / "libmojokarateclub.so"


def load_library() -> ctypes.CDLL:
    """Load the shared library, with an actionable error when it is missing."""
    path = library_path()
    if not path.exists():
        raise RuntimeError(
            f"{path} not found; run `pixi run build` to compile the Mojo sources"
        )
    return ctypes.CDLL(str(path))


_lib = load_library()

# Addresses are c_int64 because Mojo Int is 64-bit; leaving an argtype off lets
# ctypes pick a 32-bit conversion for Python ints and truncates the address.
_SIGNATURES: dict[str, tuple] = {
    "kc_fill": (None, [_I64, _F64, _I64]),
    "kc_copy": (None, [_I64, _I64, _I64]),
    "kc_axpy": (None, [_F64, _I64, _I64, _I64]),
    "kc_scale": (None, [_I64, _F64, _I64]),
    "kc_dot": (_F64, [_I64, _I64, _I64]),
    "kc_sum": (_F64, [_I64, _I64]),
    "kc_sum_squares": (_F64, [_I64, _I64]),
    "kc_frobenius": (_F64, [_I64, _I64]),
    "kc_gemm": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_gemm_nt": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_matvec": (None, [_I64, _I64, _I64, _I64, _I64]),
    "kc_jacobi_eigh": (_I64, [_I64, _I64, _I64, _I64, _F64]),
    "kc_invert": (ctypes.c_bool, [_I64, _I64, _I64, _I64]),
    "kc_dense_to_coo": (_I64, [_I64, _I64, _I64, _I64, _I64]),
    "kc_linspace": (None, [_I64, _F64, _F64, _I64]),
    "kc_l1_normalize_rows": (None, [_I64, _I64, _I64]),
    "kc_histogram": (None, [_I64, _I64, _I64, _I64, _F64, _F64]),
    "kc_clustering": (None, [_I64, _I64, _I64, _I64, _I64]),
    "kc_eccentricity": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_csr_row_scale": (None, [_I64, _I64, _I64, _I64]),
    "kc_csr_matvec": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_csr_matmul": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_csr_dense": (None, [_I64, _I64, _I64, _I64, _I64]),
    "kc_csr_scatter": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_coo_matmul": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_coo_matmul_t": (None, [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64]),
    "kc_coo_scatter": (None, [_I64, _I64, _I64, _I64, _I64, _I64]),
}

for _name, (_restype, _argtypes) in _SIGNATURES.items():
    _entry = getattr(_lib, _name)
    _entry.restype = _restype
    _entry.argtypes = _argtypes


def _address(array, dtype, name):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must be {np.dtype(dtype).name}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return array.ctypes.data


def _data(array, name="array"):
    return _address(array, np.dtype(np.float64), name)


def _index(array, name="array"):
    return _address(array, np.dtype(np.int32), name)


def _writable(array, name):
    """Address of a buffer the kernel will mutate in place."""
    if not array.flags.writeable:
        raise ValueError(f"{name} must be writeable")
    return _data(array, name)


def _index_bound(indices, limit, name="indices"):
    """Reject an index array that would send a kernel out of bounds.

    The kernels trust their index buffers the way a C loop does: `indices[e]`
    is used directly as an offset. A caller who hands over a CSR with a
    dangling column corrupts memory rather than raising, so the check lives
    here.
    """
    if not indices.size:
        return
    hi = int(indices.max())
    lo = int(indices.min())
    if lo < 0 or hi >= limit:
        raise ValueError(f"{name} must lie in [0, {limit}), got [{lo}, {hi}]")


def _csr_rows(indptr, indices, values=None):
    """Validate a CSR triple and return its row count."""
    if indptr.ndim != 1 or indptr.size < 1:
        raise ValueError("indptr must be a non-empty 1-D array")
    n = indptr.size - 1
    if int(indptr[0]) != 0:
        raise ValueError(f"indptr must start at 0, got {int(indptr[0])}")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be non-decreasing")
    end = int(indptr[-1])
    if end != indices.size:
        raise ValueError(f"indptr ends at {end} but indices holds {indices.size}")
    if values is not None and values.size != indices.size:
        raise ValueError(
            f"values holds {values.size} entries, indices holds {indices.size}"
        )
    return n


def _zeros(shape, dtype=np.float64):
    return np.zeros(shape, dtype=dtype, order="C")


# ------------------------------------------------------------------ dense


def fill(x, value):
    """Set every element of `x` to `value`, in place."""
    _lib.kc_fill(_writable(x, "x"), float(value), x.size)


def copy(src, dst):
    """`dst[:] = src`, in place on `dst`."""
    src_addr, dst_addr = _data(src, "src"), _writable(dst, "dst")
    if src.size != dst.size:
        raise ValueError(f"size mismatch: {src.size} != {dst.size}")
    _lib.kc_copy(src_addr, dst_addr, src.size)


def axpy(alpha, x, y):
    """`y += alpha * x`, in place on `y`."""
    x_addr, y_addr = _data(x, "x"), _writable(y, "y")
    if x.size != y.size:
        raise ValueError(f"size mismatch: {x.size} != {y.size}")
    _lib.kc_axpy(float(alpha), x_addr, y_addr, x.size)


def scale(x, alpha):
    """`x *= alpha`, in place on `x`."""
    _lib.kc_scale(_writable(x, "x"), float(alpha), x.size)


def dot(a, b):
    """`numpy.dot(a, b)`."""
    a_addr, b_addr = _data(a, "a"), _data(b, "b")
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} != {b.size}")
    return _lib.kc_dot(a_addr, b_addr, a.size)


def total(a):
    """`a.sum()`."""
    return _lib.kc_sum(_data(a), a.size)


def sum_squares(a):
    """`numpy.linalg.norm(a) ** 2`."""
    return _lib.kc_sum_squares(_data(a), a.size)


def frobenius(a):
    """`numpy.linalg.norm(a)`."""
    return _lib.kc_frobenius(_data(a), a.size)


def gemm(a, b):
    """`a @ b` for 2-D `a` and `b`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"gemm needs two 2-D arrays, got {a.ndim}-D and {b.ndim}-D"
        )
    m, k = a.shape
    inner, n = b.shape
    if k != inner:
        raise ValueError(f"inner dimensions disagree: {a.shape} @ {b.shape}")
    dst = _zeros((m, n))
    _lib.kc_gemm(_data(a), _data(b), _data(dst), m, k, n)
    return dst


def gemm_nt(a, b):
    """`a @ b.T` for 2-D `a` and `b`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"gemm_nt needs two 2-D arrays, got {a.ndim}-D and {b.ndim}-D"
        )
    m, k = a.shape
    n, inner = b.shape
    if k != inner:
        raise ValueError(f"inner dimensions disagree: {a.shape} @ {b.shape}.T")
    dst = _zeros((m, n))
    _lib.kc_gemm_nt(_data(a), _data(b), _data(dst), m, k, n)
    return dst


def matvec(a, x):
    """`a @ x` for a 2-D `a` and a 1-D `x`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 2 or x.ndim != 1:
        raise ValueError(
            f"matvec needs a 2-D and a 1-D array, got {a.ndim}-D and {x.ndim}-D"
        )
    rows, cols = a.shape
    if x.size != cols:
        raise ValueError(f"shape mismatch: {a.shape} @ {x.shape}")
    dst = _zeros(rows)
    _lib.kc_matvec(_data(a), _data(x), _data(dst), rows, cols)
    return dst


def eigh(a, sweeps=100, tol=1e-12):
    """`numpy.linalg.eigh` for a symmetric 2-D `a`.

    Cyclic Jacobi, so the sweep count and the eigenvector signs differ from
    LAPACK's divide and conquer. Eigenvalues come back ascending with the
    matching eigenvector columns in the same order, which is all a caller of a
    basis-invariant `U @ diag(f) @ U.T` needs. Raises if the sweeps run out
    before the off-diagonal norm reaches `tol`, because a silently
    unconverged decomposition is a wrong answer rather than a slow one.
    """
    work = np.array(a, dtype=np.float64, order="C", copy=True)
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError(f"eigh needs a square matrix, got {work.shape}")
    d = work.shape[0]
    vectors = _zeros((d, d))
    used = _lib.kc_jacobi_eigh(
        _data(work), _data(vectors), d, int(sweeps), float(tol)
    )
    if used >= sweeps:
        raise RuntimeError(f"Jacobi did not converge in {sweeps} sweeps for d={d}")
    values = np.diag(work).copy()
    order = np.argsort(values, kind="stable")
    return values[order], np.ascontiguousarray(vectors[:, order])


def dense_to_coo(a):
    """`(rows, cols, values)` nonzeros of a dense 2-D `a`, in row-major order."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        # The kernel walks an n x n block; a ragged block would read past
        # the end of `a` rather than raise.
        raise ValueError(f"dense_to_coo needs a square matrix, got {a.shape}")
    n = a.shape[0]
    rows = _zeros(n * n, np.int32)
    cols = _zeros(n * n, np.int32)
    values = _zeros(n * n)
    nnz = _lib.kc_dense_to_coo(_data(a), n, _index(rows), _index(cols), _data(values))
    return rows[:nnz], cols[:nnz], values[:nnz]


def linspace(lo, hi, n):
    """`numpy.linspace(lo, hi, n)`."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    dst = _zeros(n)
    _lib.kc_linspace(_data(dst), float(lo), float(hi), n)
    return dst


def invert(a):
    """`numpy.linalg.inv`, raising `LinAlgError` on an exactly singular pivot.

    The kernel runs Gauss-Jordan on the augmented system, so it needs a
    `d x 2d` scratch. That is allocated here because the kernel never does.
    """
    src = np.ascontiguousarray(a, dtype=np.float64)
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"invert needs a square matrix, got {a.shape}")
    d = src.shape[0]
    work = _zeros((d, 2 * d))
    dst = _zeros((d, d))
    if not _lib.kc_invert(_data(src), _data(work), _data(dst), d):
        raise np.linalg.LinAlgError("singular matrix")
    return dst


def l1_normalize_rows(x):
    """`sklearn.preprocessing.normalize(x, axis=1, norm="l1")`, in place."""
    if x.ndim != 2:
        raise ValueError(f"l1_normalize_rows needs a 2-D array, got {x.ndim}-D")
    _lib.kc_l1_normalize_rows(_writable(x, "x"), x.shape[0], x.shape[1])
    return x


def histogram(x, bins, lo, hi):
    """`numpy.histogram(x, bins=bins, range=(lo, hi))[0]`, as float64 counts."""
    if hi < lo:
        raise ValueError(f"range must satisfy lo <= hi, got ({lo}, {hi})")
    """`numpy.histogram(x, bins=bins, range=(lo, hi))[0]`, as float64 counts."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    dst = _zeros(bins)
    _lib.kc_histogram(_data(x), _data(dst), x.size, int(bins), float(lo), float(hi))
    return dst


# ------------------------------------------------------------------ graphs


def clustering(indptr, indices):
    """`networkx.clustering(G)` for an unweighted undirected CSR."""
    n = _csr_rows(indptr, indices)
    _index_bound(indices, n)
    dst = _zeros(n)
    marker = _zeros(n)
    _lib.kc_clustering(_index(indptr), _index(indices), _data(dst), n, _data(marker))
    return dst


def eccentricity(indptr, indices):
    """`networkx.eccentricity(G)`; a node that cannot reach every node is -1."""
    n = _csr_rows(indptr, indices)
    _index_bound(indices, n)
    dst = _zeros(n)
    queue = _zeros(n, np.int32)
    seen = _zeros(n)
    _lib.kc_eccentricity(
        _index(indptr), _index(indices), _data(dst), n, _index(queue), _data(seen)
    )
    return dst

def csr_row_scale(values, diag, indptr):
    """Turn an unweighted CSR into `diag(diag) @ A`, in place on `values`."""
    n = indptr.size - 1
    if n < 0 or int(indptr[-1]) != values.size:
        raise ValueError(
            f"indptr spans {int(indptr[-1])} entries but values holds {values.size}"
        )
    if diag.size != n:
        raise ValueError(f"diag holds {diag.size} entries, expected {n}")
    _lib.kc_csr_row_scale(_writable(values, "values"), _data(diag), _index(indptr), n)
    return values


def csr_matvec(indptr, indices, values, x):
    """`A @ x` for a CSR `A`."""
    n = _csr_rows(indptr, indices, values)
    _index_bound(indices, x.size)
    dst = _zeros(n)
    _lib.kc_csr_matvec(
        _index(indptr), _index(indices), _data(values), _data(x), _data(dst), n, x.size
    )
    return dst


def csr_matmul(indptr, indices, values, x):
    """`A @ x` for a CSR `A` and a 2-D `x`."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = _csr_rows(indptr, indices, values)
    _index_bound(indices, x.shape[0])
    dst = _zeros((n, x.shape[1]))
    _lib.kc_csr_matmul(
        _index(indptr),
        _index(indices),
        _data(values),
        _data(x),
        _data(dst),
        n,
        x.shape[0],
        x.shape[1],
    )
    return dst


def csr_dense(indptr, indices, values):
    """`A @ A` for a square CSR `A`, densified."""
    n = _csr_rows(indptr, indices, values)
    _index_bound(indices, n)
    dst = _zeros((n, n))
    _lib.kc_csr_dense(_index(indptr), _index(indices), _data(values), n, _data(dst))
    return dst


def csr_scatter(indptr, indices, values):
    """Densify a CSR `A` without multiplying it."""
    n = _csr_rows(indptr, indices, values)
    # `cols` is derived from the largest column index, so the destination is
    # always wide enough; the only index that could still escape it is a
    # negative one, which is what this rejects.
    _index_bound(indices, np.iinfo(np.int32).max)
    cols = int(indices.max()) + 1 if indices.size else 0
    dst = _zeros((n, cols))
    _lib.kc_csr_scatter(
        _index(indptr), _index(indices), _data(values), n, cols, _data(dst)
    )
    return dst


def _coo_extent(rows, cols, values):
    """Validate a COO triple and return its square extent."""
    if rows.ndim != 1 or cols.ndim != 1:
        raise ValueError("rows and cols must be 1-D arrays")
    if rows.size != cols.size or rows.size != values.size:
        raise ValueError(
            f"rows ({rows.size}), cols ({cols.size}) and values "
            f"({values.size}) must be the same length"
        )
    if not rows.size:
        return 0
    lo = min(int(rows.min()), int(cols.min()))
    hi = max(int(rows.max()), int(cols.max()))
    if lo < 0:
        raise ValueError(f"COO indices must be non-negative, got {lo}")
    return hi + 1


def coo_matmul(rows, cols, values, x):
    """`A @ x` for a COO `A` and a 2-D `x`; duplicate entries add."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = _coo_extent(rows, cols, values)
    _index_bound(cols, x.shape[0], "cols")
    dst = _zeros((n, x.shape[1]))
    _lib.kc_coo_matmul(
        _index(rows),
        _index(cols),
        _data(values),
        values.size,
        _data(x),
        _data(dst),
        n,
        x.shape[1],
    )
    return dst


def coo_matmul_t(rows, cols, values, x):
    """`A.T @ x` for a COO `A` and a 2-D `x`; duplicate entries add."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = _coo_extent(rows, cols, values)
    _index_bound(rows, x.shape[0], "rows")
    dst = _zeros((n, x.shape[1]))
    _lib.kc_coo_matmul_t(
        _index(rows),
        _index(cols),
        _data(values),
        values.size,
        _data(x),
        _data(dst),
        n,
        x.shape[1],
    )
    return dst


def coo_scatter(rows, cols, values):
    """Densify a COO `A`; a repeated entry overwrites, as scipy does."""
    n = _coo_extent(rows, cols, values)
    dst = _zeros((n, n))
    _lib.kc_coo_scatter(
        _index(rows), _index(cols), _data(values), values.size, n, _data(dst)
    )
    return dst
