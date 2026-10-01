"""Shared ctypes layer: the library handle, its signatures and buffer rules.

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
    default = _ROOT / "dist" / "libmojo-karateclub.so"
    return Path(override) if override else default


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
    "kc_jacobi_eigh": (_I64, [_I64] * 5 + [_F64]),
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
    "kc_md5_digest": (None, [_I64] * 6),
    "kc_ldp_log_degrees": (None, [_I64] * 4),
    "kc_ldp_features": (None, [_I64] * 5),
    "kc_ldp_embedding": (None, [_I64] * 5),
    "kc_fgsd_similarity": (
        ctypes.c_bool,
        [_I64] * 10 + [_F64, _I64, _F64],
    ),
    "kc_fgsd_embedding": (
        ctypes.c_bool,
        [_I64] * 10 + [_F64, _I64, _F64, _I64, _F64, _F64, _I64],
    ),
    "kc_sf_calculate": (ctypes.c_bool, [_I64] * 12 + [_I64, _F64]),
    "kc_netlsd_calculate": (
        ctypes.c_bool,
        [_I64] * 4
        + [_F64, _F64]
        + [_I64] * 14
        + [_I64, _F64],
    ),
    "kc_laplacian_eigenmaps_fit": (
        ctypes.c_bool,
        [_I64] * 13 + [_I64, _F64],
    ),
    "kc_nodesketch_fit": (_I64, [_I64] * 3 + [_F64] + [_I64] * 18),
    "kc_nodesketch_get_embedding": (None, [_I64] * 4),
    "kc_line_fit_first_order": (
        None,
        [_I64] * 4 + [_F64, _F64] + [_I64] * 13,
    ),
    "kc_line_fit_second_order": (
        None,
        [_I64] * 4 + [_F64, _F64] + [_I64] * 15,
    ),
    "kc_mt_init_genrand": (None, [_I64, _I64]),
    "kc_mt_genrand_uint32": (_I64, [_I64]),
    "kc_mt_genrand_res53": (_F64, [_I64]),
    "kc_random_walker_do_walk": (_I64, [_I64] * 6),
    "kc_random_walker_do_walks": (_I64, [_I64] * 7),
    "kc_biased_random_walker_do_walk": (
        _I64,
        [_I64] * 6 + [_F64, _F64] + [_I64] * 4,
    ),
    "kc_biased_random_walker_do_walks": (
        _I64,
        [_I64] * 7 + [_F64, _F64] + [_I64] * 4,
    ),
    "kc_eulerian_diffuser_run_diffusion_process": (_I64, [_I64] * 16),
    "kc_eulerian_diffuser_do_diffusions": (_I64, [_I64] * 17),
    "kc_wl_hash": (None, [_I64] * 14),
    "kc_normalized_adjacency": (None, [_I64] * 5),
    "kc_symmetric_pinv": (ctypes.c_bool, [_I64] * 5 + [_F64, _I64, _F64]),
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

def _bytes(array, name="array"):
    return _address(array, np.dtype(np.uint8), name)


def _words(array, name="array"):
    return _address(array, np.dtype(np.uint32), name)


def _csr(indptr, indices, values, n=None):
    """Validate a CSR triple and return `(indptr, indices, values, n)`.

    The estimators take a graph as the triple `Estimator._ensure_integrity`
    would leave it in, so this is the one place the invariants are checked.
    """
    rows = _csr_rows(indptr, indices, values)
    if n is not None and n != rows:
        raise ValueError(f"graph has {rows} rows but {n} nodes were declared")
    if values is None:
        values = np.ones(indices.size, dtype=np.float64)
    _index_bound(indices, rows, "indices")
    return indptr, indices, np.ascontiguousarray(values, dtype=np.float64), rows


def _mt_state(words=625):
    return np.zeros(words, dtype=np.uint32)

