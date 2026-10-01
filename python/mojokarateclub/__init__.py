"""mojo-karateclub: the compute-bound half of `karateclub` in Mojo.

Every routine takes numpy arrays and returns numpy arrays, so the caller never
sees a raw address. Buffers are handed to Mojo as 64-bit `Int` addresses and
rebuilt as pointers inside the C ABI shim; Mojo `Int` is 64-bit, so a 32-bit
ctypes argument silently truncates an address into a wild pointer.

The kernels write through the caller's memory and never allocate, so a wrong
buffer size is memory corruption rather than an exception. `_ffi._data` and
`_ffi._index` are therefore strict about dtype and contiguity, and routines
documented as in-place reject anything they would otherwise have to copy.

The estimators live in `graph_embedding` and `node_embedding` and mirror the
upstream class names, argument order and defaults. `README.md` lists what is
covered and what is not.
"""

from ._ffi import _F64, _I64, _SIGNATURES, library_path, load_library
from .estimator import Estimator
from .linalg import *  # noqa: F401,F403
from .linalg import __all__ as _linalg_all
from . import graph_embedding, node_embedding, utils
from .graph_embedding import (
    FGSD,
    LDP,
    NetLSD,
    SF,
    Graph2Vec,
)
from .node_embedding import (
    DeepWalk,
    Diff2Vec,
    FirstOrderLINE,
    LaplacianEigenmaps,
    Node2Vec,
    NodeSketch,
    SecondOrderLINE,
    Walklets,
)
from .utils import (
    BiasedRandomWalker,
    EulerianDiffuser,
    RandomWalker,
    WeisfeilerLehmanHashing,
)

__all__ = [
    *_linalg_all,
    "BiasedRandomWalker",
    "DeepWalk",
    "Diff2Vec",
    "EulerianDiffuser",
    "FGSD",
    "FirstOrderLINE",
    "Graph2Vec",
    "LDP",
    "LaplacianEigenmaps",
    "NetLSD",
    "Node2Vec",
    "NodeSketch",
    "RandomWalker",
    "SF",
    "SecondOrderLINE",
    "Walklets",
    "WeisfeilerLehmanHashing",
    "library_path",
    "load_library",
    "graph_embedding",
    "node_embedding",
    "utils",
]

__version__ = "0.1.0"
