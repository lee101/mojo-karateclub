import importlib.util, pathlib, sys, time
import numpy as np
ROOT = pathlib.Path("/nvme0n1-disk/code/mojo-karateclub")
sys.path.insert(0, str(ROOT/"python"))
from mojokarateclub import _ffi
from mojokarateclub.estimator import csr_from_graph
spec = importlib.util.spec_from_file_location("kc_fixtures", ROOT/"tests/_graphs.py")
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
lib=_ffi._lib
def adjacency(name):
    g,n = fx.build_graph(name, integrity=True)
    ip,ix,v = csr_from_graph(g)
    out=np.zeros(n*n)
    lib.kc_normalized_adjacency(_ffi._index(ip,"i"),_ffi._index(ix,"i"),_ffi._data(v),n,_ffi._data(out))
    return out.reshape(n,n), n
for name in ("karate","ws300"):
    a,d = adjacency(name)
    best=None
    for _ in range(3):
        work=a.copy(); vecs=np.zeros((d,d)); vt=np.zeros((d,d))
        t=time.perf_counter()
        s=lib.kc_jacobi_eigh(_ffi._data(work),_ffi._data(vecs),_ffi._data(vt),d,400,1e-24)
        e=time.perf_counter()-t
        best=e if best is None else min(best,e)
    print(f"{name} d={d} sweeps={s} best={best*1e3:.3f}ms")
