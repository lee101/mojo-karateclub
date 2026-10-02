import sys, numpy as np
sys.path.insert(0,"/nvme0n1-disk/code/mojo-karateclub/python")
from mojokarateclub import _ffi
from mojokarateclub.linalg import _data
rng = np.random.default_rng(7)
d=3
a=rng.normal(size=(d,d)); a=np.ascontiguousarray(a+a.T)
work=a.copy(); vectors=np.zeros((d,d)); vt=np.zeros((d,d))
used=_ffi._lib.kc_jacobi_eigh(_data(work),_data(vectors),_data(vt),d,100,1e-24)
print("used",used)
print("work\n",work)
print("eigvalsh",np.linalg.eigvalsh(a))
print("vectors\n",vectors)
print("residual", np.abs(a@vectors-vectors*np.diag(work)).max())
