import importlib.util, pathlib, sys, time, cProfile, pstats
import numpy as np
ROOT = pathlib.Path("/nvme0n1-disk/code/mojo-karateclub")
sys.path.insert(0, str(ROOT/"python"))
spec = importlib.util.spec_from_file_location("kc_fixtures", ROOT/"tests/_graphs.py")
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
import mojokarateclub as mk
from mojokarateclub import _ffi
from mojokarateclub.estimator import csr_from_graph

g = fx.build_graph("ws300", integrity=True)[0]
def run():
    m = mk.LaplacianEigenmaps(dimensions=8); m.fit(g); return m.get_embedding()
run()
t=time.perf_counter()
for _ in range(3): run()
print("fit total", (time.perf_counter()-t)/3)
pr=cProfile.Profile(); pr.enable()
for _ in range(3): run()
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(18)
