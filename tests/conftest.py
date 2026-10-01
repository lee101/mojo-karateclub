"""Shared fixtures.

`upstream_vectors` is the dump produced by `pixi run vectors` in the upstream
environment; a test that asks for it is asserting against real karateclub
1.3.3 output, not against a reference this repo wrote. If the file is missing
the test skips with the command to regenerate it rather than silently passing.
"""

import pathlib

import numpy as np
import pytest

VECTORS = pathlib.Path(__file__).resolve().parent / "vectors" / "upstream.npz"


@pytest.fixture(scope="session")
def upstream_vectors():
    if not VECTORS.exists():
        pytest.skip(f"{VECTORS} missing; run `pixi run vectors`")
    with np.load(VECTORS, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
