"""Use small, repeatable datasets and keep test files in temporary folders."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsdst-test-matplotlib")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    np.random.seed(42)
    yield
    plt.close("all")


@pytest.fixture
def classification():
    rng = np.random.default_rng(12)
    x = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    p = 1 / (1 + np.exp(-x.a + .5 * x.b))
    return x, pd.Series(rng.binomial(1, p), name="y")


@pytest.fixture
def regression():
    rng = np.random.default_rng(20)
    x = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series(2 * x.a - .5 * x.b + rng.normal(scale=.2, size=len(x)), name="y")
    return x, y
