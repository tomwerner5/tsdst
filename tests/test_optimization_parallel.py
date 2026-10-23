"""Test solvers on known objectives and check parallel task results."""

from concurrent.futures import ThreadPoolExecutor
from operator import add

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsdst import optimization as o, parallel as p


def test_sign_and_root_intervals():
    assert [o.sign(value) for value in [-2, 0, 2]] == [-1, 1, 1]
    bounds = o.signIntervals(lambda x: (x + 1) * x * (x - 1), lower=-2, upper=2, step=.1)
    assert len(bounds) == 3
    for interval, root in zip(bounds, [-1, 0, 1]):
        assert interval[0] <= root <= interval[1] + 1e-12


def test_brent_root_and_minimizers():
    result = o.Brent_1DRoot(lambda x: x * x - 2, opt={"bounds": [0, 2]}, print_=False)
    assert result["root"] == pytest.approx(np.sqrt(2), abs=1e-6)
    scalar = o.AdaptiveNMorBrent(lambda x: (x - 2) ** 2, opt={"bounds": [-5, 5]}, print_=False)
    assert scalar["Minimum"] == pytest.approx(2, abs=1e-5)
    vector = o.AdaptiveNMorBrent(lambda x: np.sum((x - [2, 3]) ** 2), start=[0, 0], print_=False)
    assert_allclose(vector["Parameters"], [2, 3], atol=1e-4)
    params, se, interval = o.max_like([0., 0.], lambda x: np.sum((x - [1, 2]) ** 2), print_=False)
    assert_allclose(params, [1, 2], atol=1e-4)
    assert_allclose(se, [1 / np.sqrt(2)] * 2, atol=1e-4)
    assert interval.shape == (2, 2)


@pytest.mark.parametrize("jobs", [1, 2])
def test_parallel_results_preserve_input_order(jobs, monkeypatch):
    args = {"a": 10}
    tasks = [{"b": 1}, {"b": 2}, {"b": 3}]
    assert p.p_prog_simp(args, tasks, add_keywords, n_jobs=jobs, verbose=False) == [11, 12, 13]
    assert args == {"a": 10}
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    assert sorted(p.p_prog_simp_memorySafe(args, tasks, add_keywords, n_jobs=jobs)) == [11, 12, 13]
    assert args == {"a": 10}


def add_keywords(a, b):
    return add(a, b)
