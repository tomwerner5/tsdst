"""Check the retained API choices, information criteria, and selection contracts."""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from tsdst import estimators, feature_selection, parallel, tmath, utils


def test_retained_random_state_and_missing_value_choices():
    values, weights = ["a", "b", "c"], [0.5, 0.3, 0.2]
    expected = random.Random(93).choices(values, weights=weights, k=100)
    assert (
        utils.sample(values, 100, prob=weights, replace=True, random_state=93)
        == expected
    )
    state = random.getstate()
    utils.sample(values, 3, replace=True)
    assert random.getstate() == state
    assert (
        utils.inferFeatureType(pd.Series([1.0, 1.0, np.nan]), n_unique=1)
        == "categorical"
    )
    assert tmath.norm(np.array([[-1.0, 0.0], [2.0, -3.0]]), 0) == 3


@pytest.mark.parametrize("name", ["LogReg", "BayesLogRegClassifier"])
def test_classification_tie_uses_the_second_class(name):
    model = getattr(estimators, name)()
    model.coef_ = np.array([0.0])
    model.intercept_ = 0.0
    model.classes_ = np.array(["first", "second"])
    model.n_features_in_ = 1
    assert model.predict(np.zeros((1, 1))).tolist() == ["second"]
    assert_allclose(model.predict_proba(np.zeros((1, 1))), [[0.5, 0.5]])


def delayed_result(value, delay):
    time.sleep(delay)
    if isinstance(value, Exception):
        raise value
    return value


def test_memory_safe_results_keep_input_alignment_and_errors(monkeypatch):
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", ThreadPoolExecutor)
    arguments = {}
    tasks = [
        {"value": "first", "delay": 0.03},
        {"value": "second", "delay": 0.001},
        {"value": ValueError("third"), "delay": 0.001},
    ]
    before = deepcopy(tasks)
    result = parallel.p_prog_simp_memorySafe(arguments, tasks, delayed_result, n_jobs=2)
    assert result[:2] == ["first", "second"]
    assert isinstance(result[2], ValueError) and str(result[2]) == "third"
    assert arguments == {}
    assert tasks[:2] == before[:2]


@pytest.mark.parametrize("family", ["binomial", "poisson", "gaussian"])
def test_forward_criterion_matches_independent_likelihood_and_parameter_count(family):
    folder = Path(__file__).parent / "fixtures" / "audit_statistics"
    data = pd.read_csv(
        folder / ("ols-data.csv" if family == "gaussian" else "aic-data.csv")
    )
    x = data[[column for column in data if column.startswith("x")]]
    if family == "binomial":
        y = data.binary
        estimator = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000)
    elif family == "poisson":
        y = data.counts
        estimator = PoissonRegressor(alpha=0, tol=1e-12, max_iter=10000)
    else:
        y = data.y
        estimator = LinearRegression()
    fitted = deepcopy(estimator).fit(x, y)
    if family == "binomial":
        log_likelihood = stats.bernoulli.logpmf(y, fitted.predict_proba(x)[:, 1]).sum()
    elif family == "poisson":
        log_likelihood = stats.poisson.logpmf(y, fitted.predict(x)).sum()
    else:
        prediction = fitted.predict(x)
        sigma = np.sqrt(np.mean((y - prediction) ** 2))
        log_likelihood = stats.norm.logpdf(y, loc=prediction, scale=sigma).sum()
    parameters = x.shape[1] + 1 + int(family == "gaussian")
    expected = -2 * log_likelihood + parameters * np.log(len(y))
    actual = feature_selection._calcForwardAICs(x, y, estimator, "bic", family)
    assert_allclose(actual, expected, atol=1e-9, rtol=0)
    assert not hasattr(estimator, "coef_")


def test_forward_stopping_and_ties_are_explicit(monkeypatch):
    data = pd.DataFrame(
        {"a": range(12), "b": range(12), "c": range(12), "y": range(12)}
    )
    weights = {"a": 10, "b": 8, "c": 6}

    def criterion(x, *args):
        return 100 - sum(weights[column] for column in x)

    monkeypatch.setattr(feature_selection, "_calcForwardAICs", criterion)
    result = feature_selection.forwardSelection(
        data,
        "y",
        LinearRegression(),
        verbose=False,
        early_stop=True,
        stop_at_p=2,
        return_type="list",
    )
    assert result == ["a", "b"]
    result = feature_selection.forwardSelection(
        data,
        "y",
        LinearRegression(),
        verbose=False,
        early_stop=True,
        perc_min=0.2,
        stop_when=1,
        return_type="list",
    )
    assert result == ["a", "b"]
    monkeypatch.setattr(feature_selection, "_calcForwardAICs", lambda *args: 80.0)
    assert feature_selection.forwardSelection(
        data, "y", LinearRegression(), verbose=False, return_type="list"
    ) == ["a"]


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("probabilities", [False, True])
def test_forward_serial_and_parallel_return_the_same_selected_model(
    family, probabilities
):
    if family == "gaussian" and probabilities:
        return
    rng = np.random.default_rng(109)
    x = pd.DataFrame(rng.normal(size=(250, 3)), columns=["a", "b", "c"])
    signal = 1.5 * x.a - 0.8 * x.b + 0.3 * x.c
    if family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-signal)))
        estimator = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000)
    else:
        y = signal + rng.normal(size=len(x)) * 0.2
        estimator = LinearRegression()
    data = x.assign(y=y)
    serial = feature_selection.forwardSelection(
        data,
        "y",
        estimator,
        family=family,
        use_probabilities=probabilities,
        verbose=False,
    )
    workers = feature_selection.forwardSelection(
        data,
        "y",
        estimator,
        family=family,
        use_probabilities=probabilities,
        verbose=False,
        n_jobs=2,
        use_threads=True,
    )
    assert serial[0] == workers[0]
    assert_frame_equal(serial[2], workers[2])
    assert_allclose(serial[1].predict(serial[2]), workers[1].predict(workers[2]))
    assert not hasattr(estimator, "coef_")


def test_unfinished_vif_drop_fails_explicitly():
    with pytest.raises(NotImplementedError, match="modeling.vif"):
        feature_selection.vifDrop()
