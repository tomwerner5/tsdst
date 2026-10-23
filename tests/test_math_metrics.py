"""Check numerical results against reference formulas and SciPy."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, r2_score

from tsdst import metrics as m, tmath as t


def test_basic_math():
    x = np.array([-3., 0., 2.])
    assert_allclose([t.custom_abs(value) for value in x], np.abs(x))
    assert len(list(t.powerset([1, 2, 3]))) == 8
    assert t.percentIncrease(2, 3) == .5
    assert_allclose(t.rank(x), [3, 2, 1])
    assert_allclose(t.cov2cor(np.array([[4., 2.], [2., 9.]])), [[1, 1 / 3], [1 / 3, 1]])
    assert_allclose(t.scaling(x), [0., .6, 1.])
    assert t.norm(x, 2, use_expo=True) == pytest.approx(np.linalg.norm(x))
    assert_allclose(t.norm_der(x, 2), 2 * x)
    assert t.norm(x, 0) == 3


def test_rank_statistics_and_correlations():
    x, y = np.array([1., 3., 5.]), np.array([2., 4., 8.])
    result = t.mann_whitney(x, y, y_is="values")
    assert result["U1"] == stats.mannwhitneyu(x, y).statistic
    assert 0 <= t.biserial_rank_corr(x, y, y_is="values") <= 1
    frame = pd.DataFrame({"x": [1, 3, 2, 4], "y": [4, 1, 2, 3]})
    for method in ["pearson", "spearman", "kendall"]:
        assert_allclose(t.corr(frame, method=method), frame.corr(method=method))
    with pytest.raises(ValueError):
        t.corr(frame, method="missing")


def test_modes_and_distances():
    data = np.array([0., .1, .2, .2, .2, .3, 1.])
    assert t.mode_histogram(data, bin_max=3, surround=False) in data
    assert t.mode_kde(data) in data
    x = np.array([[0., 0.], [1., 2.], [2., 0.], [3., 3.]])
    centered = x - x.mean(axis=0)
    expected = np.einsum('ij,jk,ik->i', centered, np.linalg.inv(np.cov(x.T)), centered)
    assert_allclose(t.mahalanobis(x, produce="squared"), expected)
    assert_allclose(t.mahalanobis(x, produce="leverage"), expected / 3 + 1 / 4)


def test_classification_metrics(classification):
    x, y = classification
    model = LogisticRegression().fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    assert m.top_20p(y, probability) == m.top_20p_score(model, x, y)
    assert .5 <= m.auc_score(model, x, y) <= 1
    assert m.number_of_nonzero_coef(x, model) == 3
    assert m.number_of_nonzero_coef(x, SimpleNamespace()) == 3
    truth = np.array([0, 0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 0, 1])
    result = m.conf_mat_metrics(truth, prediction)
    assert result["Sens/Recall"] == .5
    assert result["Specificity"] == pytest.approx(1 / 3)
    assert result["ppv"] == pytest.approx(1 / 3)
    assert m.conf_mat_metrics(np.zeros(3), np.zeros(3), "Specificity") == 1
    assert m.accuracy(truth, prediction) == accuracy_score(truth, prediction)
    assert m.accuracy(truth[:, None], prediction[:, None]) == accuracy_score(truth, prediction)
    assert m.accuracy(np.eye(2)[truth], np.eye(2)[prediction]) == accuracy_score(truth, prediction)


def test_regression_metrics(regression):
    x, y = regression
    model = LinearRegression().fit(x, y)
    p = model.predict(x)
    assert m.bias(y, p) == pytest.approx(np.mean(p - y))
    assert m.rpmse(y, p) == pytest.approx(np.sqrt(np.mean((y - p) ** 2)))
    assert m.rpmse(y, p, root=False) == pytest.approx(np.mean((y - p) ** 2))
    assert m.r2(y, p) == pytest.approx(r2_score(y, p))
    sd = np.sqrt(np.sum((y - p) ** 2) / (len(y) - 4))
    assert m.lr_sd_fromModel(model, x, y) == pytest.approx(sd)
    assert m.lr_sd_fromPrediction(y, p, num_params=3) == pytest.approx(sd)
    assert m.adj_r2(y, p, x) <= m.r2(y, p)


def test_pseudo_r2_and_information_criteria():
    cs = m.cox_snell_r2(-10, -20, 100)
    assert cs == pytest.approx(1 - np.exp(-.2))
    assert m.nagelkerke_r2(-20, 100, cs_r2=cs) == m.nagelkerke_r2(-20, 100, ll_est=-10)
    assert m.tjur_r2(np.array([0, 1]), np.array([.1, .9])) == pytest.approx(.8)
    assert m.mcfadden_r2(-10, -20) == .5
    assert m.aicCalc(-10, 2, 100, metric="aic") == 24
    assert m.aicCalc(-10, 2, 100, metric="aicc") > 24
    assert m.aicCalc(-10, 2, 100, metric="bic") == pytest.approx(20 + 2 * np.log(100))
    assert m.aicCalc(-10, 2, 100, metric="ebic") > 24
    with pytest.raises(ValueError):
        m.aicCalc(-10, 2, 100, metric="missing")


@pytest.mark.parametrize("family", ["binomial", "gaussian", "poisson"])
def test_regularized_aic(family, classification, regression):
    x, y = classification if family == "binomial" else regression
    if family == "binomial":
        first, second = LogisticRegression(C=.5), LogisticRegression(C=100)
    elif family == "poisson":
        y = np.abs(np.round(y)).astype(int)
        first, second = PoissonRegressor(alpha=.5), PoissonRegressor(alpha=0)
    else:
        first, second = LinearRegression(), LinearRegression()
    first.fit(x, y)
    second.fit(x, y)
    for method in ["kawano", "hastie"]:
        assert np.isfinite(m.glm_regularized_AIC(x, y, first, second, method=method, family=family))


def test_probability_distances():
    p, q = np.array([.3, .7]), np.array([.6, .4])
    assert m.kl_div(p, q) == pytest.approx(stats.entropy(p, q))
    assert m.js_div(p, q) == pytest.approx(m.js_div(q, p))
    assert m.js_div(p, p) == 0
    assert m.HellingerDistanceUN(0, 0, 1, 1) == 0
    assert m.HellingerDistanceMVN(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2)) == 0
    assert m.HotellingsTwoSampleMVTtest(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), 10, 10) == 1
    assert m.HotellingsTwoSampleMVTtest(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), 10, 10, pval=False) == 0
