"""Test estimator contracts and the full quick-analysis workflow."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import LogisticRegression

from tsdst import estimators as e, quick_analysis as q


@pytest.mark.parametrize("name", ["BayesLogRegClassifier", "BayesPoissonRegressor", "BayesWeibullRegressor", "LogReg"])
def test_estimator_clone_and_type(name):
    estimator = getattr(e, name)()
    copied = clone(estimator)
    assert copied.get_params().keys() == estimator.get_params().keys()
    if name in ["LogReg", "BayesLogRegClassifier"]:
        assert is_classifier(estimator)
    else:
        assert is_regressor(estimator)


@pytest.mark.parametrize("name", ["BayesLogRegClassifier", "LogReg"])
def test_logistic_fit_and_calibrated_probabilities(name, classification):
    x, y = classification
    options = dict(C=1, niter=80, algo="rwm", initialize_weights="zeros", verbose=False,
                    algo_options={"progress": False}) if name.startswith("Bayes") else dict(lamb=0, method="BFGS")
    model = getattr(e, name)(**options).fit(x, y)
    probability = model.predict_proba(x)
    decision = np.asarray(x) @ model.coef_.reshape(-1) + np.asarray(model.intercept_).item()
    assert_allclose(probability[:, 1], expit(decision))
    assert_allclose(np.exp(model.predict_log_proba(x)), probability)
    assert set(model.predict(x)) <= {0, 1}
    assert 0 <= model.score(x, y) <= 1
    assert model.n_features_in_ == 3
    if name.startswith("Bayes"):
        assert model.adjust_params_("median") is model


def test_logistic_labels_and_sklearn_pipeline(classification):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y = classification
    labels = np.where(y, "positive", "negative")
    model = make_pipeline(StandardScaler(), e.LogReg(lamb=0, method="BFGS"))
    model.fit(x, labels)
    assert set(model.predict(x)) <= {"positive", "negative"}
    assert model.score(x, labels) >= .5


@pytest.mark.parametrize("name", ["BayesPoissonRegressor", "BayesWeibullRegressor"])
def test_positive_response_estimators(name, regression):
    x, y = regression
    y = np.abs(np.round(y)) + 1
    model = getattr(e, name)(C=1, niter=80, algo="rwm", initialize_weights="zeros",
                             verbose=False, algo_options={"progress": False}).fit(x, y)
    assert model.predict(x).shape == (len(y),)
    assert np.all(model.predict(x) > 0)
    assert np.isfinite(model.score(x, y))
    assert np.isfinite(model._deviance_dispersion_update(x, y)["deviance_"])
    assert model.adjust_params_("median") is model


@pytest.mark.parametrize("name", ["BayesLogRegClassifier", "BayesPoissonRegressor", "BayesWeibullRegressor"])
@pytest.mark.parametrize("summary", ["mean", "median", "mode_histogram", "mode_kde", "last"])
def test_posterior_coefficient_summaries(name, summary):
    model = getattr(e, name)()
    chain = np.random.default_rng(4).normal(size=(200, 4))
    coefficients, intercept, extra = model._create_coefs(chain, summary, 1)
    assert coefficients.size == 2 and np.size(intercept) == 1 and extra.size == 1


def test_quick_lasso_and_parallel_reader(classification, tmp_path):
    x, y = classification
    count, metrics = q.calcAICsLasso(1, x, y, len(y))
    assert 0 <= count <= 4 and len(metrics) == 6 and np.isfinite(metrics).all()
    data = x.assign(y=y)
    data.to_feather(tmp_path / "data.fth")
    result = q.parallel_AIC_data_retriever(str(tmp_path), "/data", None, len(y), None,
                                          123, "y", list(x.columns), 1)
    assert result[0] == count


def test_quick_analysis_full_workflow(classification):
    x, y = classification
    data = x.assign(y=y)
    analysis = q.QuickAnalysis(data, data.iloc[:20].copy(), "y")
    analysis.GenCorrStats()
    assert len(analysis.sort_y_corr()) == 3
    analysis.genHighCorrs(.95)
    analysis.plot_y_corr()
    analysis.RunFullAnalysis(dropCor=False, dropVar=False, cv_iterations=2,
                             default_penalty=1, verbose=False, try_parallel=False,
                             reduce_features_by=2, num_cs=3, red_log_low=0, red_log_high=1,
                             for_stride=1)
    assert "baseline_mod" in analysis.steps and "final_mod" in analysis.steps
    assert analysis.aics.shape == (3, 6)
    assert analysis.confusion_matrix(print_=False).iloc[-1, -1] == 20


def test_quick_parallel_penalty_sweep(classification):
    x, y = classification
    data = x.assign(y=y)
    analysis = q.QuickAnalysis(data, data.copy(), "y")
    analysis.reduction = np.array([.5, 1., 2.])
    analysis.calcAICs(3, True, n_jobs=2, random_state=3, remove_msg=True, chunk=True)
    assert np.isfinite(analysis.aics).all()
    assert len(analysis.nonzero_coefs) == 3
