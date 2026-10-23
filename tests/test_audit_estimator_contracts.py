"""Check estimator contracts and Weibull quantities against independent formulas."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse, stats

from tsdst import estimators as e
from tsdst import modeling, utils


def fixed_weibull():
    model = e.BayesWeibullRegressor()
    model.coef_ = np.array([0.2])
    model.intercept_ = np.log(2.0)
    model.scale_ = 3.0
    model.n_features_in_ = 1
    return model


@pytest.mark.parametrize("format_", ["csr", "csc", "coo"])
def test_sparse_predictions_match_dense_inputs(format_):
    x = np.array([[-1.0], [0.0], [1.0]])
    for model in (
        fixed_weibull(),
        e.BayesPoissonRegressor(),
        e.BayesLogRegClassifier(),
        e.LogReg(),
    ):
        model.coef_ = np.array([0.2])
        model.intercept_ = 0.1
        model.n_features_in_ = 1
        model.classes_ = np.array([0, 1])
        model.dispersion_delta_ = 0.0
        matrix = getattr(sparse, format_ + "_matrix")(x)
        assert_allclose(model.predict(matrix), model.predict(x))


def test_weibull_predictions_and_likelihood_match_scipy():
    model = fixed_weibull()
    x = np.array([[-2.0], [-0.5], [0.0], [1.0], [2.0]])
    shape = np.exp(np.log(2.0) + 0.2 * x.ravel())
    distribution = stats.weibull_min(shape, scale=3.0)
    y = np.array([0.4, 1.0, 2.0, 3.0, 4.0])
    assert_allclose(model.predict_shape(x), shape)
    assert_allclose(model.predict(x), distribution.mean())
    assert_allclose(model.predict_variance(x), distribution.var(), rtol=1e-12)
    assert_allclose(model.predict_log_likelihood(x, y), distribution.logpdf(y))
    status = np.array([1, 0, 1, 0, 1])
    expected = np.where(status, distribution.logpdf(y), distribution.logsf(y))
    assert_allclose(model.predict_log_likelihood(x, y, status), expected)
    weights = np.array([1.0, 2.0, 0.0, 1.0, 3.0])
    assert_allclose(
        model.score(x, y, weights, scorer="log_likelihood", status=status),
        np.average(expected, weights=weights),
    )


def test_weibull_diagnostics_use_variance_and_cumulative_hazard():
    model = fixed_weibull()
    x = np.arange(6).reshape(-1, 1) / 5
    y = np.array([0.6, 1.4, 2.0, 3.0, 3.5, 4.2])
    weights = np.array([1.0, 2.0, 1.0, 0.0, 3.0, 1.0])
    result = model._deviance_dispersion_update(x, y, weights)
    shape = model.predict_shape(x)
    hazard = (y / 3.0) ** shape
    expected_deviance = 2 * (hazard - np.log(hazard) - 1)
    distribution = stats.weibull_min(shape, scale=3.0)
    residuals = (y - distribution.mean()) / np.sqrt(distribution.var())
    assert_allclose(result["deviance_"], weights @ expected_deviance)
    assert_allclose(result["pearson_residuals_"], residuals)
    assert_allclose(result["pearson_chi2_"], weights @ residuals**2)
    assert_allclose(result["log_likelihood_"], weights @ distribution.logpdf(y))
    assert result["df_residuals_"] == 2
    assert_allclose(result["dispersion_scale_"], weights @ residuals**2 / 2)


def test_weibull_variance_remains_positive_at_large_shapes():
    model = fixed_weibull()
    model.coef_ = np.array([0.0])
    model.intercept_ = np.log(1e8)
    variance = model.predict_variance(np.zeros((2, 1)))
    assert np.all(variance > 0)
    assert_allclose(variance, 3**2 * np.pi**2 / (6 * 1e16), rtol=1e-7)


def test_constant_weibull_data_have_no_finite_null_mle():
    model = fixed_weibull()
    result = model._deviance_dispersion_update(np.zeros((4, 1)), np.ones(4))
    assert np.isnan(result["null_deviance_"])
    assert np.isnan(result["model_d2_"])


@pytest.mark.parametrize("name", ["BayesPoissonRegressor", "BayesWeibullRegressor"])
@pytest.mark.parametrize(
    "invalid", [[1, np.nan, 1, 1], [1, np.inf, 1, 1], [1, -1, 1, 1], [0, 0, 0, 0]]
)
def test_diagnostic_weights_reject_invalid_values(name, invalid):
    model = (
        fixed_weibull()
        if name.startswith("BayesWeibull")
        else e.BayesPoissonRegressor()
    )
    with pytest.raises(ValueError, match="weight"):
        model._deviance_dispersion_update(np.ones((4, 1)), np.arange(1, 5), invalid)


def test_constant_poisson_response_has_undefined_d2():
    model = e.BayesPoissonRegressor()
    model.coef_ = np.array([0.0])
    model.intercept_ = 0.0
    model.n_features_in_ = 1
    result = model._deviance_dispersion_update(np.zeros((4, 1)), np.ones(4))
    assert result["deviance_"] == 0
    assert result["null_deviance_"] == 0
    assert np.isnan(result["model_d2_"])


def test_database_url_preserves_components_and_escapes_credentials(monkeypatch):
    captured = []
    engine = Mock()
    monkeypatch.setattr(
        utils, "create_engine", lambda url, **kwargs: captured.append(url) or engine
    )
    monkeypatch.setattr(pd, "read_sql", lambda *args, **kwargs: pd.DataFrame())
    utils.dsn_getTable(
        "select 1",
        "a@b",
        "postgresql",
        "localhost:5432/example?sslmode=require",
        pw="p@:/word",
    )
    url = captured[0]
    assert (url.username, url.password) == ("a@b", "p@:/word")
    assert (url.host, url.port, url.database) == ("localhost", 5432, "example")
    assert url.query == {"sslmode": "require"}
    engine.dispose.assert_called_once()


def test_legacy_score_keeps_all_classification_fields():
    truth = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.2, 0.8, 0.9, 0.4])
    model = SimpleNamespace(
        predict=lambda x: probabilities >= 0.5,
        predict_proba=lambda x: np.column_stack([1 - probabilities, probabilities]),
    )
    result = modeling.scoreModel_legacy(np.zeros((6, 1)), truth, model, print_=False)
    expected = {
        "Accuracy",
        "F1",
        "Sens/Recall",
        "Specificity",
        "ppv",
        "npv",
        "Conf_Mat",
        "AUC",
        "McFadden R2",
        "Tjur R2",
        "Nagelkerke R2",
        "Cox/Snell R2",
    }
    assert set(result) == expected
    assert_array_equal(result["Conf_Mat"], [[2, 1], [1, 2]])
    assert_allclose(result["Sens/Recall"], 2 / 3)
