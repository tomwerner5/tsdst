"""Compare statistical outputs with saved results from independent R packages."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from tsdst import estimators, mcmc, metrics, modeling

FOLDER = Path(__file__).parent / "fixtures" / "audit_statistics"


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("intercept", [0.0, 0.4])
def test_regularized_aic_matches_authors_r_package(family, intercept):
    data = pd.read_csv(FOLDER / "aic-data.csv")
    references = pd.read_csv(FOLDER / "aic-reference.csv")
    coefficients = pd.read_csv(FOLDER / "aic-unregularized-coefficients.csv")
    x = data[["x1", "x2", "x3"]].to_numpy()
    y = data.binary if family == "binomial" else data.counts

    def model(beta):
        def predict(values):
            return np.exp(beta[0] + np.asarray(values) @ beta[1:])

        def predict_proba(values):
            p = expit(beta[0] + np.asarray(values) @ beta[1:])
            return np.column_stack([1 - p, p])

        return SimpleNamespace(
            intercept_=beta[0],
            coef_=beta[1:],
            predict=predict,
            predict_proba=predict_proba,
        )

    regularized = model(np.array([intercept, 0.3, 0.0, -0.2]))
    unregularized = model(
        coefficients.loc[coefficients.family == family, "value"].to_numpy()
    )
    expected = references.loc[
        (references.family == family) & (references.intercept == intercept), "AIC"
    ].item()
    actual = metrics.glm_regularized_AIC(
        x, y, regularized, unregularized, family=family, tol=1e-5
    )
    assert_allclose(actual, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("q", [0.025, 0.5, 0.975])
@pytest.mark.parametrize("spacing", [1, 2])
def test_raftery_matches_coda_r(q, spacing):
    chain = pd.read_csv(FOLDER / "raftery-chain.csv").to_numpy()
    references = pd.read_csv(FOLDER / "raftery-reference.csv")
    expected = references.loc[(references.q == q) & (references.spacing == spacing)]
    actual = mcmc.raftery(chain, q=q, r=0.02, s=0.95, thin=spacing)
    assert_allclose(actual.iloc[:, :3], expected[["M", "N", "Nmin"]], atol=0, rtol=0)
    # coda reports three significant digits; tsdst retains the full ratio.
    rounded = np.array([float(f"{value:.3g}") for value in actual.iloc[:, 3]])
    assert_allclose(rounded, expected.I, atol=0, rtol=0)
    assert_allclose(actual.iloc[:, 3], actual.iloc[:, 1] / actual.iloc[:, 2])


@pytest.mark.parametrize("interval", ["confidence", "prediction"])
def test_ols_intervals_and_covariance_match_r(interval):
    data = pd.read_csv(FOLDER / "ols-data.csv")
    expected = pd.read_csv(FOLDER / "ols-intervals.csv")
    x = data[["x1", "x2"]]
    model = LinearRegression().fit(x, data.y)
    result = modeling.predict(
        expected[["x1", "x2"]], x, data.y, model, logit=False, interval=interval
    )
    bounds = (
        ["lwr", "upr"]
        if interval == "confidence"
        else ["prediction_lower", "prediction_upper"]
    )
    assert_allclose(result, expected[["fit"] + bounds], atol=1e-10, rtol=0)
    covariance = modeling.GenerateRegCov(x, data.y, model, logit=False)[0]
    assert_allclose(
        covariance, pd.read_csv(FOLDER / "ols-covariance.csv"), atol=1e-12, rtol=0
    )


def test_weibull_quantities_and_survival_residuals_match_r():
    expected = pd.read_csv(FOLDER / "weibull-reference.csv")
    x = np.array([-2, -0.5, 0, 1, 2]).reshape(-1, 1)
    model = estimators.BayesWeibullRegressor()
    model.coef_ = np.array([0.2])
    model.intercept_ = np.log(2.0)
    model.scale_ = 3.0
    model.n_features_in_ = 1
    assert_allclose(model.predict(x), expected["mean"], atol=1e-12, rtol=0)
    assert_allclose(model.predict_variance(x), expected.variance, atol=1e-12, rtol=0)
    assert_allclose(
        model.predict_log_likelihood(x, expected.time),
        expected.logpdf,
        atol=1e-12,
        rtol=0,
    )
    assert_allclose(
        model.predict_log_likelihood(x, expected.time, np.zeros(5)),
        expected.logsf,
        atol=1e-12,
        rtol=0,
    )
    result = model._deviance_dispersion_update(x, expected.time, status=expected.events)
    assert_allclose(
        result["deviance_residuals_"], expected.deviance, atol=1e-12, rtol=0
    )


def test_logistic_wald_intervals_match_r():
    data = pd.read_csv(FOLDER / "aic-data.csv")
    expected = pd.read_csv(FOLDER / "logistic-intervals.csv")
    x = data[["x1", "x2", "x3"]]
    model = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(x, data.binary)
    result = modeling.predict(
        expected[["x1", "x2", "x3"]], x, data.binary, model, interval="confidence"
    )
    assert_allclose(result, expected[["eta", "lower", "upper"]], atol=2e-7, rtol=0)


@pytest.mark.parametrize("logit", [False, True])
@pytest.mark.parametrize("kind", ["sampling", "curvature"])
@pytest.mark.parametrize("penalize_intercept", [False, True])
def test_penalized_covariance_matches_direct_matrix_formula(
    logit, kind, penalize_intercept
):
    if logit:
        data = pd.read_csv(FOLDER / "aic-data.csv")
        x, y = data[["x1", "x2", "x3"]], data.binary
        model = LogisticRegression(C=1 / 2.5, tol=1e-12, max_iter=10000).fit(x, y)
    else:
        data = pd.read_csv(FOLDER / "ols-data.csv")
        x, y = data[["x1", "x2"]], data.y
        model = Ridge(alpha=2.5).fit(x, y)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1])
    if not penalize_intercept:
        penalty[0, 0] = 0
    if logit:
        p = model.predict_proba(x)[:, 1]
        information = design.T @ np.diag(p * (1 - p)) @ design
        scale = 1.0
    else:
        information = design.T @ design
        ordinary = np.linalg.lstsq(design, y, rcond=None)[0]
        scale = np.sum((y - design @ ordinary) ** 2) / (len(y) - design.shape[1])
    inverse = np.linalg.solve(information + 2.5 * penalty, np.eye(design.shape[1]))
    expected = inverse @ information @ inverse if kind == "sampling" else inverse
    actual = modeling.GenerateRegCov(
        x,
        y,
        model,
        logit=logit,
        lamda=2.5,
        covariance=kind,
        penalize_intercept=penalize_intercept,
    )[0]
    assert_allclose(actual, scale * expected, atol=1e-12, rtol=0)


def test_gaussian_active_set_aic_uses_mle_variance_and_counts_sigma():
    data = pd.read_csv(FOLDER / "ols-data.csv")
    x, y = data[["x1", "x2"]], data.y
    model = LinearRegression().fit(x, y)
    residuals = y - model.predict(x)
    sigma2_mle = np.mean(residuals**2)
    log_likelihood = -len(y) / 2 * (np.log(2 * np.pi * sigma2_mle) + 1)
    # Two slopes, one fitted intercept, and one estimated variance.
    expected = -2 * log_likelihood + 2 * 4
    for method in ["kawano", "hastie"]:
        assert_allclose(
            metrics.glm_regularized_AIC(
                x, y, model, model, family="gaussian", method=method
            ),
            expected,
            atol=1e-10,
            rtol=0,
        )
    r_result = pd.read_csv(FOLDER / "ols-aic.csv")
    assert_allclose(expected, r_result.AIC.item(), atol=1e-10, rtol=0)


def test_ols_interval_coverage_on_fixed_simulations():
    rng = np.random.default_rng(607)
    x = rng.normal(size=(35, 2))
    design = np.column_stack([np.ones(35), x])
    coefficients = np.array([1.2, -0.3, 0.7])
    test_x = np.array([[0.5, -0.2]])
    true_mean = np.r_[1.0, test_x.ravel()] @ coefficients
    covered_mean = covered_prediction = 0
    repetitions = 1000
    for _ in range(repetitions):
        y = design @ coefficients + rng.normal(scale=0.6, size=35)
        model = LinearRegression().fit(x, y)
        confidence = modeling.predict(
            test_x, x, y, model, logit=False, interval="confidence"
        ).iloc[0]
        prediction = modeling.predict(
            test_x, x, y, model, logit=False, interval="prediction"
        ).iloc[0]
        new_response = true_mean + rng.normal(scale=0.6)
        covered_mean += confidence.lower_int <= true_mean <= confidence.upper_int
        covered_prediction += (
            prediction.lower_int <= new_response <= prediction.upper_int
        )
    assert 0.925 <= covered_mean / repetitions <= 0.975
    assert 0.925 <= covered_prediction / repetitions <= 0.975
