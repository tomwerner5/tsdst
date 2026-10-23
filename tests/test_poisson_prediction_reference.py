"""Check Poisson predictions against fixed counts, PMF moments, and base R."""

import os
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import gammaln

from tsdst.distributions import (
    ap_poisson_lasso,
    ap_poisson_lasso_od,
    posterior_poisson_lasso_od,
)
from tsdst.estimators import BayesPoissonRegressor


def generalized_poisson_logpmf(counts, theta, delta):
    """Evaluate the published GP-0 probability formula directly."""
    return (
        np.log(theta)
        + (counts - 1) * np.log(theta + delta * counts)
        - theta
        - delta * counts
        - gammaln(counts + 1)
    )


def model_with_fixed_parameters(mu, delta):
    """Set coefficients directly to test prediction without MCMC error."""
    model = BayesPoissonRegressor(over_dispersion=True, verbose=False)
    model.coef_ = np.array([0.0])
    model.intercept_ = np.log(mu)
    model.dispersion_delta_ = delta
    model.n_features_in_ = 1
    return model


def test_quasipoisson_reference_counts_have_exact_mean_and_pearson_dispersion():
    counts = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    mu = counts.mean()
    phi = np.sum((counts - mu) ** 2 / mu) / (len(counts) - 1)
    assert mu == 5
    assert_allclose(phi, 40 / 7)


@pytest.mark.parametrize("delta", [0.0, 0.2, 1 - np.sqrt(7 / 40)])
def test_fixed_c_posterior_likelihood_matches_generalized_poisson_reference(delta):
    design = np.ones((8, 1))
    counts = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    other_counts = np.zeros(8)
    parameters = np.array([np.log(5.0), delta])
    theta = 5 * (1 - delta)
    # The parameter priors cancel because both calls use the same parameters.
    difference = posterior_poisson_lasso_od(
        parameters, design, counts
    ) - posterior_poisson_lasso_od(parameters, design, other_counts)
    reference = np.sum(
        generalized_poisson_logpmf(counts, theta, delta)
        - generalized_poisson_logpmf(other_counts, theta, delta)
    )
    assert_allclose(difference, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize("delta", [0.0, 0.2, 1 - np.sqrt(7 / 40)])
def test_probability_sum_mean_and_variance_match_fixed_parameters(delta):
    counts = np.arange(10000, dtype=float)
    mu = 5.0
    probability = np.exp(generalized_poisson_logpmf(counts, mu * (1 - delta), delta))
    assert_allclose(probability.sum(), 1, atol=1e-12, rtol=0)
    assert_allclose(counts @ probability, mu, atol=1e-12, rtol=0)
    assert_allclose(
        (counts - mu) ** 2 @ probability, mu / (1 - delta) ** 2, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("delta", [0.0, 0.2, 1 - np.sqrt(7 / 40)])
def test_predict_returns_the_mean_of_its_fixed_c_likelihood(delta):
    model = model_with_fixed_parameters(5.0, delta)
    assert_allclose(model.predict(np.zeros((8, 1))), np.full(8, 5.0), atol=1e-12)


@pytest.mark.parametrize("delta", [0.0, 0.2, 0.55])
def test_adaptive_posterior_uses_the_same_conditional_mean(delta):
    design = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    counts = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    parameters = np.array([np.log(5.0), 0.3, -0.2, delta])
    theta = np.exp(design @ parameters[:-2]) * (1 - delta)
    difference = ap_poisson_lasso_od(parameters, design, counts) - ap_poisson_lasso_od(
        parameters, design, np.zeros(8)
    )
    reference = np.sum(
        generalized_poisson_logpmf(counts, theta, delta)
        - generalized_poisson_logpmf(np.zeros(8), theta, delta)
    )
    assert_allclose(difference, reference, atol=1e-12, rtol=0)


def test_zero_delta_adaptive_posterior_matches_ordinary_poisson():
    design = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    counts = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    parameters = np.array([np.log(5.0), 0.3, -0.2])
    assert_allclose(
        ap_poisson_lasso_od(np.r_[parameters, 0.0], design, counts),
        ap_poisson_lasso(parameters, design, counts),
        atol=1e-12,
    )


def test_base_r_quasipoisson_reference():
    runner = os.environ.get("TSDST_RSCRIPT") or shutil.which("Rscript")
    if runner is None:
        pytest.skip(
            "Set TSDST_RSCRIPT or install Rscript to run the direct R comparison"
        )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_quasipoisson_prediction.R"
    )
    script_argument = str(script)
    if runner.lower().endswith(".exe") and os.name != "nt":
        script_argument = subprocess.run(
            ["wslpath", "-w", str(script)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    result = subprocess.run(
        [runner, "--vanilla", script_argument],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    reference = pd.read_csv(StringIO(result.stdout))
    assert reference["case"].tolist() == ["intercept", "groups", "groups"]
    assert_allclose(reference.response, [5, 5, 10], rtol=1e-8)
    assert_allclose(reference.phi, [40 / 7, 60 / 7, 60 / 7], rtol=1e-8)
    assert_allclose(reference.se_dispersion4, 2 * reference.se_dispersion1, rtol=1e-8)
    for row in reference.itertuples():
        model = model_with_fixed_parameters(np.exp(row.eta), row.delta)
        prediction = model.predict(np.zeros((1, 1)))[0]
        assert_allclose(prediction, row.response, rtol=1e-8)
        assert not np.isclose(prediction, row.theta)
