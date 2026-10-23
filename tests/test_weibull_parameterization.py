"""Check both Weibull models against fixed formulas and an independent R fit."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse, stats
from scipy.optimize import minimize
from sklearn.base import clone

from tsdst.distributions import weibull_regression_post
from tsdst.estimators import BayesWeibullRegressor


FOLDER = Path(__file__).parent / "fixtures" / "weibull_regression"


def repeat_start(start, niter, lpost, postArgs, options):
    assert np.isfinite(lpost(start, **postArgs))
    return {"parameters": np.tile(start, (niter, 1)), "prev_vals": {}}


def fixed_model(mode):
    model = BayesWeibullRegressor(regression_parameter=mode)
    model.coef_ = np.array([0.3])
    model.intercept_ = np.log(2.0)
    model.n_features_in_ = 1
    if mode == "shape":
        model.scale_ = 3.0
    else:
        model.shape_ = 3.0
    return model


@pytest.mark.parametrize("mode", ["shape", "scale"])
@pytest.mark.parametrize("shared_prior", [None, (3.5, 0.7)])
@pytest.mark.parametrize("neg,log", [(False, True), (True, True), (False, False), (True, False)])
def test_posterior_matches_density_survival_priors_and_jacobian(mode, shared_prior, neg, log):
    x = np.column_stack([np.ones(5), [-1.0, -0.2, 0.0, 0.6, 1.2]])
    parameters = np.array([0.2, -0.3, np.log(1.7)])
    time = np.array([0.4, 0.9, 1.2, 2.1, 3.0])
    status = np.array([1, 0, 1, 0, 1])
    varying, shared = np.exp(x @ parameters[:-1]), np.exp(parameters[-1])
    shape, scale = (varying, shared) if mode == "shape" else (shared, varying)
    prior = shared_prior or ((0.01, 100.0) if mode == "shape" else (2.0, 1.0))
    distribution = stats.weibull_min(shape, scale=scale)
    expected = np.where(status, distribution.logpdf(time), distribution.logsf(time)).sum()
    expected += stats.norm.logpdf(parameters[0], scale=10)
    expected += stats.laplace.logpdf(parameters[1], scale=0.8)
    expected += stats.gamma.logpdf(shared, prior[0], scale=prior[1]) + parameters[-1]
    if not log:
        expected = np.exp(expected)
    if neg:
        expected = -expected
    actual = weibull_regression_post(parameters, x, time, status, l_scale=0.8,
                                    regression_parameter=mode, shared_prior=shared_prior,
                                    neg=neg, log=log)
    assert_allclose(actual, expected, rtol=2e-14)


@pytest.mark.parametrize("mode", ["shape", "scale"])
@pytest.mark.parametrize("status", [None, np.zeros(5), np.ones(5)])
def test_posterior_handles_complete_and_censored_samples(mode, status):
    x = np.column_stack([np.ones(5), np.linspace(-0.3, 0.3, 5)])
    time = np.linspace(0.5, 2.5, 5)
    value = weibull_regression_post(np.array([0.1, 0.2, 0.3]), x, time, status,
                                    regression_parameter=mode)
    assert np.isfinite(value)


@pytest.mark.parametrize("mode", ["shape", "scale"])
def test_predictions_scores_and_diagnostics_use_both_conditional_parameters(mode):
    model = fixed_model(mode)
    x = np.linspace(-1, 1, 6).reshape(-1, 1)
    y = np.array([0.5, 1.2, 1.8, 2.4, 3.1, 4.0])
    varying = np.exp(np.log(2) + 0.3*x.ravel())
    shape, scale = (varying, np.full(6, 3.0)) if mode == "shape" else (np.full(6, 3.0), varying)
    distribution = stats.weibull_min(shape, scale=scale)
    assert_allclose(model.predict_shape(x), shape)
    assert_allclose(model.predict_scale(x), scale)
    assert_allclose(model.predict(x), distribution.mean())
    assert_allclose(model.predict_variance(x), distribution.var(), rtol=1e-12)
    assert_allclose(model.predict(sparse.csr_matrix(x)), distribution.mean())
    status = np.array([1, 0, 1, 1, 0, 1])
    log_values = np.where(status, distribution.logpdf(y), distribution.logsf(y))
    assert_allclose(model.predict_log_likelihood(x.tolist(), y, status), log_values)
    weights = np.array([1.0, 2.0, 0.0, 1.0, 3.0, 1.0])
    assert_allclose(model.score(x, y, weights, scorer="log_likelihood", status=status),
                    np.average(log_values, weights=weights))
    result = model._deviance_dispersion_update(x, y, weights, status)
    hazard = (y/scale)**shape
    residual = np.sign(status-hazard)*np.sqrt(2*(hazard-status-status*np.log(hazard)))
    assert_allclose(result["deviance_residuals_"], residual)
    assert_allclose(result["deviance_"], weights @ residual**2)
    assert_allclose(result["log_likelihood_"], weights @ log_values)
    assert np.isnan(result["pearson_residuals_"][status == 0]).all()


def test_scale_regression_has_exact_aft_and_proportional_hazards_effects():
    model = fixed_model("scale")
    model.coef_ = np.array([np.log(1.3)])
    x = np.array([[0.0], [1.0]])
    shape, scale = model.predict_shape(x), model.predict_scale(x)
    distribution = stats.weibull_min(shape, scale=scale)
    for probability in [0.05, 0.25, 0.5, 0.9, 0.99]:
        quantiles = distribution.ppf(probability)
        assert_allclose(quantiles[1]/quantiles[0], 1.3)
    assert_allclose(model.predict(x)[1]/model.predict(x)[0], 1.3)
    expected_hazard_ratio = np.exp(-3*np.log(1.3))
    for time in [0.1, 1.0, 4.0, 10.0]:
        hazard = shape/scale*(time/scale)**(shape-1)
        assert_allclose(hazard[1]/hazard[0], expected_hazard_ratio)


def test_shape_regression_has_a_common_landmark_and_crossing_hazards():
    model = fixed_model("shape")
    model.intercept_ = np.log(0.5)
    model.coef_ = np.array([np.log(4.0)])
    x = np.array([[0.0], [1.0]])
    shape, scale = model.predict_shape(x), model.predict_scale(x)
    distribution = stats.weibull_min(shape, scale=scale)
    assert_allclose(distribution.cdf(3.0), [1-np.exp(-1)]*2)
    crossing_time = 3.0*(shape[1]/shape[0])**(1/(shape[0]-shape[1]))
    ratios = []
    for time in [crossing_time/4, crossing_time, crossing_time*4]:
        hazard = shape/scale*(time/scale)**(shape-1)
        ratios.append(hazard[0]/hazard[1])
    assert ratios[0] > 1 > ratios[2]
    assert_allclose(ratios[1], 1)
    assert not np.isclose(distribution.ppf(0.25)[1]/distribution.ppf(0.25)[0],
                          distribution.ppf(0.9)[1]/distribution.ppf(0.9)[0])


@pytest.mark.parametrize("mode", ["shape", "scale"])
@pytest.mark.parametrize("initialization", ["zeros", "ones", "random", "sklearn"])
def test_fitting_initialization_and_shared_prior_are_consistent(mode, initialization):
    np.random.seed(28)
    x = np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    y = np.tile([0.5, 1.0, 2.0, 3.0], 3)
    model = BayesWeibullRegressor(regression_parameter=mode, niter=4, algo=repeat_start,
                                  initialize_weights=initialization, shared_prior=(3, 2)).fit(x, y)
    assert model.regression_parameter_ == mode
    assert model.shared_prior_ == (3, 2)
    assert model.shared_parameter_ > 0
    assert hasattr(model, "scale_" if mode == "shape" else "shape_")
    assert not hasattr(model, "shape_" if mode == "shape" else "scale_")
    assert clone(model).get_params() == model.get_params()
    if initialization == "sklearn":
        k, _, scale = stats.weibull_min.fit(y, floc=0)
        expected = [k, scale] if mode == "shape" else [scale, k]
        assert_allclose(np.exp(model.start_[[0, -1]]), expected, rtol=1e-4)


def test_changing_parameterization_requires_refitting_and_clears_old_scalar():
    x = np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    y = np.tile([0.5, 1.0, 2.0, 3.0], 3)
    model = BayesWeibullRegressor(niter=4, algo=repeat_start).fit(x, y)
    old_prediction = model.predict(x)
    model.set_params(regression_parameter="scale")
    assert_array_equal(model.predict(x), old_prediction)
    model.fit(x, y)
    assert model.regression_parameter_ == "scale"
    assert model.shared_prior_ == (2.0, 1.0)
    assert hasattr(model, "shape_") and not hasattr(model, "scale_")
    model.mcmc_params[-1, -1] = np.log(4)
    model.adjust_params_("last_value")
    assert_allclose(model.predict_shape(x), 4)
    assert_allclose(model.shared_parameter_, 4)


@pytest.mark.parametrize("mode", ["bad", None, 1])
def test_invalid_parameterization_is_rejected(mode):
    model = BayesWeibullRegressor(regression_parameter=mode, niter=4, algo=repeat_start)
    with pytest.raises(ValueError, match="regression_parameter"):
        model.fit(np.zeros((6, 1)), np.arange(1, 7))


@pytest.mark.parametrize("prior", [(0, 1), (2, -1), (np.nan, 1), (2, np.inf), [1], [[2, 1]], "bad"])
def test_invalid_shared_priors_are_rejected(prior):
    model = BayesWeibullRegressor(shared_prior=prior, niter=4, algo=repeat_start)
    with pytest.raises(ValueError, match="shared_prior"):
        model.fit(np.zeros((6, 1)), np.arange(1, 7))


@pytest.mark.parametrize("mode", ["shape", "scale"])
def test_sampler_rejects_zero_or_nonfinite_distribution_parameters(mode):
    def sampler(start, niter, lpost, postArgs, options):
        for index in [0, -1]:
            for value in [-1000.0, 1000.0, np.nan]:
                candidate = start.copy()
                candidate[index] = value
                assert lpost(candidate, **postArgs) == -np.inf
        return repeat_start(start, niter, lpost, postArgs, options)

    x = np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    model = BayesWeibullRegressor(regression_parameter=mode, niter=4, algo=sampler)
    model.fit(x, np.tile([0.5, 1.0, 2.0, 3.0], 3))


def test_scale_mode_matches_independent_r_survreg_fit_on_censored_lung_data():
    data = pd.read_csv(FOLDER / "lung-data.csv")
    reference = pd.read_csv(FOLDER / "survreg-predictions.csv")
    parameters = pd.read_csv(FOLDER / "survreg-parameters.csv").value.to_numpy()
    summary = pd.read_csv(FOLDER / "survreg-summary.csv").iloc[0]
    x = data[["age10", "female", "ecog"]]
    model = BayesWeibullRegressor(regression_parameter="scale", start=parameters,
                                  niter=4, algo=repeat_start).fit(x, data.time, status=data.event)
    assert len(data) > 200 and 0 < data.event.sum() < len(data)
    assert_allclose(model.shape_, 1/summary.survreg_scale)
    assert_allclose(model.predict_scale(x), reference.scale, rtol=1e-13)
    assert_allclose(model.predict(x), reference["mean"], rtol=1e-13)
    assert_allclose(model.predict_variance(x), reference.variance, rtol=1e-12)
    contributions = model.predict_log_likelihood(x, data.time, data.event)
    assert_allclose(contributions, reference.log_contribution, rtol=1e-12)
    assert_allclose(contributions.sum(), summary.log_likelihood, atol=1e-10)
    distribution = stats.weibull_min(model.predict_shape(x), scale=model.predict_scale(x))
    for p, column in [(0.25, "q25"), (0.5, "median"), (0.75, "q75")]:
        assert_allclose(distribution.ppf(p), reference[column], rtol=1e-13)
    design = np.column_stack([np.ones(len(x)), x])

    def objective(params):
        shape, scale = np.exp(params[-1]), np.exp(design @ params[:-1])
        return -np.where(data.event, stats.weibull_min.logpdf(data.time, shape, scale=scale),
                         stats.weibull_min.logsf(data.time, shape, scale=scale)).sum()

    result = minimize(objective, parameters + [0.1, -0.05, 0.04, 0.05, -0.1], method="BFGS",
                      options={"gtol": 1e-6})
    assert np.linalg.norm(result.jac) < 2e-4
    assert_allclose(result.x, parameters, atol=2e-6)
    assert_allclose(-result.fun, summary.log_likelihood, atol=1e-8)


@pytest.mark.parametrize("mode", ["shape", "scale"])
def test_builtin_sampler_runs_both_models_with_right_censoring(mode):
    np.random.seed(812)
    random.seed(812)
    rng = np.random.default_rng(812)
    x = rng.normal(scale=0.3, size=(40, 1))
    shape = np.exp(0.4 + 0.2*x.ravel()) if mode == "shape" else np.full(40, 1.7)
    scale = np.full(40, 2.0) if mode == "shape" else np.exp(0.6 + 0.2*x.ravel())
    lifetime = scale*rng.weibull(shape)
    time = np.minimum(lifetime, 2.5)
    status = lifetime <= 2.5
    model = BayesWeibullRegressor(regression_parameter=mode, niter=200, algo="rwm",
                                  algo_options={"progress": False,
                                                "prev_vals": {"E_0": 0.01*np.eye(3)}}).fit(x, time, status)
    assert np.isfinite(model.mcmc_params).all()
    assert np.unique(model.mcmc_params, axis=0).shape[0] > 20
    assert np.isfinite(model.predict(x)).all()
    assert np.isfinite(model.log_likelihood_)
