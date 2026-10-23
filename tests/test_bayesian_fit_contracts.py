"""Check parameter layout, valid starts, proposal failures, and summaries."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from tsdst import estimators as e


def repeat_start(start, niter, lpost, postArgs, options):
    assert np.isfinite(lpost(start, **postArgs))
    return {"parameters": np.tile(start, (niter, 1)), "prev_vals": {}}


@pytest.mark.parametrize("family", ["binomial", "poisson", "weibull"])
@pytest.mark.parametrize("initialize", ["zeros", "ones", "random", "sklearn"])
def test_automatic_starts_are_valid_and_have_the_correct_parameter_layout(
    family, initialize
):
    x = np.linspace(-0.3, 0.3, 12).reshape(-1, 1)
    classes = {
        "binomial": e.BayesLogRegClassifier,
        "poisson": e.BayesPoissonRegressor,
        "weibull": e.BayesWeibullRegressor,
    }
    y = np.tile([0, 1], 6) if family == "binomial" else np.tile([1.0, 2.0, 3.0, 4.0], 3)
    options = {"over_dispersion": True} if family == "poisson" else {}
    model = classes[family](
        C=1,
        niter=4,
        algo=repeat_start,
        initialize_weights=initialize,
        verbose=False,
        **options,
    ).fit(x, y)
    expected = 2 if family == "binomial" else 3
    assert model.start_.shape == (expected,)
    assert model.mcmc_params.shape == (4, expected)
    if family == "poisson":
        assert model.start_[-1] == 0
    if family == "weibull" and initialize == "sklearn":
        shape, _, scale = stats.weibull_min.fit(y, floc=0)
        assert_allclose(np.exp(model.start_[[0, -1]]), [shape, scale], rtol=1e-4)


@pytest.mark.parametrize("name", ["BayesLogRegClassifier", "BayesPoissonRegressor"])
def test_adaptive_prior_starts_have_the_required_extra_parameters(name):
    model = getattr(e, name)(
        C=None, niter=4, algo=repeat_start, initialize_weights="zeros", verbose=False
    )
    y = np.tile([0, 1], 6) if "LogReg" in name else np.tile([1, 2], 6)
    model.fit(np.linspace(-0.2, 0.2, 12).reshape(-1, 1), y)
    assert model.start_.shape == (3,)
    assert model.start_[-1] == 0


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_unexpected_posterior_failures_are_not_silently_rejected(monkeypatch, invalid):
    monkeypatch.setattr(e, "posterior_logreg_lasso", lambda *args, **kwargs: invalid)
    model = e.BayesLogRegClassifier(
        C=1, niter=4, algo=repeat_start, initialize_weights="zeros", verbose=False
    )
    with pytest.raises(FloatingPointError, match="posterior"):
        model.fit(np.zeros((4, 1)), [0, 1, 0, 1])


def test_poisson_proposals_reject_unsupported_dispersion_and_keep_valid_density():
    seen = []

    def sampler(start, niter, lpost, postArgs, options):
        for delta in [-0.1, 1.0, np.inf]:
            proposal = start.copy()
            proposal[-1] = delta
            assert lpost(proposal, **postArgs) == -np.inf
        seen.append(lpost(start, **postArgs))
        return repeat_start(start, niter, lpost, postArgs, options)

    start = np.array([0.2, -0.1, 0.3])
    x = np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    y = np.tile([1, 2, 3], 4)
    model = e.BayesPoissonRegressor(
        over_dispersion=True, start=start, niter=4, algo=sampler, verbose=False
    ).fit(x, y)
    expected = e.posterior_poisson_lasso_od(
        start, np.column_stack([np.ones(12), x]), y, l_scale=2
    )
    assert_allclose(seen, [expected], atol=1e-12, rtol=0)
    assert_array_equal(model.start, start)
    assert not np.shares_memory(model.start_, start)


def test_supplied_constant_is_validated_and_used_once():
    x = np.column_stack([np.ones(12), np.linspace(-0.2, 0.2, 12)])
    y = np.tile([0, 1], 6)
    model = e.BayesLogRegClassifier(
        C=1,
        has_constant=True,
        start=[0.2, -0.1],
        niter=4,
        algo=repeat_start,
        verbose=False,
    ).fit(x, y)
    assert_allclose(model.decision_function(x), x @ [0.2, -0.1])
    with pytest.raises(ValueError, match="column of ones"):
        e.BayesLogRegClassifier(has_constant=True, niter=4, verbose=False).fit(
            x * 0.5, y
        )


def test_poisson_parameter_summary_refreshes_sampled_delta():
    model = e.BayesPoissonRegressor(over_dispersion=True)
    model.extra_params = 1
    model.mcmc_params = np.array([[0.0, 1.0, 0.1], [0.0, 3.0, 0.2], [0.0, 5.0, 0.6]])
    model.adjust_params_("median")
    assert model.dispersion_delta_ == 0.2
    assert_allclose(model.dispersion_estimation_, 1 / 0.8**2)


def test_weibull_fit_passes_right_censoring_to_the_posterior():
    x = np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    y = np.tile([1.0, 2.0, 3.0, 4.0], 3)
    status = np.tile([1, 0, 1, 0], 3)

    def sampler(start, niter, lpost, postArgs, options):
        assert_array_equal(postArgs["status"], status)
        return repeat_start(start, niter, lpost, postArgs, options)

    model = e.BayesWeibullRegressor(
        niter=4, algo=sampler, initialize_weights="zeros", verbose=False
    ).fit(x, y, status=status)
    assert_array_equal(model.status_, status)
    assert np.isfinite(model.log_likelihood_)
    assert np.isnan(model.pearson_residuals_[status == 0]).all()


@pytest.mark.parametrize("adaptive", [False, True])
def test_logistic_posteriors_remain_finite_for_saturated_logits(adaptive):
    design = np.column_stack([np.ones(2), [-1.0, 1.0]])
    coefficients = np.array([0.0, 1000.0])
    y = np.array([0.0, 1.0])
    if adaptive:
        parameters = np.r_[coefficients, 0.0]
        actual = e.ap_logreg_lasso(parameters, design, y)
        # The closed-form Laplace log density remains finite in this tail.
        expected = (
            stats.norm.logpdf(0.0, scale=1.0)
            - np.log(2.0)
            - 1000.0
            + stats.expon.logpdf(1.0, scale=0.38)
        )
    else:
        actual = e.posterior_logreg_lasso(coefficients, design, y, l_scale=2.0)
        expected = stats.norm.logpdf(0.0, scale=10.0) + stats.laplace.logpdf(
            1000.0, scale=2.0
        )
    assert np.isfinite(actual)
    assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("delta", [-0.1, 1.0, np.nan, np.inf])
def test_overdispersed_posteriors_reject_invalid_distribution_support(delta):
    x = np.column_stack([np.ones(4), np.linspace(-0.2, 0.2, 4)])
    y = np.array([0, 1, 2, 3])
    for function, parameters in [
        (e.posterior_poisson_lasso_od, [0.2, -0.1, delta]),
        (e.ap_poisson_lasso_od, [0.2, -0.1, 0.0, delta]),
    ]:
        assert function(np.array(parameters), x, y) == -np.inf
        assert function(np.array(parameters), x, y, neg=True) == np.inf
        assert function(np.array(parameters), x, y, log=False) == 0


@pytest.mark.parametrize(
    "events", [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]]
)
def test_weibull_posterior_matches_independent_densities_priors_and_jacobian(events):
    x = np.column_stack([np.ones(4), np.arange(4)])
    times = np.array([1.0, 2.0, 3.0, 4.0])
    parameters = np.array([0.1, 0.2, np.log(1.3)])
    shape = np.exp(x @ parameters[:-1])
    distribution = stats.weibull_min(shape, scale=1.3)
    likelihood = np.where(
        events, distribution.logpdf(times), distribution.logsf(times)
    ).sum()
    expected = (
        likelihood
        + stats.norm.logpdf(0.1, scale=10.0)
        + stats.laplace.logpdf(0.2, scale=1.0)
        + stats.gamma.logpdf(1.3, a=0.01, scale=100.0)
        + np.log(1.3)
    )
    assert_allclose(
        e.weibull_regression_post(parameters, x, times, np.array(events)),
        expected,
        atol=1e-12,
        rtol=0,
    )


def test_all_zero_poisson_counts_have_a_finite_start_and_undefined_d2():
    model = e.BayesPoissonRegressor(niter=4, algo=repeat_start, verbose=False).fit(
        np.zeros((6, 1)), np.zeros(6)
    )
    assert np.isfinite(model.start_).all()
    assert np.isnan(model.model_d2_)
