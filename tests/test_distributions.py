"""Compare distribution functions with SciPy and exact small cases."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from tsdst import distributions as d


def test_normal_approximations():
    p = np.array([.01, .25, .5, .75, .99])
    assert_allclose(d.qnorm_approx(p), stats.norm.ppf(p), atol=.004)
    assert_allclose(d.qnorm_approx(p, lt=False), stats.norm.isf(p), atol=.004)
    x = np.linspace(-4, 4, 25)
    assert_allclose(d.pnorm_approx(x), stats.norm.cdf(x), atol=1e-7)
    assert_allclose(d.pnorm_approx(x, lt=False, log=True), stats.norm.logsf(x), atol=.001)


def test_poisson_binomial():
    p = np.array([.1, .4, .7])
    expected = np.zeros(4)
    for outcome in itertools.product([0, 1], repeat=3):
        expected[sum(outcome)] += np.prod(np.where(outcome, p, 1 - p))
    assert_allclose(d.dpoibin_exact(np.arange(4), p), expected)
    assert_allclose(d.dpoibin_FT(np.arange(4), p), expected, atol=1e-14)
    assert_allclose(d.dpoibin_exact(np.arange(4), p, cdf=True), np.cumsum(expected))
    assert_allclose(d.dpoibin_PA(np.arange(4), p), stats.poisson.pmf(np.arange(4), p.sum()))
    cdf = d.ppoibin_RNA(np.arange(4), p)
    assert np.all(np.diff(cdf) >= 0) and np.all((cdf >= 0) & (cdf <= 1))
    assert_allclose(d.dpoibin_exact([-1, 4], p), [0, 0])


@pytest.mark.parametrize("name,params,reference", [
    ("normal", [1., 2.], stats.norm(loc=1, scale=2)),
    ("exponential", [.5], stats.expon(scale=2)),
    ("weibull", [1.5, 2.], stats.weibull_min(c=1.5, scale=2)),
    ("lnorm", [.5, 1.], stats.lognorm(s=1, scale=np.exp(.5))),
    ("gamma", [2., 3.], stats.gamma(a=2, scale=3)),
])
@pytest.mark.parametrize("operation", ["pdf", "cdf", "sf"])
def test_distribution_wrapper(name, params, reference, operation):
    values = np.array([.2, 1., 3.])
    expected = getattr(reference, operation)(values)
    assert_allclose(d.dwrap(values, params, operation, name), expected)
    assert_allclose(d.dwrap(values, params, operation, name, log=True), np.log(expected))


def test_wrapper_quantiles_and_errors():
    p = np.array([.1, .5, .9])
    assert_allclose(d.dwrap(p, [2, 3], "inv-cdf", "weibull"), stats.weibull_min.ppf(p, c=2, scale=3))
    truncated = d.dwrap(np.array([1., 2.]), [2, 3, 1], "left-truncated-cdf", "weibull")
    assert_allclose(d.dwrap(truncated, [2, 3, 1], "left-truncated-inv-cdf", "weibull"), [1, 2])
    with pytest.raises(ValueError):
        d.dwrap(p, [1, 1], "pdf", "missing")


@pytest.mark.parametrize("log_p", [False, True])
@pytest.mark.parametrize("name,params,reference", [
    ("normal", [1., 2.], stats.norm(loc=1, scale=2)),
    ("weibull", [2., 3.], stats.weibull_min(c=2, scale=3)),
    ("exponential", [.5], stats.expon(scale=2)),
    ("lnorm", [.5, 1.], stats.lognorm(s=1, scale=np.exp(.5))),
    ("gamma", [2., 3.], stats.gamma(a=2, scale=3)),
])
def test_wrapper_quantiles_follow_r_probability_convention(name, params, reference, log_p):
    """R's q functions use log.p for the input probability."""
    p = np.array([0., .25, .5, .75, .975, 1.])
    with np.errstate(divide="ignore"):
        values = np.log(p) if log_p else p
        actual = d.dwrap(values, params, "inv-cdf", name, log_p=log_p)
    assert_allclose(actual, reference.ppf(p), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("log_p", [False, True])
@pytest.mark.parametrize("shape,scale,threshold", [(1., 2., 3.), (2., 3., 1.)])
def test_truncated_weibull_quantiles_use_conditional_probabilities(shape, scale, threshold, log_p):
    p = np.array([0., .1, .25, .5, .9, 1.])
    reference = stats.weibull_min(c=shape, scale=scale)
    expected = reference.ppf(reference.cdf(threshold) + reference.sf(threshold)*p)
    with np.errstate(divide="ignore"):
        values = np.log(p) if log_p else p
        actual = d.dwrap(values, [shape, scale, threshold],
                         "left-truncated-inv-cdf", "weibull", log_p=log_p)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    recovered_log_p = d.dwrap(actual[1:-1], [shape, scale, threshold],
                              "left-truncated-cdf", "weibull", log=True)
    assert_allclose(recovered_log_p, np.log(p[1:-1]), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("name,params,operation", [
    ("normal", [1., 2.], "inv-cdf"),
    ("weibull", [.5, 3.], "inv-cdf"),
    ("weibull", [.5, 3., 1.], "left-truncated-inv-cdf"),
    ("exponential", [.5], "inv-cdf"),
    ("lnorm", [.5, 1.], "inv-cdf"),
    ("gamma", [2., 3.], "inv-cdf"),
])
@pytest.mark.parametrize("log_p,values", [
    (False, np.array([-.1, 1.1, np.inf, np.nan])),
    (True, np.array([.1, np.inf, np.nan])),
])
def test_wrapper_quantiles_reject_invalid_probabilities(name, params, operation, log_p, values):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        actual = d.dwrap(values, params, operation, name, log_p=log_p)
    assert np.isnan(actual).all()


def test_normal_log_quantiles_preserve_extreme_probabilities():
    log_p = np.array([-800., -1e-20])
    actual = d.dwrap(log_p, [1., 2.], "inv-cdf", "normal", log_p=True)
    assert np.isfinite(actual).all()
    assert_allclose(stats.norm.logcdf(actual, loc=1, scale=2), log_p, rtol=1e-12)


@pytest.mark.parametrize("operation,threshold", [
    ("inv-cdf", 0.), ("left-truncated-inv-cdf", 1.),
])
def test_weibull_log_quantiles_preserve_tail_probabilities(operation, threshold):
    log_p = np.array([-100., -1e-20])
    params = [2., 3.] if operation == "inv-cdf" else [2., 3., threshold]
    expected = np.array([3*np.exp(-50) if threshold == 0 else threshold,
                         3*np.sqrt((threshold/3)**2 + 20*np.log(10))])
    actual = d.dwrap(log_p, params, operation, "weibull", log_p=True)
    assert_allclose(actual, expected, rtol=1e-13, atol=0)


def test_normal_log_density_remains_a_log_likelihood():
    observations = np.array([-40., -1., 0., 1., 40.])
    log_density = d.dwrap(observations, [0., 1.], "pdf", "normal", log=True)
    assert np.isfinite(log_density).all()
    negative_log_likelihood = -log_density.sum()
    assert negative_log_likelihood == pytest.approx(-stats.norm.logpdf(observations).sum())


@pytest.mark.parametrize("name,params,reference", [
    ("normal", [1., 2.], stats.norm(loc=1, scale=2)),
    ("exponential", [.5], stats.expon(scale=2)),
    ("weibull", [2., 3.], stats.weibull_min(c=2, scale=3)),
    ("lnorm", [.5, 1.], stats.lognorm(s=1, scale=np.exp(.5))),
    ("gamma", [2., 3.], stats.gamma(a=2, scale=3)),
])
@pytest.mark.parametrize("operation", ["inv-cdf", "left-truncated-inv-cdf"])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("log_p", [False, True])
def test_quantile_input_and_output_scales_are_independent(name, params, reference, operation, log, log_p):
    p = np.array([.75, .9, .975])
    if operation == "left-truncated-inv-cdf":
        params = params + [1.]
        expected = reference.ppf(reference.cdf(1.) + reference.sf(1.)*p)
    else:
        expected = reference.ppf(p)
    values = np.log(p) if log_p else p
    actual = d.dwrap(values, params, operation, name, log=log, log_p=log_p)
    assert_allclose(actual, np.log(expected) if log else expected, rtol=1e-12)


def test_logged_normal_quantiles_keep_existing_domain_behavior():
    with np.errstate(invalid="ignore", divide="ignore"):
        actual = d.dwrap([.25, .5, .75], [0., 1.], "inv-cdf", "normal", log=True)
    assert np.isnan(actual[0])
    assert actual[1] == -np.inf
    assert actual[2] == pytest.approx(-.3937987996003316)


@pytest.mark.parametrize("name,params,reference", [
    ("normal", [1., 2.], stats.norm(loc=1, scale=2)),
    ("exponential", [.5], stats.expon(scale=2)),
    ("weibull", [2., 3.], stats.weibull_min(c=2, scale=3)),
    ("lnorm", [.5, 1.], stats.lognorm(s=1, scale=np.exp(.5))),
    ("gamma", [2., 3.], stats.gamma(a=2, scale=3)),
])
@pytest.mark.parametrize("log", [False, True])
def test_left_truncated_cdf_has_conditional_support(name, params, reference, log):
    values = np.array([-1., 0., 1., 2., 3., np.inf, np.nan])
    expected = np.where(values <= 1., 0.,
                        (reference.cdf(values) - reference.cdf(1.))/reference.sf(1.))
    with np.errstate(divide="ignore"):
        actual = d.dwrap(values, params + [1.], "left-truncated-cdf", name, log=log)
        expected = np.log(expected) if log else expected
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name,params,reference", [
    ("normal", [1., 2.], stats.norm(loc=1, scale=2)),
    ("exponential", [.5], stats.expon(scale=2)),
    ("weibull", [2., 3.], stats.weibull_min(c=2, scale=3)),
    ("lnorm", [.5, 1.], stats.lognorm(s=1, scale=np.exp(.5))),
    ("gamma", [2., 3.], stats.gamma(a=2, scale=3)),
])
@pytest.mark.parametrize("operation", ["pdf", "cdf", "sf"])
@pytest.mark.parametrize("log", [False, True])
def test_wrapper_density_and_probability_boundaries(name, params, reference, operation, log):
    values = np.array([-np.inf, -1., 0., np.inf, np.nan])
    method = "log" + operation if log else operation
    actual = d.dwrap(values, params, operation, name, log=log)
    with np.errstate(invalid="ignore"):
        expected = getattr(reference, method)(values)
    if operation == "pdf":
        expected[np.isinf(values)] = -np.inf if log else 0.
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_wrapper_broadcasts_parameter_rows_and_accepts_lists():
    params = [[0., 1.], [2., 3.]]
    values = np.array([0., 1., 2.])[:, None]
    actual = d.dwrap(values, params, "pdf", "normal")
    assert_allclose(actual, stats.norm.pdf(values, loc=[0., 2.], scale=[1., 3.]))
    actual = d.dwrap(np.log([.75, .9]), params, "inv-cdf", "normal", log_p=True)
    assert_allclose(actual, stats.norm.ppf([.75, .9], loc=[0., 2.], scale=[1., 3.]))
    assert_allclose(d.dwrap([0., 1.], .5, "cdf", "exponential"), stats.expon.cdf([0., 1.], scale=2.))


@pytest.mark.parametrize("name,params,lower", [
    ("normal", [1., 2.], -np.inf), ("exponential", [.5], 0.),
    ("weibull", [2., 3.], 0.), ("lnorm", [.5, 1.], 0.), ("gamma", [2., 3.], 0.),
])
@pytest.mark.parametrize("log_p", [False, True])
def test_truncated_quantile_endpoints_respect_original_support(name, params, lower, log_p):
    p = np.array([0., .5, 1.])
    with np.errstate(divide="ignore"):
        values = np.log(p) if log_p else p
    below_support = d.dwrap(values, params + [-np.inf], "left-truncated-inv-cdf", name, log_p=log_p)
    untruncated = d.dwrap(values, params, "inv-cdf", name, log_p=log_p)
    assert_allclose(below_support, untruncated, rtol=1e-12, atol=1e-14)
    assert below_support[0] == lower
    conditional = d.dwrap(values, params + [1.], "left-truncated-inv-cdf", name, log_p=log_p)
    assert conditional[0] == pytest.approx(1.)
    assert conditional[-1] == np.inf


@pytest.mark.parametrize("operation", ["left-truncated-cdf", "left-truncated-inv-cdf"])
@pytest.mark.parametrize("cutoff", [np.inf, np.nan])
def test_empty_or_invalid_truncation_returns_nan(operation, cutoff):
    assert np.isnan(d.dwrap([.25, .75], [0., 1., cutoff], operation, "normal")).all()


@pytest.mark.parametrize("operation", ["pdf", "cdf", "sf", "left-truncated-cdf"])
def test_log_probability_inputs_are_only_valid_for_quantiles(operation):
    params = [0., 1., 0.] if operation.startswith("left-truncated-") else [0., 1.]
    with pytest.raises(ValueError, match="log_p applies only"):
        d.dwrap([0.], params, operation, "normal", log_p=True)


@pytest.mark.parametrize("params", [[], [0.], [0., 1., 2.], np.ones((1, 1, 2))])
def test_wrapper_rejects_wrong_parameter_shapes(params):
    with pytest.raises(ValueError, match="parameters per row"):
        d.dwrap([0.], params, "cdf", "normal")


@pytest.mark.parametrize("name,params", [
    ("normal", [0., 0.]), ("exponential", [-1.]), ("weibull", [2., -3.]),
    ("lnorm", [0., -1.]), ("gamma", [-2., 3.]),
])
def test_wrapper_invalid_distribution_parameters_return_nan(name, params):
    assert np.isnan(d.dwrap([0., 1.], params, "cdf", name)).all()


def test_wrapper_rejects_unknown_operation():
    with pytest.raises(ValueError, match="distribution type"):
        d.dwrap([0.], [0., 1.], "missing", "normal")


@pytest.mark.parametrize("name,observed,predicted,reference", [
    ("bernoulli", [0, 1, 1], [.2, .8, .7], stats.bernoulli),
    ("poisson", [0, 2, 4], [.5, 2., 3.], stats.poisson),
])
def test_discrete_likelihoods(name, observed, predicted, reference):
    observed, predicted = np.array(observed), np.array(predicted)
    fn = getattr(d, "likelihood_" + name)
    expected = reference.logpmf(observed, predicted).sum()
    assert fn(observed, predicted, neg=False) == pytest.approx(expected)
    assert fn(observed, predicted) == pytest.approx(-expected)
    assert fn(observed, predicted, neg=False, log=False) == pytest.approx(np.exp(expected))


def test_gaussian_and_glm_likelihoods():
    x = np.column_stack([np.ones(4), np.arange(4)])
    beta = np.array([.2, .1])
    y = np.array([0., 1., 0., 2.])
    assert d.likelihood_gaussian(y, x @ beta, sigma=2, neg=False) == pytest.approx(stats.norm.logpdf(y, x @ beta, 2).sum())
    for name, transform in [("bernoulli", lambda z: 1 / (1 + np.exp(-z))),
                             ("poisson", np.exp), ("gaussian", lambda z: z)]:
        kwargs = {"sigma": 1} if name == "gaussian" else {}
        expected = getattr(d, "likelihood_" + name)(y, transform(x @ beta), **kwargs)
        actual = getattr(d, "glm_likelihood_" + name)(beta, x, y, lamb=0, **kwargs)
        assert actual == pytest.approx(expected)


def test_exponential_mle():
    rate, se, interval = d.exponential_mle(np.array([1., 2., 3.]), np.array([1, 1, 0]))
    assert rate == pytest.approx(1 / 3)
    assert se == pytest.approx(rate / np.sqrt(2))
    assert interval[0] < rate < interval[1]


@pytest.mark.parametrize("name,params", [
    ("posterior_logreg_lasso", [.1, .2]), ("ap_logreg_lasso", [.1, .2, 0.]),
    ("posterior_poisson_lasso", [.1, .2]), ("ap_poisson_lasso", [.1, .2, 0.]),
    ("posterior_poisson_lasso_od", [.1, .2, .1]),
    ("ap_poisson_lasso_od", [.1, .2, 0., .1]),
    ("weibull_regression_post", [.1, .2, 0.]),
])
def test_regression_posteriors(name, params):
    x = np.column_stack([np.ones(4), np.arange(4) / 4])
    y = np.array([1., 1., 0., 1.]) if "logreg" in name else np.array([1., 2., 3., 4.])
    fn = getattr(d, name)
    value = fn(np.array(params), x, y)
    assert np.isfinite(value)
    assert fn(np.array(params), x, y, neg=True) == pytest.approx(-value)
    assert fn(np.array(params), x, y, log=False) == pytest.approx(np.exp(value))


@pytest.mark.parametrize("name,param", [("pois_uniform", .2), ("pois_gamma", .2),
                                        ("pois_gamma_ada", np.array([.2, 0., 0.]))])
def test_poisson_posteriors(name, param):
    fn = getattr(d, name)
    data = np.array([0, 1, 2])
    assert np.isfinite(fn(param, data))
    assert fn(param, data, neg=True) == pytest.approx(-fn(param, data))


def test_survival_posteriors():
    data = np.array([1., 2., 3., 4.])
    status = np.array([1, 0, 1, 0])
    assert np.isfinite(d.exp_gamma(.2, data, status))
    assert np.isfinite(d.weibull_gamma(np.array([.1, .2]), data, status))
    assert np.isfinite(d.weibull_regression_post(np.array([.1, .2, 0.]),
                       np.column_stack([np.ones(4), np.arange(4)]), data, status))
    args = (np.array([0., .1]), np.arange(1., 5.), 2., np.array([1, 2, 1]), .5, 3., 2., 1.)
    assert np.isfinite(d.NHPP_posterior(*args))
    assert d.NHPP_posterior(*args, neg=True) == pytest.approx(-d.NHPP_posterior(*args))
