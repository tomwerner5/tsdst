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
    positive_quantiles = np.array([.6, .75, .9])
    assert_allclose(d.dwrap(positive_quantiles, [0, 1], "inv-cdf", "normal", log=True),
                    np.log(stats.norm.ppf(positive_quantiles)))
    assert_allclose(d.dwrap(p, [2, 3], "inv-cdf", "weibull"), stats.weibull_min.ppf(p, c=2, scale=3))
    truncated = d.dwrap(np.array([1., 2.]), [2, 3, 1], "left-truncated-cdf", "weibull")
    assert_allclose(d.dwrap(truncated, [2, 3, 1], "left-truncated-inv-cdf", "weibull"), [1, 2])
    with pytest.raises(ValueError):
        d.dwrap(p, [1, 1], "pdf", "missing")


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
