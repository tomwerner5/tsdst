"""Test sampler output, chain operations, and diagnostic calculations."""

from timeit import default_timer

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsdst import mcmc as m


def gaussian_log_density(x):
    return -.5 * np.dot(x, x)


def test_cholesky_updates_and_progress(capsys):
    factor = np.linalg.cholesky(np.array([[2., .5], [.5, 3.]])).T
    x = np.array([.2, .3])
    updated = m.cholupdate(factor.copy(), x.copy())
    assert_allclose(updated.T @ updated, factor.T @ factor + np.outer(x, x))
    restored = m.cholupdate(updated.copy(), x.copy(), update=False)
    assert_allclose(restored, factor)
    m._updateProgBarMCMC(1, 2, default_timer(), .5)
    assert "50.00%" in capsys.readouterr().out


@pytest.mark.parametrize("algorithm", ["rwm", "adaptive_mcmc", "rwm_with_lap"])
def test_sampler_output_and_resume(algorithm):
    fn = getattr(m, algorithm)
    result = m.applyMCMC(np.array([.1, -.1]), 100, gaussian_log_density, fn,
                         algoOpts={"progress": False})
    assert result["parameters"].shape == (100, 2)
    assert np.isfinite(result["parameters"]).all()
    assert np.std(result["parameters"]) > 0
    continuation = fn(result["parameters"][-1], 40, gaussian_log_density,
                       options={"progress": False, "prev_vals": result["prev_vals"]})
    assert_allclose(continuation["parameters"][0], result["parameters"][-1])


def test_batch_and_spectral_covariance():
    chain = np.random.default_rng(3).normal(size=(600, 2))
    for fn in [m.mbmc, m.mobmc, m.msvec]:
        covariance = fn(chain, 10)
        assert covariance.shape == (2, 2)
        assert_allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() > 0
    assert m.lag(0, 10, "bartlett") == 1
    assert m.lag(10, 10, "tukey") == 0
    adjusted = m.adjust_matrix(np.array([[1., 1.1], [1.1, 1.]]), N=600)
    assert np.linalg.eigvalsh(adjusted).min() > 0


@pytest.mark.parametrize("method", ["bm", "obm", "bartlett", "tukey"])
@pytest.mark.parametrize("r", [1, 3])
def test_mcse_and_effective_sample_sizes(method, r):
    chain = np.random.default_rng(8).normal(size=(600, 2))
    result = m.mcse_multi(chain, method=method, r=r)
    assert_allclose(result["est"], chain.mean(axis=0))
    assert result["cov"].shape == (2, 2)
    assert m.multiESS(chain, covmat=result["cov"]) > 0
    target = m.minESS(2, eps=.1)
    assert m.minESS(2, ess=target) == pytest.approx(.1, abs=.001)


def test_chain_storage_diagnostics_and_plots(tmp_path):
    chain = np.random.default_rng(7).normal(size=(600, 2))
    obj = m.mcmcObject("test")
    obj.addChain(chain, "samples")
    obj.addChain(chain[:10], "samples", concat=True)
    obj.showChain("samples")
    obj.burnin("samples", 10)
    assert obj.chains["samples_burnin10"].shape == (600, 2)
    obj.acf("samples", lag=5)
    obj.acf("samples", lag=5, partial=True)
    for plot in [obj.plotTrace, obj.plotDensity, obj.plotACF]:
        fig, axes = plot("samples", write=True, display=False, pdir=str(tmp_path) + "/")
        assert len(fig.axes) >= 2
    obj.tj_convergence_test("samples", window_size=20, num_windows=2, slide=10)
    assert len(obj.diagnostic_results["samples_Convtest"]) == 2
    diagnostics = m.raftery(chain, q=.5, r=.1, s=.9)
    assert diagnostics.shape == (2, 5)
    assert np.isfinite(diagnostics.to_numpy()).all()
    assert m.samp_size_calc_raftery(q=.5, r=.1, s=.9)[1] < len(chain)
    obj.bestRaftery("samples", q=[.25, .5], r=.1, s=.9)
    assert obj.diagnostic_results["samples_Raftery"].shape == (1, 5)
    obj.removeChain("samples", print_=False)
    assert "samples" not in obj.chains


def test_run_mcmc_with_explicit_chain_name():
    obj = m.mcmcObject()
    obj.runMCMC(np.zeros(2), 40, gaussian_log_density, m.rwm,
                 algoOpts={"progress": False}, chainName="short", do_raftery=False,
                 burnin=None, acfType=None, plot_trace=False, plot_density=False, plot_acf=False)
    assert obj.chains["short"].shape == (40, 2)
