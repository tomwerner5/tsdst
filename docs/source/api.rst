.. _api_ref:

.. currentmodule:: tsdst

API reference
=============

.. _distributions_api:

Distributions
-------------

This section contains statistical functions for various distributions for a wide range of purposes.

General Functions
~~~~~~~~~~~~~~~~~

This section includes approximations about the normal distribution, as well as
calculations for the poison-binomial distribution.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.dpoibin_exact
    distributions.dpoibin_FT
    distributions.dpoibin_PA
    distributions.dwrap
    distributions.pnorm_approx
    distributions.ppoibin_RNA
    distributions.qnorm_approx

Likelihood Functions
~~~~~~~~~~~~~~~~~~~~

These common likelihood functions are presented in two formats, one that is the
simple likelihood definition (usable for any data), and one that assumes your
data is part of a GLM or similar model.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.likelihood_bernoulli
    distributions.likelihood_gaussian
    distributions.likelihood_poisson
    distributions.glm_likelihood_bernoulli
    distributions.glm_likelihood_gaussian
    distributions.glm_likelihood_poisson
    distributions.ExactMLE_exp

Posterior Functions
~~~~~~~~~~~~~~~~~~~

These functions are mainly for Bayesian Inference, and for use with the
MCMC module.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.exp_gamma
    distributions.NHPP_posterior
    distributions.pois_gamma
    distributions.pois_gamma_ada
    distributions.pois_uniform
    distributions.posterior_logreg_lasso
    distributions.ap_logreg_lasso
    distributions.posterior_poisson_lasso
    distributions.posterior_poisson_lasso_od
    distributions.ap_poisson_lasso_od
    distributions.weibull_gamma
    distributions.weibull_regression_post

.. _estimators_api:

Estimators
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    estimators.BayesLogRegClassifier
    estimators.BayesPoissonRegressor
    estimators.BayesWeibullRegressor

.. _feature_selection_api:

Feature Selection
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    feature_selection.forwardSelection
    feature_selection.getHighCorrs
    feature_selection.dropCorrProcedure
    feature_selection.dropHighCorrs
    feature_selection.naiveScoreDrop
    feature_selection.naiveVarDrop
    feature_selection.permutation_importance
    feature_selection.permutation_importance_CV
    feature_selection.vifDrop

.. _mcmc_api:

MCMC
----

MCMC Object
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mcmc.mcmcObject

MCMC Supporting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mcmc.applyMCMC
    mcmc.cholupdate
    mcmc.minESS
    mcmc.multiESS
    mcmc.raftery
    mcmc.samp_size_calc_raftery

MCMC Algorithms
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mcmc.adaptive_mcmc
    mcmc.rwm
    mcmc.rwm_with_lap

.. _modeling_api:

Modeling
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    modeling.beta_trans
    modeling.crossVal
    modeling.EstimatorSelectionHelper
    modeling.scoreModel
