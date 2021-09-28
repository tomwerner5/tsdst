.. _api_ref:

.. currentmodule:: tsdst

API reference
=============

.. _distributions_api:

Distributions
-------------

General Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.dpoibin_exact
    distributions.dpoibin_FT
    distributions.dpoibin_PA
    distributions.dwrap
    distributions.pnorm_approx
    distributions.ppoibin_RNA
    distributions.qnorm_aprox

Likelihood Functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.likelihood_bernoulli
    distributions.likelihood_gaussian
    distributions.likelihood_poisson
    distributions.glm_likelihood_bernoulli
    distributions.glm_likelihood_gaussian
    distributions.glm_likelihood_poisson
    distributions.exactMLE_exp

Posterior Functions
~~~~~~~~~~~~~~~~~~~

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


.. _modeling_api:

Modeling
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    modeling.crossVal
    modeling.EstimatorSelectionHelper

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
    feature_selection.vif_drop

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
    mcmc.rwm_lap
