.. _api_ref:

.. currentmodule:: tsdst

API reference
=============

Automatic EDA
-------------

Explore tabular data without a response variable. See :doc:`eda` for examples.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    eda.auto_eda
    eda.EDAReport

.. _distributions_api:

Distributions
-------------

This section contains statistical functions for various distributions for a wide range of purposes.

General Functions
~~~~~~~~~~~~~~~~~

This section includes approximations about the normal distribution, as well as
calculations for the Poisson-binomial distribution.

``dwrap`` supports the normal, exponential, Weibull, lognormal, and gamma
distributions. Each supports ``pdf``, ``cdf``, ``sf``, ``inv-cdf``,
``left-truncated-cdf``, and ``left-truncated-inv-cdf``. Append the lower cutoff
to the distribution parameters for a truncated operation.

The two log options control separate scales:

.. list-table:: Quantile input and output options
    :header-rows: 1

    * - ``log_p``
      - ``log``
      - Input
      - Output
    * - False
      - False
      - Probability
      - Quantile
    * - True
      - False
      - Log probability
      - Quantile
    * - False
      - True
      - Probability
      - Log quantile
    * - True
      - True
      - Log probability
      - Log quantile

``log=True`` keeps its existing meaning: return the log of the output.
For quantiles, use ``log_p=True`` to accept log probabilities, as in R's
``qnorm(..., log.p=TRUE)``. The same options apply to truncated quantiles.
Logged quantiles are finite only for positive quantiles. Zero produces
negative infinity, and a negative quantile produces NaN.

.. code-block:: python

    import numpy as np
    from tsdst.distributions import dwrap

    p = np.array([0.75, 0.975])
    q = dwrap(p, [0, 1], "inv-cdf", "normal")
    q_from_log_p = dwrap(np.log(p), [0, 1], "inv-cdf", "normal", log_p=True)
    log_q = dwrap(p, [0, 1], "inv-cdf", "normal", log=True)
    log_q_from_log_p = dwrap(
        np.log(p), [0, 1], "inv-cdf", "normal", log=True, log_p=True
    )

For densities, ``log=True`` returns log densities. Sum these values for the
log likelihood of independent observations. Negate the sum for the negative
log likelihood:

.. code-block:: python

    observations = np.array([-1.0, 0.0, 1.0])
    nll = -dwrap(observations, [0, 1], "pdf", "normal", log=True).sum()

``log_p`` is an input option for quantiles only. Use ``log=True`` for log
outputs from ``pdf``, ``cdf``, ``sf``, and ``left-truncated-cdf``.

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

    distributions.exponential_mle
    distributions.likelihood_bernoulli
    distributions.likelihood_gaussian
    distributions.likelihood_poisson
    distributions.glm_likelihood_bernoulli
    distributions.glm_likelihood_gaussian
    distributions.glm_likelihood_poisson

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

Bayesian Estimators
~~~~~~~~~~~~~~~~~~~

Bayesian Estimators using MCMC to calculate model parameters and associated errors.

``BayesPoissonRegressor.predict`` returns the conditional count mean,
``exp(intercept + X @ coefficients)``, with or without overdispersion.
For the generalized Poisson likelihood, ``theta = mean * (1 - delta)``
and the conditional variance is ``mean / (1 - delta)**2``.
Predictions use the coefficient summary selected by ``param_summary``.
They do not average count means over all posterior draws.

The adaptive overdispersion likelihood used with ``C=None`` now uses this
same mean parameterization. Refit models trained with that branch.

``BayesWeibullRegressor`` accepts ``regression_parameter="shape"`` (the existing
model) or ``regression_parameter="scale"`` (a Weibull AFT model). The selected
parameter varies with covariates; the other parameter is a shared sampled
scalar. ``predict_shape`` and ``predict_scale`` expose both parameters,
``predict`` returns the mean, and ``predict_variance`` returns the variance.
``fit`` and likelihood scoring accept right-censoring indicators through
``status``. The ``over_dispersion`` argument remains unused for this distribution.
See :doc:`weibull_regression` for interpretation, priors, and examples.

Weibull ``score`` defaults to R2. ``scorer="log_likelihood"`` returns the
weighted mean log density or log survival contribution. ``scorer="D2"``
compares sums of Weibull survival-deviance residuals against a fitted null
Weibull model. This descriptive D2 is not a GLM likelihood-ratio test.
The Pearson residuals use the actual Weibull variance; their empirical
inflation estimate is a diagnostic, not a separate distribution parameter.

Generalized Poisson fitting uses ``0 <= delta < 1`` for overdispersion.
Unexpected NaN or positive-infinite posterior values raise an error. Negative
infinity remains a valid rejection of a proposal outside posterior support.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    estimators.BayesLogRegClassifier
    estimators.BayesPoissonRegressor
    estimators.BayesWeibullRegressor

Regularized regression uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GenerateRegCov``, ``RegressionSE``, and ``modeling.predict`` accept
``covariance="sampling"`` or ``covariance="curvature"``. Sampling is the
default. With information matrix S and L2 penalty matrix P, set A=S+lamda*P.
Sampling covariance uses A^-1 S A^-1; curvature uses A^-1. Linear regression
multiplies the matrix by an OLS noise-variance estimate. Logistic regression
uses its fitted Bernoulli information matrix and unit dispersion.

A separately fitted intercept is unpenalized by default. Set
``penalize_intercept=True`` to include it in the penalty. The supplied
``lamda`` must match the fitted model's effective L2 penalty. These options
apply to the standard-error helpers, not the Bayesian posterior priors.

OLS intervals use Student t critical values. Logistic Wald intervals use
normal critical values. Intervals for penalized fits are approximations:
sampling covariance describes the shrunken estimator's variation and does
not remove shrinkage bias. Curvature intervals use a local Gaussian-prior
interpretation with an estimated noise scale when needed.

.. _feature_selection_api:

Feature Selection
-----------------

Value Drop Procedures
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    feature_selection.getHighCorrs
    feature_selection.dropCorrProcedure
    feature_selection.dropHighCorrs
    feature_selection.naiveScoreDrop
    feature_selection.naiveVarDrop
    feature_selection.vifDrop

Selection Algorithms
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    feature_selection.forwardSelection

Feature Importance
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    feature_selection.permutation_importance
    feature_selection.permutation_importance_CV

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

MCMC Diagnostics
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

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

Modeling Tools
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    modeling.beta_trans
    modeling.bPCA
    modeling.crossVal
    modeling.EstimatorSelectionHelper
    modeling.getPriors
    modeling.prettyConfMat
    modeling.printScores
    modeling.RegressionSE
    modeling.runScorers
    modeling.scoreModel
    modeling.vif

.. _nn_api:

Neural Networks
---------------

Activations
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nn.activations.create_jac_mask
    nn.activations.cross_entropy
    nn.activations.cross_entropy_binary
    nn.activations.cross_entropy_binary_der
    nn.activations.cross_entropy_der
    nn.activations.elu
    nn.activations.elu_der
    nn.activations.gelu
    nn.activations.gelu_approx
    nn.activations.gelu_der
    nn.activations.gelu_speedy
    nn.activations.gelu_speedy_der
    nn.activations.linear
    nn.activations.linear_der
    nn.activations.mse
    nn.activations.mse_der
    nn.activations.mse_linear_der
    nn.activations.relu
    nn.activations.relu_der
    nn.activations.selu
    nn.activations.selu_der
    nn.activations.sigmoid
    nn.activations.sigmoid_cross_entropy_binary_der
    nn.activations.sigmoid_der
    nn.activations.sigmoid_scale
    nn.activations.sigmoid_scale_der
    nn.activations.softmax
    nn.activations.softmax_cross_entropy_der
    nn.activations.softmax_cross_entropy_der_fullmath
    nn.activations.softmax_der
    nn.activations.softmax_oneout

Initializers
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nn.initializers.he_normal
    nn.initializers.he_uniform
    nn.initializers.lecun_uniform
    nn.initializers.lecun_normal
    nn.initializers.random_normal
    nn.initializers.xavier_normal
    nn.initializers.xavier_uniform

Model
~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nn.model.NeuralNetwork

Optimizers
~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nn.optimizers.adam
    nn.optimizers.gradient_descent

Additional distribution functions
---------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distributions.ap_poisson_lasso

Additional estimators
---------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    estimators.LogReg

Additional MCMC tools
---------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mcmc.lag
    mcmc.adjust_matrix
    mcmc.mbmc
    mcmc.mobmc
    mcmc.msvec
    mcmc.mcse_multi

Metrics
-------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    metrics.top_20p
    metrics.top_20p_score
    metrics.auc_score
    metrics.cox_snell_r2
    metrics.nagelkerke_r2
    metrics.tjur_r2
    metrics.mcfadden_r2
    metrics.number_of_nonzero_coef
    metrics.conf_mat_metrics
    metrics.bias
    metrics.rpmse
    metrics.r2
    metrics.lr_sd_fromModel
    metrics.lr_sd_fromPrediction
    metrics.adj_r2
    metrics.aicCalc
    metrics.glm_regularized_AIC
    metrics.js_div
    metrics.kl_div
    metrics.HellingerDistanceMVN
    metrics.HellingerDistanceUN
    metrics.HotellingsTwoSampleMVTtest
    metrics.accuracy

Additional modeling tools
-------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    modeling.scoreModel_legacy
    modeling.GenerateRegCov
    modeling.predict
    modeling.corrmat_validity_check
    modeling.BPCA_initmodel
    modeling.BPCA_dostep

Optimization
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    optimization.sign
    optimization.signIntervals
    optimization.Brent_1DRoot
    optimization.AdaptiveNMorBrent
    optimization.max_like

Parallel tasks
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    parallel.p_prog_simp
    parallel.p_prog_simp_memorySafe

Quick analysis
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    quick_analysis.calcAICsLasso
    quick_analysis.parallel_AIC_data_retriever
    quick_analysis.QuickAnalysis

Sampling
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    sampling.downSample
    sampling.genRandSampFromDF
    sampling.latinHypercube1D

Mathematical tools
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tmath.custom_abs
    tmath.powerset
    tmath.percentIncrease
    tmath.rank
    tmath.cov2cor
    tmath.mann_whitney
    tmath.biserial_rank_corr
    tmath.corr
    tmath.mode_histogram
    tmath.mode_kde
    tmath.norm
    tmath.norm_der
    tmath.mahalanobis
    tmath.scaling

Utilities
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    utils.dsn_getTable
    utils.dsn_saveTable
    utils.pretty_print_time
    utils.checkDir
    utils.print_message_with_time
    utils.print_time
    utils.save_checkpoint
    utils.updateProgBar
    utils.move_columns_to_end
    utils.keywordSearch
    utils.toFrame
    utils.inferFeatureType
    utils.reshape_to_vector
    utils.decision_boundary_1D
    utils.decision_boundary
    utils.one_hot_encode
    utils.one_hot_decode
    utils.sample
