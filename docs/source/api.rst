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

.. autosummary::
    :toctree: generated/
    :nosignatures:

    estimators.BayesLogRegClassifier
    estimators.BayesPoissonRegressor
    estimators.BayesWeibullRegressor

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
    nn.initializers.lecun_uniform
    nn.initializers.lecun_normal

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
