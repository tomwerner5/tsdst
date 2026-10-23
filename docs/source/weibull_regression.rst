Weibull regression
==================

Choose the regression parameter
-------------------------------

``BayesWeibullRegressor`` supports two models. In both, the selected parameter
is ``exp(intercept + X @ coefficients)``. The other parameter is shared across
observations and sampled in MCMC. Here, "shared" means the same value for all
observations; it does not mean a known value held fixed during fitting.

.. list-table:: Model choices
   :header-rows: 1

   * - Setting
     - Covariates determine
     - Shared parameter
     - Coefficient interpretation
   * - ``regression_parameter="scale"``
     - Weibull scale
     - Weibull shape
     - AFT time ratios and constant hazard ratios
   * - ``regression_parameter="shape"``
     - Weibull shape
     - Weibull scale
     - Changes in the time pattern of risk

Shape regression remains the default to preserve the existing model. Choose
scale regression explicitly when the AFT interpretation fits the application.
The choice changes the statistical model, rather than only its notation.

Scale regression
----------------

With shared shape ``k`` and scale ``lambda(x)=exp(intercept + x @ beta)``,
the survival function is

.. math::

   S(t\mid x) = \exp\left[-\left(t/\lambda(x)\right)^k\right].

Increasing one predictor by one unit multiplies every conditional lifetime
quantile by ``exp(beta_j)``. It also multiplies the mean by that factor. For
example, ``beta_j=log(1.3)`` gives a 30% increase in median lifetime, with the
other predictors held constant. This is an accelerated failure time model.

The hazard ratio is ``exp(-k*beta_j)`` and is constant over time. The Weibull
model therefore also has a proportional-hazards interpretation. Its PH
coefficient is ``-k*beta_j``, not the AFT coefficient ``beta_j``.

All observations share the same increasing, decreasing, or constant hazard
direction, according to whether ``k`` is above, below, or equal to one.
Scale regression is a common starting point when covariates act on lifetime.
It does not guarantee convergence or precise estimates under heavy censoring.

Shape regression
----------------

With shared scale ``lambda`` and shape ``k(x)=exp(intercept + x @ beta)``,
covariates change how risk evolves over time. Shapes below one give decreasing
hazards; shapes above one give increasing hazards.

All observations have ``F(lambda)=1-exp(-1)``: approximately 63.2% have failed
at the shared scale. Survival curves with different shapes cross at that time.
Their hazards also cross, though generally at another time. These effects
do not have constant time-ratio or hazard-ratio interpretations.

This model can be useful when the scientific hypothesis concerns a change in
failure mechanism or wearout pattern. Censoring can make shape effects hard
to identify. Estimation also depends on observed failures, covariate support,
and priors. Neither model is guaranteed to fail or converge solely because of
its parameter choice. This implementation supports right censoring; it does
not add left- or interval-censoring support.

Fit and predict
---------------

.. code-block:: python

   import numpy as np
   from tsdst.estimators import BayesWeibullRegressor

   rng = np.random.default_rng(42)
   X = rng.normal(size=(200, 1))
   true_scale = np.exp(1.0 + 0.3*X[:, 0])
   lifetime = true_scale*rng.weibull(2.0, size=len(X))
   censoring_time = np.full(len(X), 4.0)
   time = np.minimum(lifetime, censoring_time)
   status = (lifetime <= censoring_time).astype(int)

   model = BayesWeibullRegressor(
       regression_parameter="scale",
       shared_prior=(2.0, 1.0),
       niter=10000,
       algo_options={"progress": False},
   ).fit(X, time, status=status)

   mean = model.predict(X)
   shape = model.predict_shape(X)
   scale = model.predict_scale(X)
   variance = model.predict_variance(X)
   score = model.score(X, time, scorer="log_likelihood", status=status)

Shape and scale predictions contain one value per feature row. Mean predictions
use the selected coefficient summary and shared parameter; they do not average
conditional means over all posterior draws. Use log-likelihood or D2 scoring
for censored times. R2 remains the default for complete times.

The example illustrates the interface. Choose a sufficient chain length and
check the actual chain diagnostics for the application.

Priors and sampled parameters
-----------------------------

The intercept has a normal prior with mean zero and standard deviation 10.
Slopes have Laplace priors with scale ``2*C``. With ``C=None``, the existing
Weibull implementation uses a fixed slope-prior scale of one.

``shared_prior`` supplies Gamma **shape and scale**, in that order:

* Scale regression defaults to Gamma(2, 1) for the shared Weibull shape.
* Shape regression keeps Gamma(0.01, 100) for the shared Weibull scale.

Both hyperparameters must be positive and finite. Their interpretation differs
between modes. Weibull scale has time units, while Weibull shape is dimensionless.
Changing time units can therefore require changing a prior on shared scale.
Review prior sensitivity, particularly when censoring limits the information.

Each sampled parameter vector contains the intercept and slopes, followed by
the log shared parameter. The posterior includes the corresponding log-scale
Jacobian. Priors on the coefficients themselves require no additional Jacobian
for the regression link.

``shared_parameter_`` contains the exponentiated summary of the final log
parameter. ``scale_`` is present for shape regression and ``shape_`` is present
for scale regression. Use ``predict_scale`` and ``predict_shape`` for row-wise
values in either mode. Changing ``regression_parameter`` after fitting requires
a new fit; it does not reinterpret the existing chain.

Comparison with R
-----------------

For scale regression, R's ``survival::survreg(..., dist="weibull")`` uses the
same AFT coefficients. Its reported ``scale`` is ``1/k``, the inverse Weibull
shape. Its linear predictor is log Weibull scale. This naming difference must
be accounted for when comparing estimates and likelihoods.

See the `R survreg documentation
<https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html>`_
and the `NIST Weibull formulas
<https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm>`_.
