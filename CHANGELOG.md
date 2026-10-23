# Changes

## 2.1.0

- Replaced the obsolete conda publishing action and built a noarch package.
- Set the required PyPI upload username and added secret-format diagnostics.
- Added selectable Weibull shape or scale regression and conditional scale predictions.
- Added configurable shared-parameter priors and documented AFT and hazard interpretations.

- Restored factual docstrings, parameter descriptions, and explanatory comments.
- Clarified spectral covariance, sampled parameter scales, and covariance interpretations.
- Rebuilt the Sphinx documentation and completed the public API index.
- Added a docs dependency group and corrected documentation rendering and links.
- Corrected overdispersed Poisson mean predictions and aligned the adaptive likelihood.
- Kept covariate-dependent Weibull shape and a shared sampled scale.
- Added Weibull shape, variance, censored-likelihood, and survival-deviance outputs.
- Kept R2 as the default Weibull score and added an explicit log-likelihood score.
- Validated Bayesian starts, parameter layouts, sample weights, and proposal failures.
- Kept unexpected NaN and positive-infinite posterior failures visible.
- Matched regularized GLM AIC to the authors' R implementation.
- Added sampling and curvature covariance options for L2 uncertainty calculations.
- Matched OLS and logistic Wald intervals to independent R results.
- Matched Raftery sample counts to coda and retained unrounded dependence ratios.
- Corrected hidden-layer, activity-penalty, and batch-normalization gradients.
- Kept dropout predictions from changing stored normalization state.
- Restored sparse prediction support, DSN parsing, and legacy score fields.
- Counted nonzero entries for the L0 norm, including negative values.
- Kept local sampling RNGs, missing-value cutoff rules, and input-order parallel results.
- Kept the second-class decision for binary probability ties.
- Corrected forward-selection parameter counts and checked its stopping rules.

## 2.0.0

- Added automatic EDA with an optional response variable.
- Expanded all six `dwrap` operations to all five supported distributions.
- Added `dwrap(log_p=True)` for log probability inputs to quantiles.
- Kept `dwrap(log=True)` as the log output option, including for quantiles.
- Added tests for every package function and method.
- Added a check that reports package functions without test execution.
- Updated the package for Python 3.12 and current scientific packages.
- Moved build metadata and dependency requirements to `pyproject.toml`.
- Added PyArrow as an optional dependency for Feather and Parquet files.
- Replaced removed pandas methods and NumPy aliases.
- Replaced private scikit-learn imports with public estimator APIs.
- Removed unused imports and the unused statsmodels runtime dependency.
- Fixed estimator cloning, validation, binary probabilities, and model scores.
- Corrected overdispersed Poisson predictions to return the conditional count mean.
- Aligned the adaptive Poisson likelihood with the same mean parameterization.
- Fixed confusion-matrix counts, information criteria, and regression intervals.
- Fixed censored Weibull likelihoods and Weibull mean predictions.
- Fixed MCMC diagnostics and chain summaries for current pandas.
- Fixed neural-network gradients, optimizer state, and prediction samples.
- Fixed parallel task arguments and the chunked penalty sweep.
- Added a test workflow for Python 3.12, 3.13, and 3.14.

### Changes that can affect existing code

- Raised the Python and dependency requirements. Earlier versions are not supported.
- Removed MessagePack storage. Use Feather or Parquet and install `tsdst[io]`.
- Corrected `forwardSelection`'s `serialze_flavor` argument to `serialize_flavor`.
- Removed the `(1 - delta)` factor from `BayesPoissonRegressor.predict`.
  The factor remains in the likelihood's parameter conversion.
- Changed the overdispersed Poisson likelihood used with `C=None`.
  Refit models trained with that branch.
- Changed `BayesWeibullRegressor.predict` to return the Weibull mean.
  Its default score now uses R2. The optional D2 score remains a Poisson-deviance diagnostic.
- Changed `scoreModel_legacy` to use the same metric definitions as `scoreModel`.
- Changed the unfinished `vifDrop` function to raise `NotImplementedError`.
  Use `modeling.vif` to inspect variance inflation factors.

The baseline commit is `1389521`. It contains the earlier local changes and
the first EDA implementation. It precedes these tests and updates.
