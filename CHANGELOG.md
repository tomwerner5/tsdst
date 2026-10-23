# Changes

## 2.0.0

- Added automatic EDA with an optional response variable.
- Added tests for every package function and method.
- Added a check that reports package functions without test execution.
- Updated the package for Python 3.12 and current scientific packages.
- Moved build metadata and dependency requirements to `pyproject.toml`.
- Added PyArrow as an optional dependency for Feather and Parquet files.
- Replaced removed pandas methods and NumPy aliases.
- Replaced private scikit-learn imports with public estimator APIs.
- Removed unused imports and the unused statsmodels runtime dependency.
- Fixed estimator cloning, validation, binary probabilities, and model scores.
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
- Changed `BayesWeibullRegressor.predict` to return the Weibull mean.
  Its default score now uses R2. The optional D2 score remains a Poisson-deviance diagnostic.
- Changed `scoreModel_legacy` to use the same metric definitions as `scoreModel`.
- Changed the unfinished `vifDrop` function to raise `NotImplementedError`.
  Use `modeling.vif` to inspect variance inflation factors.

The baseline commit is `1389521`. It contains the earlier local changes and
the first EDA implementation. It precedes these tests and updates.
