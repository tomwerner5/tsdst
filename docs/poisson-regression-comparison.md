# Poisson regression comparison with R

The fitted example supported returning `exp(X @ beta)` as the conditional
count mean. The old multiplication by `(1 - delta)` returned the generalized
Poisson parameter `theta` and substantially reduced the predictions.

## Data and fits

The comparison used the 54 observations in R's
[warpbreaks data](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/warpbreaks.html).
There are nine observations for each combination of two wool types and three
tension levels. The regression formula was `breaks ~ wool + tension`.
R supplied the factor coding used by both implementations.

The following fits ran locally with R 4.1.0 and the project Python environment:

- R `glm(..., family=quasipoisson())` and `glm(..., family=poisson())`.
  Their fitted means agreed within `1e-8`.
- An independent base R implementation of the fixed-C generalized Poisson
  likelihood and the current tsdst priors, fitted by optimization and MCMC.
- Python optimization of `posterior_poisson_lasso_od` and four actual
  `BayesPoissonRegressor.fit` calls with the built-in `rwm` sampler.

The Bayesian fits used `C=1000`, so the coefficient-prior scale was 2000.
The intercept and delta priors matched the existing tsdst implementation.
In particular, R reproduced the existing `dnorm(exp(delta), 0, 0.4)` prior
term without adding a Jacobian. This check reproduced that target; it did
not assess whether that prior was the intended choice.

Each language ran four independent chains of 100,000 draws. The comparison
discarded the first 20,000 draws from each chain and retained 320,000 draws
per language. It used independent seeds in R and Python. Each Python fit
also verified that its actual `predict` output equaled the inverse log link
of its stored coefficient summary.

## Fitted predictions

The Bayesian columns below use `exp(design @ mean(retained_coefficients))`.
They apply the same coefficient-summary rule as `param_summary="mean"`.
They are not averages of response means over posterior draws.
The old-formula column uses those same Python coefficients and delta.

| Wool / tension | R quasi-Poisson | R Bayesian | Python Bayesian, corrected | Python old formula |
|---|---:|---:|---:|---:|
| A / L | 40.1235 | 38.2341 | 38.3552 | 20.1979 |
| A / M | 29.0972 | 28.4313 | 28.4023 | 14.9567 |
| A / H | 23.8904 | 23.8793 | 23.8771 | 12.5737 |
| B / L | 32.6542 | 32.2334 | 32.2358 | 16.9754 |
| B / M | 23.6806 | 23.9691 | 23.8709 | 12.5704 |
| B / H | 19.4430 | 20.1315 | 20.0677 | 10.5677 |

Against the independently fitted R Bayesian values, prediction RMSE fell
from 13.4715 counts with the old formula to 0.0698 with the corrected formula.
Against R quasi-Poisson, RMSE fell from 13.9826 to 0.8378 counts.

## Checks and limits

The checks required posterior-mode agreement within `2e-5`, sampled means
within four combined Monte Carlo standard errors, ordinary split R-hat below
1.01, and batch-means effective sample sizes above 1000. All passed.

- Independent log-posterior evaluations agreed within `1e-10` absolute
  tolerance, with no relative tolerance.
- R and Python posterior modes differed by at most `3.04e-7`.
- Sampled parameter means differed by at most 2.53 combined Monte Carlo
  standard errors. Sampled response-mean averages differed by at most 2.87.
- No retained draw in either language had a negative delta.
- The R quasi-Poisson dispersion was 4.26152. The generalized Poisson
  variance multiplier evaluated at R's mean sampled delta was 3.60382.

Quasi-Poisson and Bayesian generalized Poisson are different fitting methods.
The independent R Bayesian fit was the reference for matching the same
posterior. The quasi-Poisson fit supplied a separate comparison of expected
counts and dispersion. Their estimated coefficients and dispersion need not
be identical. These results support this prediction correction; they do not
validate every prior, sampler, or underdispersion case in the package.

The adaptive `C=None` likelihood was aligned with the same mean
parameterization. Fixed-parameter tests checked its likelihood against an
independent generalized Poisson probability formula at several delta values,
and checked equality with ordinary Poisson at delta zero. The full fitted
R/Python MCMC comparison above used fixed `C=1000`, not the adaptive branch.
Models trained with the old adaptive likelihood need to be refitted.

## Reproduce the comparison

From the repository root, with Rscript and the Python dependencies installed:

```bash
mkdir -p /tmp/tsdst-poisson-regression-comparison
Rscript --vanilla scripts/compare_poisson_regression.R 100000 > /tmp/tsdst-poisson-regression-comparison/r-reference.txt
MPLCONFIGDIR=/tmp/tsdst-matplotlib NUMBA_DISABLE_JIT=1 PYTHONPATH=. python scripts/compare_poisson_regression.py --r-reference /tmp/tsdst-poisson-regression-comparison/r-reference.txt --output /tmp/tsdst-poisson-regression-comparison --iterations 100000
python -m pytest tests/test_poisson_prediction_reference.py tests/test_poisson_regression_fit.py -q
```

On this WSL machine, the R script ran through the installed Windows
`Rscript.exe`, with the script path converted to a Windows path. Set
`TSDST_RSCRIPT` to that executable when running the direct R pytest check.

The R script emits the observations, fitted values, posterior evaluations,
proposal matrix, and sampling summaries as named CSV sections. The Python
script saves these sections, its comparison tables, its retained chains,
and `summary.json` in the requested output folder. The run's artifacts are
local at `/tmp/tsdst-poisson-regression-comparison`.

Small reference files from this R run are saved under
`tests/fixtures/poisson_warpbreaks`. The deterministic regression test fits
the model again and compares its likelihood, posterior mode, and predictions
with these saved R results. The separate probability-mass tests check that
the probabilities sum to one and reproduce the stated mean and variance.
