# Resolution of the modernization audit

This record describes task 1. The earlier audit remains a historical assessment
of dc0a055. Task 2 restores factual comments and docstrings from its parent and
updates the minor version in a separate commit.

## Decisions

- Kept covariate-dependent Weibull shape and one shared sampled scale.
- Kept Weibull mean predictions and R2 scoring. Added explicit shape, variance,
  and log-likelihood outputs and Weibull survival-deviance diagnostics.
- Kept the new random-state, missing-value cutoff, classification-tie, and
  input-order parallel behavior, with explicit checks of those contracts.
- Defined L0 as the count of nonzero entries, including negative entries.
- Added sampling and curvature covariance choices to the L2 uncertainty
  helpers. Sampling is the default. Existing Bayesian prior definitions remain.

## Resolved items

| Audit item | Resolution and evidence |
|---|---|
| Weibull mean predictions | Kept the accepted mean and exposed `predict_shape`. Checked mean, variance, and log density against SciPy and independent R outputs. |
| Weibull score and dispersion | Kept R2, added mean log-likelihood scoring, and replaced raw-response Poisson diagnostics with Weibull cumulative-hazard deviance residuals. The retained `over_dispersion` argument is unused; shape and scale determine variance. |
| Bayesian fitting helper | Checked coefficient/extra-parameter layout, supplied constants, label encoding, valid response domains, finite starts, sparse prediction, sampler selection, and fitted summaries. The completed Poisson helper also passed the independent fitted R comparison. |
| Bayesian initialization | Checked zeros, ones, random, and warm starts against the actual posterior. Automatic Poisson dispersion starts at zero. Weibull warm starts use its own marginal likelihood. Supplied starts are copied and validated. All-zero counts have a valid finite fallback. |
| Bayesian proposal wrapper | Negative infinity rejects unsupported proposals. Unexpected NaN or positive infinity raises a numerical error. Overdispersed Poisson uses `0 <= delta < 1`. Stable logistic likelihoods handle saturated logits without `log(0)`. |
| Weighted diagnostics | Restored finite-weight validation, scalar-weight support, positive-weight degrees of freedom, and weighted Pearson sums. Constant-response Poisson D2 is undefined and returns NaN. Weibull residuals use Weibull variance and distinguish censored observations. |
| Regularized GLM AIC | Matched binomial and Poisson results to the authors' sAIC 1.0.1 package, including a zero intercept. Retained the published factor of two and used an ordinary solve. Gaussian AIC uses maximum-likelihood residual variance and counts the estimated variance parameter; it matches R's `AIC(lm(...))`. |
| L2 covariance | Added `covariance='sampling'` and `'curvature'` and an explicit intercept-penalty option. Sampling uses the sandwich formula; curvature uses the inverse penalized information. Linear noise variance is estimated by OLS rather than inflated by shrinkage bias. Checked direct matrix formulas for both families and penalty choices. |
| Regression intervals | OLS confidence and prediction intervals match R. Logistic Wald intervals match R's normal critical values. A fixed 1,000-repetition simulation checks OLS coverage. Penalized intervals are documented approximations rather than exact coverage claims. |
| Raftery diagnostic | Matched burn-in, total samples, and minimum samples to coda 0.19-4.1 at several quantiles and sampling intervals. Used the exact normal quantile and coda's interpretation of existing sample spacing. Retained the approved unrounded dependence ratio. |
| Forward selection | Corrected intercept and Gaussian variance parameter counts. Checked independent likelihood-based criteria, stopping limits, small-improvement stopping, ties, estimator ownership, serialization, and serial/parallel agreement, including probability compression. |
| Neural-network training | Corrected activation and activity-penalty derivatives around dropout and batch normalization. Checked every parameter by finite differences across hidden layers, binary/multiclass outputs, regularization, dropout, and normalization. Checked optimizer selection, persistent Adam moments, batching, and permutation coverage. |
| Predictive neural samples | Used fixed inference normalization statistics, inferred output width, and checked dropout variability and output alignment. Predictions preserve the stored normalization state; deterministic inference does not consume dropout RNG draws. |
| Weighted sample | Kept copied weights and local RNG behavior. Checked seeded draws against Python's weighted sampling and confirmed that unseeded calls leave the global RNG state unchanged. |
| Feature-type inference | Kept exclusion of missing values from the unique-value cutoff and checked the disputed missing-value example directly. |
| L0 norm | Counts nonzero entries rather than array length or positive entries. Checks include negative values, zeros, and matrices. |
| dwrap | Kept the accepted independent log-output and log-probability-input options, all six operations, and all five families. Reference tests cover boundaries, truncation, and logarithmic tails. |
| vifDrop | Kept the approved explicit `NotImplementedError` and the `modeling.vif` alternative. The unfinished placeholder no longer reports a numeric success. |
| Memory-safe parallel helper | Kept input order and bounded batches. Checked delayed completion, exception placement, and unchanged input arguments. |
| DSN parsing regression | Restored host, port, database, and query parsing while retaining credential escaping and engine disposal. |
| Sparse prediction regression | Restored accepted sparse inputs in all four estimator prediction paths. CSR, CSC, and COO predictions match dense predictions. |
| Legacy score fields | Restored the confusion matrix, pseudo-R2 values, and lowercase `ppv`/`npv` fields. Kept the explicit legacy regression definition in the legacy function; current scoring remains available through `scoreModel`. |
| Classification ties | Kept the approved second-class decision at a zero linear predictor and checked the corresponding probabilities of one half. |

## Numerical references

The references ran locally with the installed R 4.1.0. The source packages were
sAIC 1.0.1 and coda 0.19-4.1. Their external source files were kept outside Git.
The small numerical inputs and outputs are under
`tests/fixtures/audit_statistics`; `scripts/check_audit_references.R` regenerates
them using the installed R packages or a separately supplied source directory.

- [sAIC manual](https://cran.r-project.org/web/packages/sAIC/sAIC.pdf)
- [coda Raftery implementation](https://github.com/cran/coda/blob/master/R/raftery.R)
- [R linear-model prediction](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.lm.html)
- [R GLM prediction](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.glm.html)
- [Weibull moments](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm)

The full fitted Poisson comparison is recorded separately in
[poisson-regression-comparison.md](poisson-regression-comparison.md). It uses
four chains per language, independent mode optimization, and Monte Carlo
standard-error checks for the same posterior.

## Definitions and limits

Weibull D2 compares sums of survival-deviance residuals with an intercept-only
Weibull fit. It is a descriptive comparison, not a GLM likelihood-ratio test.
Censored observations contribute log survival and martingale/deviance residuals;
ordinary Pearson residuals are undefined for those incomplete event times.
Constant complete times have no finite two-parameter Weibull null MLE, so the
corresponding null diagnostic is NaN.

The covariance choices concern the L2 standard-error helpers. They do not
replace MCMC posterior uncertainty. Sampling covariance describes repeated-data
variation of a shrunken estimator. It does not remove shrinkage bias relative
to the unpenalized population parameter. Curvature is a local Gaussian-prior
interpretation with the stated scale convention. Neither option makes generic
penalized intervals exact.

Some existing Poisson functions named `lasso` evaluate normal coefficient
priors, while the fixed-C overdispersed function uses a Laplace prior. Their
prior definitions were preserved; documentation must describe the active code.
The adaptive overdispersed mean parameterization changed and requires refitting
models trained with that branch.

The evidence checks the stated formulas and contracts. It is not a claim of
posterior calibration for every data-generating process or every sampler.
