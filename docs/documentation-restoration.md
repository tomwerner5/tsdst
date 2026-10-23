# Documentation restoration

This record covers task 2. Commit `97d1ae1` completed task 1. The documentation
review used `1f68d81`, the parent of `dc0a055`, as the historical source and
preserved the behavior accepted in task 1.

The review covered 35 changed historical docstrings and 163 missing comment
lines. Explanations were restored where they still applied. When code moved to
a shared helper, its explanation moved with it or into the caller's notes.
Parameter descriptions were checked against the current signatures and code.

| Area | Restored or clarified context |
|---|---|
| Distribution wrapper | Original R-like purpose, parameter layouts, left truncation, right censoring, and the separate log-input and log-output options. |
| Bayesian estimators | Fit inputs, label handling, initialization, parameter summaries, prior scales, retry behavior, and unused dispersion arguments. |
| Posterior functions | Sampled parameter order, normal versus Laplace coefficient priors, transformed positive scales, and Jacobian terms. |
| Forward selection | Candidate fits, temporary storage, returned objects, greedy versus exhaustive search, parameter counts, and stopping rules. |
| MCMC | Adaptive mixture construction, component selection, covariance updates, spectral covariance, and Raftery transition calculations. |
| Information criteria | Active coefficient counts, the information-matrix correction, and Gaussian noise-variance estimation. |
| Model diagnostics | Confusion-matrix axes, legacy R2, covariance assumptions, standard errors, and interval definitions. |
| Parallel helpers | Argument ownership, result order, exception behavior, process requirements, and bounded task submission. |
| QuickAnalysis | Worker inputs, penalty calculations, result storage, and temporary-file cleanup. |
| Utilities | SQL connection arguments, dtype inference, missing-value cutoffs, weighted sampling, and random-state behavior. |
| Neural networks | Gradient normalization, dropout, batch normalization, prediction state, and the layout and meaning of repeated draws. |

## Corrections to historical explanations

- `msvec` estimates spectral covariance, including serial dependence. It
  estimates the asymptotic covariance of the scaled Monte Carlo mean. Divide
  it by the chain length for covariance of the sample mean. Its input must be
  centered; `mcse_multi` performs that step.
- Adaptive mixture sampling selects the small fixed proposal when `U < beta`.
  The old explanation reversed the components. Direct mixture sampling does
  not require inverting the mixture CDF.
- The adaptive Bayesian helpers sample prior scales rather than optimize a penalty.
  Some Poisson functions with `lasso` in their names use normal priors. Their
  active prior definitions were documented without changing those definitions.
- The Weibull posterior samples shape-regression coefficients and one log
  Weibull scale. The separate `l_scale` argument controls the Laplace slope
  prior. Its log-scale Jacobian belongs to the positive Weibull scale prior.
- A Laplace slope prior contributes `abs(beta)/scale` to the negative log
  posterior. An external L1 penalty must also match the likelihood and
  intercept conventions. The old speculative universal `scale = 2/lambda`
  explanation was replaced with the actual project convention.
- Pearson residuals divide raw residuals by the square root of the variance
  function. Weibull variance comes from its shape and scale. The completed
  Weibull implementation made its old variance and initialization TODOs obsolete.
- Regression covariance is model based. The old proposed robust formula used
  unsquared residuals and was incorrect. It was replaced with an explanation
  of the assumptions and the accepted L2 covariance choices.
- The legacy regression ratio `SSR/SST` need not equal `1-SSE/SST` or squared
  correlation outside the usual fitted OLS conditions.
- L0 counts nonzero entries. Higher positive powers change the penalty;
  `abs(x)**p` is differentiable at zero for `p > 1`.
- Feather and Parquet replaced removed pandas MessagePack storage. SQL write
  arguments refer to `to_sql`, and connection credentials are escaped after
  parsing the DSN.

Obsolete debug statements, commented-out implementations, resolved TODOs, and
unsupported claims about old IDE behavior were omitted. Their useful context
was retained in the current explanations. Historical binary-label restrictions
and the old tie rule were replaced with the accepted current behavior.

## Validation and version

- Compared executable Python syntax trees with task 1 after removing
  docstrings. The only executable change was `__version__`, from `2.0.0` to
  `2.1.0`.
- Checked parameter coverage in all 35 affected historical docstrings.
- Ran 44 distribution and imputation documentation examples successfully.
- Ran the full suite: 419 passed and one optional Windows R check was skipped.
  Task 1 had already run the direct R checks with the installed Windows R.
- Updated the package version, conda recipe, and changelog to `2.1.0`.
  Sphinx reads the package version directly.
- Built the wheel and checked that its metadata reports `2.1.0`.

Installed the documentation dependencies with uv in the project .venv. The
build used Sphinx 9.1.1, numpydoc 1.11.0, sphinx-bootstrap-theme 0.8.1, and
sphinx-copybutton 0.5.2. The docs extra records these build requirements.

A clean HTML build passed with warnings treated as errors. The review corrected
broken attribute references, malformed headings, duplicate EDA entries, and
neural-network and root-finding example formatting. It removed raw template
markup that closed the theme's content container, refreshed source-code links,
and removed two stale pages for methods that no longer exist.

The API index now includes all 184 public functions and classes defined in the
package. The review checked their inventory entries and rendered pages, including
the spectral covariance, log options, confusion-matrix axes, posterior parameter
scales, and regularized covariance definitions. The updated root-finding example
also returned the two expected roots. The final check covered 292 HTML pages
and found no broken local file or anchor links.

The statistical context follows the
[mcmcse description of multivariate Monte Carlo errors](https://search.r-project.org/CRAN/refmans/mcmcse/html/mcse.multi.html),
the [coda Raftery diagnostic](https://search.r-project.org/CRAN/refmans/coda/html/raftery.diag.html),
and the numerical references recorded in
[audit-resolution.md](audit-resolution.md).
