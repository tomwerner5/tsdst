# Weibull regression references

Generated locally with R 4.1.0 and survival 3.2.11 by
`scripts/check_weibull_regression.R`. The data came from `survival::lung`.
Complete cases for time, status, age, sex, and ECOG score gave 227 observations:
164 observed deaths and 63 right-censored times.

The model was `Surv(time, event) ~ age10 + female + ecog`, with
`age10=(age-60)/10`, `female=sex-1`, and a Weibull distribution. R's AFT
coefficients were:

| Parameter | Value |
|---|---:|
| Intercept | 6.22599943 |
| Age per ten years | -0.07475439 |
| Female | 0.40109054 |
| ECOG score | -0.33963810 |
| Weibull shape | 1.367785 |
| Log likelihood | -1132.43874588232 |

The CSV files retain more precision. R's reported `survreg` scale was
0.731108992164631, the inverse Weibull shape. It is distinct from the
covariate-dependent Weibull scale.

The checks compare means, variances, quantiles, and every observed or censored
likelihood contribution with R. They also optimize the likelihood independently
in Python and compare its coefficients and likelihood with `survreg`.

A repeated-parameter sampler feeds the R coefficients through the estimator's
fit path to check parameter translation and diagnostics. This does not claim
that an MCMC posterior mean equals an MLE. Separate checks compare both Bayesian
posteriors with explicit SciPy likelihoods, priors, and Jacobians, and run the
built-in sampler for both parameterizations.

Reference: [R survival documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html).
