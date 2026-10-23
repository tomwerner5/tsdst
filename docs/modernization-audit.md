# Audit of commit dc0a055

This commit did more than update dependencies. It combined compatibility work,
repairs to existing code, changes to statistical outputs, and API changes.
The tests did not establish that all of those changes were correct.

The audit found reproducible regressions. It also found numerical repairs with
independent checks. Those findings should be reviewed separately.

## Scope and evidence

- Compared `dc0a055` with its parent, `dc0a055^`.
- Found 79 functions or methods whose parsed code changed after removing
  docstrings. This includes added helpers and methods. It excludes changes to
  imports, class bases, packaging, and dependency versions.
- Recorded 34 audit cases in a separate comparison script. The cases include
  verified repairs, deliberate changes, equivalent results, and regressions.
- Left implementation changes from this audit out of the package. The pending
  `dwrap` expansion and its new `log_p` argument are separate work.

To reproduce the cases from an environment with the current dependencies:

```bash
PYTHONPATH=. MPLBACKEND=Agg NUMBA_DISABLE_JIT=1 \
python scripts/audit_modernization.py --output /tmp/tsdst-modernization-evidence.json
```

The script loads individual function bodies from the two Git revisions.
Unchanged helper functions and third-party dependencies come from the current
environment. It supplies removed public imports explicitly. For the two
classification tie checks, it restores NumPy's removed `int` alias in an
isolated namespace so that the old comparison can run. This does not recreate
the complete 2021 environment or prove package-wide equivalence.

The existing suite exercised all package functions, but execution coverage is
not input coverage. For example, a test that only checks that AIC is finite
does not establish that its penalty matches the cited statistical method.

## Regressions and output contracts that need correction

| Function | Before | In dc0a055 | Finding |
|---|---|---|---|
| `utils.dsn_getTable`, `utils.dsn_saveTable` | Parsed a DSN containing a host, port, database path, and query as URL components. | Passed the complete DSN as the `host` argument to `URL.create`. | Broke DSNs such as `localhost:5432/example?sslmode=require`. The resulting URL has no port, database, or query, and its hostname contains the entire DSN. The intended password-escaping improvement did not justify losing URL parsing. |
| `BayesPoissonRegressor.predict` and `BayesWeibullRegressor.predict` | Explicitly accepted sparse input. | Used `validate_data` with its default rejection of sparse input. | Removed working input support. The Poisson comparison reproduced a valid old prediction of `e` followed by a new `TypeError` for the same CSR input. Public validation did not require this restriction. |
| The Poisson and Weibull `_deviance_dispersion_update` methods | Used scikit-learn's sample-weight validation, which rejected NaN weights. | Checked shape, negativity, and the total, but did not check finiteness. | Accepted NaN weights and returned NaN diagnostics. The Poisson case reproduced the difference. The Weibull method contains the same validation code. |
| `modeling.scoreModel_legacy` | Returned the confusion matrix, four pseudo-R2 values, and lowercase `ppv` and `npv` keys. | Delegated to `scoreModel`, whose default classification result has seven metrics and uppercase `PPV` and `NPV` keys. | Lost returned information. Repairing the confusion counts did not require removing these fields. |
| `BayesLogRegClassifier.predict`, `LogReg.predict` | Chose the first class when the linear predictor was exactly zero: `score > 0`. | Chose the second class: `score >= 0`. | Changed a tie convention without a requirement. Replacing `np.int` did not require changing the comparison. |

The SQLAlchemy API accepts the host, port, database, and query as separate
fields. The audit checked the resulting URL fields without opening a database
connection. [SQLAlchemy URL documentation](https://docs.sqlalchemy.org/en/20/core/engines.html#creating-urls-programmatically)

## Numerical changes with independent support

### Censored Weibull regression

The existing model calculated one shape per observation:

```python
shape = np.exp(X @ parms[:-1])
fails = Y[status == 1]
shape_fail = shape[status == 1]
```

Each failure needs its own shape in the Weibull log density:

```text
log(shape_i) - log(scale)
+ (shape_i - 1) * log(time_i / scale)
- (time_i / scale)**shape_i
```

The shape vector and the times therefore need the same failure mask. The
change from `shape` to `shape_fail` in both the `xlogy` term and the power term
corrected the pairing. A censored observation contributes its log survival
probability instead. [NIST Weibull density](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm),
[NIST likelihood for censored data](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm)

For four observations with different shapes, the comparison produced:

| Event status | Before | After | Independent SciPy reference |
|---|---|---|---|
| `[1, 0, 1, 0]` | Broadcasting error | -24.302294232139225 | -24.30229423213922 |
| `[0, 1, 0, 0]` | -29.854912278043503 | -24.70411556063912 | -24.704115560639117 |
| `[1, 1, 1, 1]` | -22.536922385237055 | -22.53692238523706 | -22.536922385237055 |
| `[0, 0, 0, 0]` | Broadcasting error | -24.89246449351985 | -24.892464493519846 |

These are posterior values. The reference combined SciPy Weibull log densities
for failures, log survival probabilities for censored observations, the same
priors, and the existing log-scale Jacobian term. The all-failure result agreed
because the full shape vector and the failure-only shape vector were identical.
With one failure, the old code silently broadcast that failure against all four
shapes. This was an indexing repair, not a change to the chosen regression model.

### Binary logistic probabilities

The old probability methods applied softmax to `[-z, z]`. The positive-class
probability was therefore `sigmoid(2*z)`. The fitted logistic likelihood used
`sigmoid(z)`. With the same fitted coefficient and `z=1`, the probabilities were:

| Calculation | Positive-class probability |
|---|---|
| Before | 0.8807970779778823 |
| After | 0.7310585786300049 |
| `1 / (1 + exp(-1))` | 0.7310585786300049 |

Using `expit(z)` made prediction use the same link as the fitted likelihood.
This changed probability estimates, log probabilities, calibration, and any
metric that depends on the predicted probabilities. The separate zero-score
tie change was not required. [SciPy logistic sigmoid definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)

### Confusion-matrix metrics

Scikit-learn places actual classes on rows and predicted classes on columns.
Thus `[0, 1]` is false positive and `[1, 0]` is false negative. The old code
swapped these assignments. For truth `[0,0,0,1,1]` and predictions `[0,1,1,0,1]`,
recall changed from `1/3` to `1/2`, and precision changed from `1/2` to `1/3`.
The corrected results follow the counts. This also affected callers in model
scoring and quick analysis. [Scikit-learn confusion-matrix definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### Neural-network gradients

The output derivative was already divided by `batch_m * m_scale`. The old
weight and bias calculations divided the data gradient by that quantity again.
The repair removed that second division and kept the penalty scaling.

For a three-observation sigmoid network without regularization:

| Gradient | Before | After | Central finite difference of the stated average loss |
|---|---|---|---|
| Weight | -0.09934693039019671 | -0.2980407911705901 | -0.2980407911745697 |
| Bias | -0.045233810359580476 | -0.13570143107874144 | -0.1357014310754412 |

This justifies the scaling change for that loss and architecture. It does not
validate every activation, regularizer, hidden-layer combination, dropout
setting, or batch-normalization calculation.

### Spectral covariance

`msvec` added the zero-lag term once before its loop, then added a symmetric
zero-lag contribution in the loop as well. Its lag slices also omitted the
last available pair. The replacement counted zero lag once and used all valid
positive-lag pairs.

With the centered chain `[-1, 1, -1, 1]` and bandwidth one, only zero lag should
contribute. The old result was `2.5`; the replacement and direct second-moment
calculation both gave `1.0`. A second fixture checked positive-lag terms
against explicit matrix calculations. This changes spectral MCSE and any ESS
calculated from that covariance. This is not a calibration study of all MCSE
methods. [Spectral variance reference](https://arxiv.org/abs/1507.08266)

### Other direct repairs

- `exponential_mle` used the mathematical norm function's nonexistent `.ppf`
  attribute. The SciPy normal distribution was already imported as `normal`.
  Using that alias restored its stated confidence interval.
- `HotellingsTwoSampleMVTtest` stored the calculated F statistic in a variable
  named `f`, then called `.sf` on that scalar. The replacement called the
  separately named F distribution. The p-value matched the F survival function.
- `sigmoid_scale` used undefined `s` instead of its `scale` parameter. The
  replacement matched the scaled logistic formula.
- `beta_trans(..., type_='decimal')` compared the built-in `type` instead of
  the argument. It returned `None` before and the stated decimal change after.
- `mcse_multi` called `np.zeros(0, (p, p))`, which supplied a tuple as a dtype.
  It also had comparison expressions where method-name assignments were
  intended, and left the Tukey estimate unset for `r=1`. These were direct
  errors, but the complete statistical estimator still needs broader checks.

## Changes that need an explicit behavior decision or stronger evidence

| Change | Justification and remaining issue |
|---|---|
| Weibull `predict`: shape to mean | The fitted posterior defines `exp(X beta)` as shape. The mean is `scale * Gamma(1 + 1/shape)`. For shape 2 and scale 3, the output changed from 2 to 2.658680776358274. That is the correct mean, but changing what `predict` means was a model API decision. It should have been reviewed separately. Shape should remain available if that was the intended prediction target. [NIST Weibull mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm) |
| Weibull default score: D2 to R2 | Poisson deviance was not a Weibull likelihood score. R2 is a possible regression score, but it was a new default, not a compatibility requirement. The retained D2 option remained a Poisson-deviance diagnostic. |
| Bayesian fitting helper | Fixed missing constructor attributes, cloning, the missing Weibull scale parameter, label encoding, and parameter-count checks. It also changed initialization, finite-input requirements, handling of supplied constants, sampler-name validation, and fitted attributes. The prior functions were mostly unchanged, but finite chains can change when initialization changes. No posterior-equivalence study was done. The existing 80-draw smoke tests are not sufficient to establish posterior accuracy. |
| Bayesian initialization | Extra parameters changed from initial ones to zeros. Weibull's Poisson warm start changed to zero coefficients plus log mean response as scale. These can affect finite chains and acceptance. Some original starts were invalid, but the new defaults still require a documented choice and convergence checks. |
| Bayesian proposal wrapper | Converted every nonfinite posterior value to negative infinity. Rejecting invalid proposals is reasonable, but it can also hide a numerical NaN or positive infinity from a faulty calculation. It needs targeted tests and a clear error policy. |
| Weighted deviance diagnostics | Added weights to Pearson chi-square and honored them in scores. This repaired inconsistent weighting. It also changed constant-response handling and introduced the nonfinite-weight regression listed above. Weibull's retained Poisson diagnostics do not become Weibull-specific inference by this change. |
| `glm_regularized_AIC` | Corrected broken active-column indexing, a missing intercept in the design, and undefined Poisson weight variables. It also changed Gaussian weights, forced the intercept into the count, changed the Kawano penalty from `trace` to `2*trace`, used a pseudoinverse, and restricted method strings. The test only checked finiteness. I cannot justify the full replacement as a validated implementation of the named method. It needs a derivation and reference fixtures before use. The cited paper alone does not verify this code. [Ninomiya and Kawano publication](https://www.ism.ac.jp/editsec/resmemo/resmemo-file/resm1187.htm) |
| Regression covariance with `lamda != 0` | Changed the penalty from `lambda * I` to a matrix with an unpenalized intercept. This fits the usual separately fitted intercept convention, but changes covariance and intervals. In the fixture, intercept variance changed from 0.0117829251 to 0.0459309423. Ordinary least-squares covariance with `lamda=0` agreed exactly in the fixture. Logistic `lamda` was previously ignored and is now applied, another statistical behavior change. |
| Regression intervals | Fixed a return-value unpacking error and replaced the incorrect linear interval matrix product with one variance per prediction. The formulas agree with the usual OLS construction when its assumptions hold. The existing test checked shape and interval ordering, not interval coverage or all regularized cases. |
| `raftery` rewrite | Replaced fragile pandas tables with explicit transition counts. Also added error conditions, bounded burn-in below by zero, and stopped rounding the dependence factor. Its structure can be compared with `coda`, but no direct R-output comparison was run. Existing tests only checked finite output and shape. It remains a statistical rewrite that needs reference benchmarks. [coda implementation](https://raw.githubusercontent.com/cran/coda/master/R/raftery.R) |
| Forward selection | Fixed incorrect likelihood argument names, cloned estimators, and removed obsolete serialization. Also implemented previously ineffective early stopping, changed ties from continuing to stopping, changed progress output, and returned a cloned fitted model. These can change selected columns and caller state. A small selected-feature test did not validate every family, stopping rule, or probability-based path. |
| Neural-network training | Fixed GELU/random mappings, honored the selected optimizer, kept Adam moments between batches, made oversized batches a single batch, and changed shuffle from sampling with replacement to a permutation. These change training trajectories. The gradient check supports one part; it is not a reference study of every training mode. |
| Predictive neural samples | Fixed the wrong number of unpacked return values and changed forward propagation to inference mode with sampling enabled. The mode change controls batch-normalization behavior and needs separate sampling checks. |
| `sample` | Copied weights and added validation, repairing caller-weight mutation. Also switched to a local RNG. Calls with an explicit seed remain reproducible, but calls relying on `random.seed(...)` with `random_state=None` no longer use that global stream. The local-RNG policy was a behavior choice. [Python RNG state documentation](https://docs.python.org/3/library/random.html) |
| `inferFeatureType` | Used public pandas dtype checks and accepted lists. Changed the unique-count cutoff to exclude missing values. `[1,1,NaN]` with cutoff 1 changed from numeric to categorical. Boolean classification agreed in the fixture; it was already categorical before. The missing-value rule was a policy choice. EDA already drops missing values before calling this helper. [pandas unique-count behavior](https://pandas.pydata.org/docs/reference/api/pandas.Series.nunique.html) |
| `norm(..., p=0)` | Moved an existing `len(x)` branch before division by zero. This restored the old written intent. It did not implement standard L0: for `[-3,0,2]`, the replacement returns 3 and the conventional nonzero count is 2. The test asserted 3, so it did not validate the mathematical term L0. This needs a naming or definition decision. [NumPy norm definitions](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) |
| `dwrap` in this commit | Replaced a nonexistent `normal.logppf` call with log of the output quantile. That made the branch computable, but assumed its intended semantics. The later discussion clarified that log output should remain available and logged probability inputs should have their own argument. That later `log_p` work is not evidence about the original commit. |
| `vifDrop` | Replaced an unfinished function's constant `0` return with `NotImplementedError`. This prevents a false success, but is still an API behavior change. |
| Memory-safe parallel helper | Changed result order from completion order to input order and scheduled tasks in bounded batches. Input alignment is useful, but this was a contract and scheduling change. |

## Compatibility work intended to preserve numerical definitions

- Replaced `np.int` and `np.product` with current equivalents.
- Replaced pandas append operations with concatenation. Replaced implicit
  scalar-array conversion with explicit scalar indexing.
- Replaced seaborn's old shaded-KDE argument with its current fill argument.
- Replaced private scikit-learn inheritance and validation with public APIs.
  The sparse-input and sample-weight regressions show that this replacement
  was not behavior-neutral in every case.
- Replaced `penalty='l1'` with `l1_ratio=1` for the current logistic API. The
  audit fit both forms on the same data and seed and obtained equal
  coefficients and intercepts in the current environment.
  [Scikit-learn logistic parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Removed unused statsmodels imports and the runtime dependency. Replaced
  `add_constant` calls with explicit design construction; that construction
  has its own intercept-handling implications noted above.
- Replaced removed MessagePack storage with Feather and Parquet. This changed
  accepted serialization formats and temporary-file locations, not a
  statistical definition. A version bump makes an API break possible; it
  does not establish numerical correctness.
- Updated build metadata, dependency bounds, Python requirements, CI, and the
  version. Running on new numerical-library versions can itself change fitted
  outputs. This audit did not compare against a frozen 2021 environment.

## What the current evidence does not establish

- It does not establish that more bugs were fixed than introduced. Counting
  passing tests or executed functions cannot answer that question.
- It does not validate the complete regularized-AIC derivation, Bayesian
  posterior accuracy, Raftery results against R, confidence-interval coverage,
  or all neural-network configurations.
- It does not turn API choices into bug fixes. Prediction targets, score
  defaults, returned fields, random-state behavior, and tie conventions need
  to be deliberate.
- It does not justify publishing the modernization as if it preserved outputs.
  The confirmed regressions need correction, and the statistical rewrites need
  either stronger evidence or a smaller, separately reviewed change.

## Function inventory

The table below accounts for every function or method with a non-docstring
code change. Coordinates refer to `dc0a055`, not the pending working tree.
An entry marked review is not a claim that the replacement was validated.

| Function or method | Commit line | Finding | Output or contract justification |
|---|---|---|---|
| `distributions.dwrap` | 328 | Review | Replaced the nonexistent logppf call with log of the quantile; output semantics were assumed. Later log_p work is separate. |
| `distributions.exponential_mle` | 834 | Repair | Called the imported normal distribution for confidence limits instead of the mathematical norm function. |
| `distributions.weibull_regression_post` | 1387 | Verified repair | Applied the event mask to both shape terms. Mixed, single, all, and zero-failure fixtures matched SciPy density/survival calculations. |
| `estimators.BayesLogRegClassifier.__init__` | 107 | Compatibility | Stored the original algo argument for estimator cloning; sampler lookup moved to fit. Unknown-name behavior changed there. |
| `estimators.BayesLogRegClassifier.decision_function` | 279 | Mixed | Added public linear-predictor methods and explicit handling of supplied intercept columns. Their validation rejects sparse input. |
| `estimators.BayesLogRegClassifier.fit` | 275 | Review | Consolidated fitting, validation, initialization, sampler dispatch, and proposal rejection. Finite chains and fitted attributes can change; no posterior-equivalence benchmark was run. |
| `estimators.BayesLogRegClassifier.predict` | 286 | Unnecessary change | Changed zero-score ties from the first class to the second class. The NumPy alias repair did not require changing > to >=. |
| `estimators.BayesLogRegClassifier.predict_proba` | 290 | Verified repair | Replaced sigmoid(2*z), produced by softmax([-z,z]), with sigmoid(z), which matches the fitted logistic likelihood. |
| `estimators.BayesPoissonRegressor.__init__` | 344 | Compatibility | Stored the original algo argument for estimator cloning; sampler lookup moved to fit. Unknown-name behavior changed there. |
| `estimators.BayesPoissonRegressor._deviance_dispersion_update` | 439 | Mixed/regression | Applied sample weights to Pearson chi-square and changed constant-response handling; lost validation of nonfinite weights. |
| `estimators.BayesPoissonRegressor.fit` | 530 | Review | Consolidated fitting, validation, initialization, sampler dispatch, and proposal rejection. Finite chains and fitted attributes can change; no posterior-equivalence benchmark was run. |
| `estimators.BayesPoissonRegressor.predict` | 534 | Regression | Preserved the exponential mean formula and added intercept/feature validation, but removed accepted sparse input. |
| `estimators.BayesPoissonRegressor.score` | 541 | Mixed | Honored D2 sample weights and the standard scorer interface. Custom scorer call signatures changed. |
| `estimators.BayesWeibullRegressor.__init__` | 556 | Mixed | Stored constructor arguments and allocated the missing scale parameter. Parameter dimensions changed; Bayesian fitting remains a separate review. |
| `estimators.BayesWeibullRegressor._deviance_dispersion_update` | 687 | Mixed/regression | Applied sample weights to Pearson chi-square and changed constant-response handling; lost validation of nonfinite weights. |
| `estimators.BayesWeibullRegressor.adjust_params_` | 706 | Repair | Updated the fitted scale after changing the posterior summary; downstream mean predictions use that scale. |
| `estimators.BayesWeibullRegressor.fit` | 713 | Review | Consolidated fitting, validation, initialization, sampler dispatch, and proposal rejection. Finite chains and fitted attributes can change; no posterior-equivalence benchmark was run. |
| `estimators.BayesWeibullRegressor.predict` | 717 | Review/regression | Changed predictions from shape to the Weibull mean and removed sparse input. Correct mean formula does not by itself authorize a new prediction target. |
| `estimators.BayesWeibullRegressor.score` | 725 | Review | Changed the default from Poisson D2 to R2; repaired feature slicing and scorer invocation. This was a scoring API decision. |
| `estimators.LogReg.__init__` | 742 | Compatibility | Stored constructor parameters as attributes for cloning; stopped mutating a persistent optimizer-argument dictionary. |
| `estimators.LogReg.decision_function` | 767 | Mixed | Added public linear-predictor methods and explicit handling of supplied intercept columns. Their validation rejects sparse input. |
| `estimators.LogReg.fit` | 751 | Repair plus review | Replaced an incompatible observed-probability objective with the coefficient-based logistic likelihood; added binary label encoding and finite-data validation. |
| `estimators.LogReg.predict` | 774 | Unnecessary change | Changed zero-score ties from the first class to the second class. The NumPy alias repair did not require changing > to >=. |
| `estimators.LogReg.predict_proba` | 778 | Verified repair | Replaced sigmoid(2*z), produced by softmax([-z,z]), with sigmoid(z), which matches the fitted logistic likelihood. |
| `estimators.LogReg.score` | 786 | Repair | Removed the reference to a nonexistent scoring attribute and returned classifier accuracy. |
| `estimators._fit_bayesian` | 26 | Review | Consolidated fitting, validation, initialization, sampler dispatch, and proposal rejection. Finite chains and fitted attributes can change; no posterior-equivalence benchmark was run. |
| `estimators._fit_bayesian.log_posterior` | 73 | Review | Consolidated fitting, validation, initialization, sampler dispatch, and proposal rejection. Finite chains and fitted attributes can change; no posterior-equivalence benchmark was run. |
| `feature_selection._calcForwardAICs` | 340 | Repair plus review | Fixed invalid likelihood argument names and cloned the estimator. The underlying criterion and parameter count were not independently benchmarked. |
| `feature_selection._doParallelForward` | 389 | Mixed | Removed MessagePack support, fixed predictor-list handling, and used the probability predictor only when available. These affect previously broken parallel paths. |
| `feature_selection._prepare_for_parallel` | 404 | API change | Changed temporary-file paths and accepted formats to Feather/Parquet, with one table per file. |
| `feature_selection.forwardSelection` | 419 | Review | Implemented stopping options, stopped on equal scores, cloned models, and changed serialization and cleanup. Selected features and caller state can change. |
| `feature_selection.vifDrop` | 487 | API change | Replaced a constant-zero placeholder with NotImplementedError; no statistical algorithm was implemented. |
| `mcmc.mcmcObject.bestRaftery` | 1208 | Mixed | Replaced append and took one maximum per diagnostic column. Changed inputs inherit the raftery rewrite; summaries need reference fixtures. |
| `mcmc.mcmcObject.plotDensity` | 1760 | Compatibility | Used current seaborn x/fill keywords; intended plotted data and KDE definition were unchanged. |
| `mcmc.mcmcObject.runMCMC` | 1470 | Compatibility | Read the first diagnostic scalar with iloc instead of converting an array to int. Chain extension still depends on changed Raftery results. |
| `mcmc.mcmcObject.tj_convergence_test` | 1273 | Compatibility | Replaced removed dataframe append with concatenation; intended diagnostic equations were unchanged. |
| `mcmc.mcse_multi` | 764 | Repair plus review | Fixed covariance allocation, method-name assignments, and the missing r=1 Tukey covariance. Statistical calibration remains incompletely tested. |
| `mcmc.msvec` | 753 | Verified repair | Counted zero lag once and included all valid lag pairs. Independent zero-lag and positive-lag calculations agreed. |
| `mcmc.multiESS` | 960 | Repair | Kept one-parameter covariance two-dimensional so eigenvalue calculations can run. ESS also inherits covariance changes. |
| `mcmc.raftery` | 1024 | Review | Replaced pandas transition tables with NumPy counts and changed guards, rounding, and burn-in bounds. No direct coda result comparison was run. |
| `mcmc.rwm` | 483 | Repair/API change | Renamed saved proposal state E_O to the E_0 key used when resuming. First-run proposals were unchanged by this edit. |
| `metrics.HotellingsTwoSampleMVTtest` | 654 | Verified repair | Called the F distribution survival function instead of calling sf on the scalar test statistic. |
| `metrics.accuracy` | 699 | Input extension | Flattened binary labels/scores and retained one-hot argmax. Binary column behavior is equivalent; the new docstring overstates support for arbitrary class labels. |
| `metrics.conf_mat_metrics` | 232 | Verified repair | Corrected swapped false positives/negatives, kept a 2x2 table for single-class data, and added metric-key aliases. |
| `metrics.glm_regularized_AIC` | 519 | Unverified statistical rewrite | Repaired broken indexing/variables, but also changed the Fisher matrices, intercept count, trace factor, and pseudoinverse policy. Finite-output tests are insufficient. |
| `modeling.GenerateRegCov` | 557 | Mixed/review | Ordinary least-squares covariance agreed in the fixture. Fixed logistic return state and avoided a diagonal observation matrix, but changed intercept penalization and logistic lamda behavior. |
| `modeling.beta_trans` | 713 | Verified repair | Used type_ rather than the built-in type in the decimal branch. |
| `modeling.crossVal` | 377 | Repair | Selected existing classification/regression defaults instead of referencing undefined DEFAULT_METRICS. |
| `modeling.getPriors` | 745 | Repair | Excluded the named response column rather than assuming it was the last column. Feature summaries can therefore change. |
| `modeling.predict` | 611 | Repair plus review | Fixed covariance tuple unpacking and the linear prediction-variance calculation. Interval coverage and regularized assumptions remain unverified. |
| `modeling.prettyConfMat` | 534 | Repair/API change | Handled missing label maps and margins=False, included observed classes, and changed invalid-label errors. Numeric counts match confusion-matrix definitions. |
| `modeling.runScorers` | 213 | Repair | Selected existing classification/regression defaults instead of referencing undefined DEFAULT_METRICS. |
| `modeling.scoreModel` | 35 | Input repair | Flattened responses consistently and accepted NumPy input. Metrics also inherit corrected confusion counts. |
| `modeling.scoreModel_legacy` | 208 | Output contract loss | Delegation removed four pseudo-R2 fields and the confusion matrix, renamed ppv/npv, and changed regression metric definitions. This needed separate review. |
| `nn.activations.sigmoid_scale` | 124 | Verified repair | Replaced undefined s with the scale parameter; agreed with the scaled logistic formula. |
| `nn.model.NeuralNetwork.__init__` | 25 | Mixed | Fixed undefined random and wrong GELU mappings, honored optimizer choice, copied configuration, and separated scorer names. Ownership and training behavior changed. |
| `nn.model.NeuralNetwork.back_prop` | 456 | Verified for one architecture | Removed a second division of the data gradient by batch size/m_scale. A binary one-layer finite-difference check agreed; broader architectures remain unverified. |
| `nn.model.NeuralNetwork.draw_predictive_samples` | 719 | Mixed/review | Unpacked all six forward results and selected inference mode while sampling. Batch-normalization and predictive-sampling semantics need further checks. |
| `nn.model.NeuralNetwork.fit` | 562 | Mixed/review | Fixed scorer lookup, oversized batches, Adam state resets, and shuffle-with-replacement. These intentionally change training trajectories. |
| `optimization.AdaptiveNMorBrent` | 215 | Repair | Normalized missing options/argument mappings before indexing or keyword expansion. No intended numerical change with explicit equivalent mappings. |
| `optimization.Brent_1DRoot` | 80 | Repair | Normalized missing options/argument mappings before indexing or keyword expansion. No intended numerical change with explicit equivalent mappings. |
| `optimization.signIntervals` | 16 | Repair | Updated the reference sign after a crossing. Prevented false repeated brackets after the first root. |
| `parallel.p_prog_simp` | 7 | Mixed | Stopped mutating shared arguments and honored verbose in serial mode. Per-task keys now override duplicate common keys in serial mode instead of raising TypeError. |
| `parallel.p_prog_simp_memorySafe` | 37 | API/scheduling change | Stopped mutating shared arguments and returned input order rather than completion order. Bounded waves replaced continuous replenishment. |
| `quick_analysis.QuickAnalysis.build_model` | 402 | Compatibility | Used current L1 logistic parameters; equal coefficients/intercepts were reproduced for old and new parameter forms in the current environment. |
| `quick_analysis.QuickAnalysis.calcAICs` | 216 | Repair plus API change | Retained all chunk results instead of only the last chunk, raised task errors, and changed temporary-file management. Criterion formulas were unchanged here. |
| `quick_analysis.QuickAnalysis.confusion_matrix` | 378 | Repair | Read the stored Ypred_test attribute instead of nonexistent Ytest_pred; inherits prettyConfMat changes. |
| `quick_analysis.QuickAnalysis.forwardElimMetricCalc` | 280 | Compatibility | Used current L1 logistic parameters; equal coefficients/intercepts were reproduced for old and new parameter forms in the current environment. |
| `quick_analysis.QuickAnalysis.genHighCorrs` | 212 | Behavior repair | Restricted candidate correlations to predictors, excluding the response. Correlation-based feature removal can change. |
| `quick_analysis.calcAICsLasso` | 17 | Mixed | Updated L1 parameters and flattened coefficients so one-feature models work. Criteria also depend on downstream scoring definitions; optional regularized-AIC remains unverified. |
| `quick_analysis.parallel_AIC_data_retriever` | 73 | Refactor | Kept the Feather read, predictor/response selection, and call parameters. Removed local variable clearing; intended numerical output was unchanged. |
| `sampling.latinHypercube1D` | 89 | Repair/compatibility | Used ndarray dtype rather than nonexistent dtypes, and replaced removed np.int aliases. Restored execution without an intended sampling-rule change. |
| `tmath.norm` | 388 | Review | Moved the existing p=0 dimension-count branch before division by zero; it still differs from conventional nonzero-count L0. Corrected the invalid raise statement. |
| `utils.dsn_getTable` | 16 | Regression | Disposed engines in finally and escaped passwords through URL.create, but incorrectly treated complete DSNs as hostnames. |
| `utils.dsn_saveTable` | 30 | Regression | Disposed engines in finally and escaped passwords through URL.create, but incorrectly treated complete DSNs as hostnames. |
| `utils.inferFeatureType` | 416 | Policy change | Used public dtype checks and excluded missing values from the unique-count cutoff. Boolean classification agreed; some numeric/category outcomes changed. |
| `utils.move_columns_to_end` | 330 | Verified repair | Treated a string as one column name instead of splitting it into characters. |
| `utils.print_message_with_time` | 131 | Side effect change | Created a log directory only when logging was requested. Message text was unchanged. |
| `utils.sample` | 496 | Mixed/review | Copied caller weights, validated probabilities, and isolated random state. Global-seed behavior changed; zero-weight boundary selection also changed. |

Unchanged callers can still inherit changed outputs. Examples include `RegressionSE` through `GenerateRegCov`, model and quick-analysis scores through `conf_mat_metrics`, ESS through spectral covariance, and neural predictions through training updates. The inventory above is a direct-code inventory, not a claim that all other outputs stayed unchanged.
