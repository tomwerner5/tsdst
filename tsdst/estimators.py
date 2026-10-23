import numpy as np

from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression as glm_lr
### keeping in case using an older version of sklearn
###
from sklearn.linear_model import PoissonRegressor
from scipy.special import expit, gamma
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_is_fitted, validate_data

from .distributions import (ap_logreg_lasso,
                                 posterior_logreg_lasso,
                                 glm_likelihood_bernoulli,
                                 posterior_poisson_lasso,
                                 ap_poisson_lasso,
                                 posterior_poisson_lasso_od,
                                 ap_poisson_lasso_od,
                                 weibull_regression_post)
from .mcmc import adaptive_mcmc, rwm_with_lap, rwm, applyMCMC
from .tmath import mode_histogram, mode_kde


def _fit_bayesian(estimator, X, y, family):
    """Validate data and fit a Bayesian model with the selected sampler."""
    original_X = X
    X, y = validate_data(estimator, X, y, dtype=float)
    if estimator.niter < 2:
        raise ValueError('niter must be at least 2')
    if family == 'binomial':
        estimator.classes_, target = np.unique(y, return_inverse=True)
        if len(estimator.classes_) != 2:
            raise ValueError('The classifier requires exactly two classes')
        posterior, extra = ((ap_logreg_lasso, 1) if estimator.C is None
                            else (posterior_logreg_lasso, 0))
    elif family == 'poisson':
        if np.any(y < 0):
            raise ValueError('Poisson responses must be nonnegative')
        target = y
        if estimator.C is None:
            posterior, extra = ((ap_poisson_lasso_od, 2) if estimator.over_dispersion else (ap_poisson_lasso, 1))
        else:
            posterior, extra = ((posterior_poisson_lasso_od, 1) if estimator.over_dispersion else (posterior_poisson_lasso, 0))
    else:
        if np.any(y <= 0):
            raise ValueError('Weibull responses must be positive')
        target, posterior, extra = y, weibull_regression_post, 1
    estimator.extra_params = extra
    design = X if estimator.has_constant else np.column_stack([np.ones(len(X)), X])
    size = design.shape[1] + extra
    if estimator.start is not None:
        initial = np.asarray(estimator.start, dtype=float).copy()
    elif estimator.initialize_weights == 'ones':
        initial = np.ones(size)
    elif estimator.initialize_weights == 'random':
        initial = np.random.normal(size=size)
    elif estimator.initialize_weights == 'sklearn' and family != 'weibull':
        strength = 1 if estimator.C is None else estimator.C
        model = (glm_lr(C=strength, solver='liblinear', l1_ratio=1, fit_intercept=False)
                 if family == 'binomial' else PoissonRegressor(alpha=1/strength, fit_intercept=False))
        initial = np.r_[model.fit(design, target).coef_.ravel(), np.zeros(extra)]
    else:
        initial = np.zeros(size)
        if family == 'weibull':
            initial[-1] = np.log(np.mean(target))
    if initial.shape != (size,):
        raise ValueError('The starting parameter count does not match the model')
    estimator.start_ = initial
    algorithms = {'rosenthal': adaptive_mcmc, 'lap': rwm_with_lap, 'rwm': rwm}
    algorithm = estimator.algo if callable(estimator.algo) else algorithms[estimator.algo]
    def log_posterior(parameters, **arguments):
        # Reject proposals outside numeric support without overflow warnings.
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            value = posterior(parameters, **arguments)
        return value if np.isfinite(value) else -np.inf

    result = applyMCMC(initial, estimator.niter, log_posterior, algorithm,
                      algoOpts=dict(estimator.algo_options or {}),
                      postArgs={'X': design, 'Y': target, 'l_scale': 1 if estimator.C is None else 2*estimator.C},
                      sd=estimator.retry_sd, max_tries=estimator.retry_max_tries)
    estimator.mcmc_params, estimator.prev_vals = result['parameters'], result['prev_vals']
    estimator.coef_, estimator.intercept_, estimator.extra_params_sum_ = estimator._create_coefs(
        estimator.mcmc_params, estimator.param_summary, extra)
    estimator.n_iter_ = len(estimator.mcmc_params)
    if family == 'binomial':
        estimator.dispersion_estimation_ = None
    else:
        estimator.dispersion_delta_ = (float(estimator.extra_params_sum_[-1])
                                      if family == 'poisson' and estimator.over_dispersion else 0)
        estimator.dispersion_estimation_ = (1/(1-estimator.dispersion_delta_)**2
                                            if estimator.over_dispersion else None)
        if family == 'weibull':
            estimator.scale_ = float(np.exp(estimator.extra_params_sum_[-1]))
        for name, value in estimator._deviance_dispersion_update(original_X, y).items():
            setattr(estimator, name, value)
    return estimator


class BayesLogRegClassifier(ClassifierMixin, BaseEstimator):
    """A Logistic Regression Classifier that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    """

    def __init__(self, C=None, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False, scorer=None):
        # TODO: Implement random_state for reproduceability
        """The constructor for the BayesLogRegClassifier

        Parameters
        ----------
        C : float, optional
            The value for the inverse L1 penalty, or, the inverse
            regularization strength. If None, the MCMC process
            looks for the optimal penalty. The default is None.
            
            This value gets converted into the scale parameter for a laplace
            distribution (scale = 2*C).
        start : numpy array (float), optional
            The starting values for the MCMC. If None, the MLE estimate is used
            (solved with sklearn). The default is None.
        niter : int, optional
            The number of MCMC samples to draw. The default is 10000.
        algo : str, optional
            The MCMC (Metropolis) algorithm to use. The default is 'rosenthal',
            which is a method that tunes the covariance matrix after each
            iteration. Other options include 'rwm', which is a simple random
            metropolis walk with a fixed covariance matrix, and 'lap' which is
            another adaptive method that tunes the covariance matrix every K
            iterations.
        algo_options : dict, optional
            The options to be passed to the MCMC algorithm. Include as a
            dictionary. The default is None.
        retry_sd : float, optional
            The MCMC alorithms use a Cholesky decomposition on the covariance
            matrix. In case the decomposition fails, the algorithms will 
            attempt to jitter the covaraince matrix to help it be positive 
            definite. This value determines the strength of the jittering and
            is drawn directly from a normal distribution with zero mean and
            retry_sd standard deviation. The default is 0.02.
        retry_max_tries : int, optional
            Number of attempts to correct the cholesky decomposition if it
            fails. The default is 100.
        initialize_weights : str, optional
            Determines the method of initializing the starting model parameter
            values (if start is None). Options are 'sklearn', 'ones', 'random',
            or 'zeros'. The default is 'sklearn'.
        param_summary : str, optional
            The method used in making the final parameter summaries. Options 
            are 'mean', 'median', 'mode_kde', 'mode_histogram', or
            'final_sample'. The default is 'mean'.
        has_constant : bool, optional
            Whether or not the data provided already has a column of ones
            as the first column in the dataset for the intercept of the model.
            If not, one is created. The default is False.
        verbose : bool, optional
            If True, a progress bar, along with timestamps, is provided.
            The default is True.
        over_dispersion : bool, optional
            Whether or not to account for overdispersion in the model.
            The default is False.
        scorer : function or str
            The function or string to use for the default scoring method.
            Otherwise, pass None for accuracy. The default is None.

        Returns
        -------
        None.

        """
        self.C = C
        self.start = start
        self.niter = niter
        self.algo = algo
        self.algo_options = algo_options
        self.param_summary = param_summary
        self.initialize_weights = initialize_weights
        self.has_constant = has_constant
        self.verbose = verbose
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        self.scorer = scorer
        self.over_dispersion = over_dispersion
        if C is None:
            # TODO: Implement over-dispersion
            #if over_dispersion:
            #    self.lpost = ap_logreg_lasso_od
            #    self.extra_params = 2
            #else:
            self.lpost = ap_logreg_lasso
            self.extra_params = 1
        else:
            #if over_dispersion:
            #    self.lpost = posterior_logreg_lasso_od
            #    self.extra_params = 1
            #else:
            self.lpost = posterior_logreg_lasso
            self.extra_params = 0
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        """Creates Coefficients for MCMC model by aggregating the MCMC samples,
        using mean, median, mode, or last sampled value.

        Parameters
        ----------
        mcmc_params : numpy array (float)
            MCMC chain, one column for each parameter.
        param_summary : str
            The method of aggregation (either mean, median, mode_kde,
            mode_histogram, or last_value.
        extra_params : int
            The number of extra params used in the MCMC process (for example,
        parameters for priors or overdispersion).

        Returns
        -------
        coefs : numpy array (float)
            The coefficients of the model (aggregated from the MCMC chain).
        intercept : float
            The intercept of the model (aggregated from the MCMC chain).
        extra_parm : numpy array (float)
            Any extra parameters that were solved with the model (aggregated
            from the MCMC chain).

        """
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([mode_histogram(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        elif param_summary == 'mode_kde':
            sum_parms = np.array([mode_kde(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        else:
            sum_parms = mcmc_params[-1, :]
        
        if extra_params > 0:
            extra_parm = sum_parms[-extra_params:]
            coefs = sum_parms[1:-extra_params].reshape(1, -1)
        else:
            extra_parm = None
            coefs = sum_parms[1:].reshape(1, -1)
        intercept = np.array(sum_parms[0])
        
        return coefs, intercept, extra_parm
        
    def adjust_params_(self, new_param_summary):
        """This method is intended to update the aggregated parameters with a new
        summary as defined in new_param_summary. For example, if someone wanted
        to switch from `coef_` representing the mean to `coef_` representing
        the median, they could use this function to do so.

        Parameters
        ----------
        new_param_summary : str
            The method of aggregating the MCMC chain parameters (either mean,
            median, mode_kde, mode_histogram, or last_value).

        Returns
        -------
        self
            Updates the `coef_`, `intercept_`, and `extra_params_sum_` attributes.

        """
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
        """Fit the model with finite numeric features and a compatible response."""
        return _fit_bayesian(self, X, y, 'binomial')
    
    def decision_function(self, X):
        """Return the linear predictor for each row."""
        check_is_fitted(self, 'coef_')
        X = validate_data(self, X, reset=False)
        linear = X[:, 1:] if self.has_constant else X
        return (linear @ self.coef_.ravel() + self.intercept_).ravel()

    def predict(self, X):
        """Return the class with probability at least one half."""
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]
    
    def predict_proba(self, X):
        """Return binary probabilities with the logistic link."""
        positive = expit(self.decision_function(X))
        return np.column_stack([1-positive, positive])
    
    def predict_log_proba(self, X):
        """Generate predicted log probability.

        Parameters
        ----------
        X : numpy array
            Feature (or design) matrix.

        Returns
        -------
        numpy array (float)
            The predicted log probabilities.

        """
        return np.log(self.predict_proba(X))
    
    def score(self, X, y, sample_weight=None):
        """Scores the model using the scoring method passed, or, the default
        scorer. In this case, the default scorer is accuracy.

        Parameters
        ----------
        X : numpy array or pandas dataframe
            The design or feature matrix.
        y : numpy array or pandas series
            The target or response variable.
        sample_weight : numpy array, optional
            An array containing the weights for each sample.
            The default is None.

        Returns
        -------
        score
            The result of the scoring function.

        """
        scorer = self.scorer or 'accuracy'
        scorer = get_scorer(scorer)
        
        return scorer(self, X, y, sample_weight=sample_weight)


class BayesPoissonRegressor(RegressorMixin, BaseEstimator):
    """
    A Poisson Regressor that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    """

    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False, scorer='D2'):
        """
        The constructor for the BayesPoissonClassifier

        Parameters
        ----------
        C : float, optional
            The value for the inverse L1 penalty, or, the inverse
            regularization strength. If None, the MCMC process
            looks for the optimal penalty. The default is None.
            
            This value is gets converted into the scale parameter for a laplace
            distribution (scale = 2*C).
        start : numpy array (float), optional
            The starting values for the MCMC. If None, the MLE estimate is used
            (solved with sklearn). The default is None.
        niter : int, optional
            The number of MCMC samples to draw. The default is 10000.
        algo : str, optional
            The MCMC (Metropolis) algorithm to use. The default is 'rosenthal', which is 
            a method that tunes the covariance matrix after each iteration.
            Other options include 'rwm', which is a simple random metropolis
            walk with a fixed covariance matrix, and 'lap' which is another 
            adaptive method that tunes the covariance matrix every K
            iterations.
        algo_options : dict, optional
            The options to be passed to the MCMC algorithm. Include as a
            dictionary. The default is None.
        retry_sd : float, optional
            The MCMC alorithms use a Cholesky decomposition on the covariance
            matrix. In case the decomposition fails, the algorithms will 
            attempt to jitter the covaraince matrix to help it be positive 
            definite. This value determines the strength of the jittering and
            is drawn directly from a normal distribution with zero mean and
            retry_sd standard deviation. The default is 0.02.
        retry_max_tries : int, optional
            Number of attempts to correct the cholesky decomposition if it
            fails. The default is 100.
        initialize_weights : str, optional
            Determines the method of initializing the starting model parameter
            values (if start is None). Options are 'sklearn', 'ones', 'random',
            or 'zeros'. The default is 'sklearn'.
        param_summary : str, optional
            The method used in making the final parameter summaries. Options 
            are 'mean', 'median', 'mode_kde', 'mode_histogram', or
            'final_sample'. The default is 'mean'.
        has_constant : bool, optional
            Whether or not the data provided already has a column of ones
            as the first column in the dataset for the intercept of the model.
            If not, one is created. The default is False.
        verbose : bool, optional
            If True, a progress bar, along with timestamps, is provided.
            The default is True.
        over_dispersion : bool, optional
            Whether or not to account for overdispersion in the model.
            The default is False.

        Returns
        -------
        None.

        """
        self.C = C
        self.start = start
        self.niter = niter
        self.algo = algo
        self.algo_options = algo_options
        self.param_summary = param_summary
        self.initialize_weights = initialize_weights
        self.has_constant = has_constant
        self.verbose = verbose
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        self.over_dispersion = over_dispersion
        self.scorer = scorer
        if C is None:
            if over_dispersion:
                self.lpost = ap_poisson_lasso_od
                self.extra_params = 2
            else:
                self.lpost = ap_poisson_lasso
                self.extra_params = 1
        else:
            if over_dispersion:
                print("using OD")
                self.lpost = posterior_poisson_lasso_od
                self.extra_params = 1
            else:
                self.lpost = posterior_poisson_lasso
                self.extra_params = 0
    
    def _deviance_dispersion_update(self, X, y, sample_weight=None):
        """Return positive-response diagnostics based on Poisson deviance."""
        y = np.asarray(y).ravel()
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != y.shape or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Supply one nonnegative sample weight per response')
        predicted = self.predict(X)
        mean = np.average(y, weights=weights)
        deviance = 2*np.sum(weights*(xlogy(y, y/predicted)-y+predicted))
        null = 2*np.sum(weights*(xlogy(y, y/mean)-y+mean))
        residuals = (y-predicted)/np.sqrt(predicted)
        chi2 = np.sum(weights*residuals**2)
        degrees = len(y)-self.coef_.size-1
        dispersion = chi2/degrees
        return dict(deviance_=deviance, null_deviance_=null, pearson_residuals_=residuals,
                    pearson_chi2_=chi2, model_d2_=1-deviance/null if null else float(deviance == 0),
                    df_model_=self.coef_.size, df_residuals_=degrees, df_total_=len(y)-1,
                    dispersion_scale_=dispersion, dispersion_scale_sqrt_=np.sqrt(dispersion))
        
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        """
        Creates Coefficients for MCMC model by aggregating the MCMC samples,
        using mean, median, mode, or last sampled value.

        Parameters
        ----------
        mcmc_params : numpy array (float)
            MCMC chain, one column for each parameter.
        param_summary : str
            The method of aggregation (either mean, median, mode_kde,
            mode_histogram, or last_value.
        extra_params : int
            The number of extra params used in the MCMC process (for example,
        parameters for priors or overdispersion).

        Returns
        -------
        coefs : numpy array (float)
            The coefficients of the model (aggregated from the MCMC chain).
        intercept : float
            The intercept of the model (aggregated from the MCMC chain).
        extra_parm : numpy array (float)
            Any extra parameters that were solved with the model (aggregated
            from the MCMC chain).

        """
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([mode_histogram(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        elif param_summary == 'mode_kde':
            sum_parms = np.array([mode_kde(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        else:
            sum_parms = mcmc_params[-1, :]
        
        if extra_params > 0:
            extra_parm = sum_parms[-extra_params:]
            coefs = sum_parms[1:-extra_params].reshape(-1, )
        else:
            extra_parm = None
            coefs = sum_parms[1:].reshape(-1, )
        intercept = np.array(sum_parms[0])
        
        return coefs, intercept, extra_parm
        
    def adjust_params_(self, new_param_summary):
        """This method is intended to update the aggregated parameters with a new
        summary as defined in new_param_summary. For example, if someone wanted
        to switch from `coef_` representing the mean to `coef_` representing
        the median, they could use this function to do so.

        Parameters
        ----------
        new_param_summary : str
            The method of aggregating the MCMC chain parameters (either mean,
            median, mode_kde, mode_histogram, or last_value).

        Returns
        -------
        self
            Updates the `coef_`, `intercept_`, and `extra_params_sum_` attributes.

        """
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        
        return self
    
    def fit(self, X, y):
        """Fit the model with finite numeric features and a compatible response."""
        return _fit_bayesian(self, X, y, 'poisson')
    
    def predict(self, X):
        """Return positive count predictions."""
        check_is_fitted(self, 'coef_')
        X = validate_data(self, X, reset=False)
        linear = X[:, 1:] if self.has_constant else X
        return np.exp(linear @ self.coef_ + self.intercept_)*(1-self.dispersion_delta_)
    
    def score(self, X, y, sample_weight=None):
        """Return D2 or use the configured scikit-learn scorer."""
        if self.scorer == 'D2':
            return self._deviance_dispersion_update(X, y, sample_weight)['model_d2_']
        scorer = get_scorer(self.scorer) if isinstance(self.scorer, str) else self.scorer
        return scorer(self, X, y, sample_weight=sample_weight)

    
class BayesWeibullRegressor(RegressorMixin, BaseEstimator):
    """
    A Weibull Regressor that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    """

    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False):
        """
        The constructor for the BayesWeibullClassifier

        Parameters
        ----------
        C : float, optional
            The value for the inverse L1 penalty, or, the inverse
            regularization strength. If None, the MCMC process
            looks for the optimal penalty. The default is None.
            
            This value is gets converted into the scale parameter for a laplace
            distribution (scale = 2*C).
        start : numpy array (float), optional
            The starting values for the MCMC. If None, the MLE estimate is used
            (solved with sklearn). The default is None.
        niter : int, optional
            The number of MCMC samples to draw. The default is 10000.
        algo : str, optional
            The MCMC (Metropolis) algorithm to use. The default is 'rosenthal', which is 
            a method that tunes the covariance matrix after each iteration.
            Other options include 'rwm', which is a simple random metropolis
            walk with a fixed covariance matrix, and 'lap' which is another 
            adaptive method that tunes the covariance matrix every K
            iterations.
        algo_options : dict, optional
            The options to be passed to the MCMC algorithm. Include as a
            dictionary. The default is None.
        retry_sd : float, optional
            The MCMC alorithms use a Cholesky decomposition on the covariance
            matrix. In case the decomposition fails, the algorithms will 
            attempt to jitter the covaraince matrix to help it be positive 
            definite. This value determines the strength of the jittering and
            is drawn directly from a normal distribution with zero mean and
            retry_sd standard deviation. The default is 0.02.
        retry_max_tries : int, optional
            Number of attempts to correct the cholesky decomposition if it
            fails. The default is 100.
        initialize_weights : str, optional
            Determines the method of initializing the starting model parameter
            values (if start is None). Options are 'sklearn', 'ones', 'random',
            or 'zeros'. The default is 'sklearn'.
        param_summary : str, optional
            The method used in making the final parameter summaries. Options 
            are 'mean', 'median', 'mode_kde', 'mode_histogram', or
            'final_sample'. The default is 'mean'.
        has_constant : bool, optional
            Whether or not the data provided already has a column of ones
            as the first column in the dataset for the intercept of the model.
            If not, one is created. The default is False.
        verbose : bool, optional
            If True, a progress bar, along with timestamps, is provided.
            The default is True.
        over_dispersion : bool, optional
            ---CURRENTLY NOT IMPLEMENTED---
            Whether or not to account for overdispersion in the model.
            The default is False.

        Returns
        -------
        None.

        """
        self.C = C
        self.start = start
        self.niter = niter
        self.algo = algo
        self.algo_options = algo_options
        self.param_summary = param_summary
        self.initialize_weights = initialize_weights
        self.has_constant = has_constant
        self.verbose = verbose
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        self.over_dispersion = over_dispersion
        self.lpost = weibull_regression_post
        self.extra_params = 1
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        """
        Creates Coefficients for MCMC model by aggregating the MCMC samples,
        using mean, median, mode, or last sampled value.

        Parameters
        ----------
        mcmc_params : numpy array (float)
            MCMC chain, one column for each parameter.
        param_summary : str
            The method of aggregation (either mean, median, mode_kde,
            mode_histogram, or last_value.
        extra_params : int
            The number of extra params used in the MCMC process (for example,
        parameters for priors or overdispersion).

        Returns
        -------
        coefs : numpy array (float)
            The coefficients of the model (aggregated from the MCMC chain).
        intercept : float
            The intercept of the model (aggregated from the MCMC chain).
        extra_parm : numpy array (float)
            Any extra parameters that were solved with the model (aggregated
            from the MCMC chain).

        """
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([mode_histogram(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        elif param_summary == 'mode_kde':
            sum_parms = np.array([mode_kde(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
        else:
            sum_parms = mcmc_params[-1, :]
        
        if extra_params > 0:
            extra_parm = sum_parms[-extra_params:]
            coefs = sum_parms[1:-extra_params].reshape(-1, )
        else:
            extra_parm = None
            coefs = sum_parms[1:].reshape(-1, )
        intercept = np.array(sum_parms[0])
        
        return coefs, intercept, extra_parm
    
    def _deviance_dispersion_update(self, X, y, sample_weight=None):
        """Return positive-response diagnostics based on Poisson deviance."""
        y = np.asarray(y).ravel()
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != y.shape or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Supply one nonnegative sample weight per response')
        predicted = self.predict(X)
        mean = np.average(y, weights=weights)
        deviance = 2*np.sum(weights*(xlogy(y, y/predicted)-y+predicted))
        null = 2*np.sum(weights*(xlogy(y, y/mean)-y+mean))
        residuals = (y-predicted)/np.sqrt(predicted)
        chi2 = np.sum(weights*residuals**2)
        degrees = len(y)-self.coef_.size-1
        dispersion = chi2/degrees
        return dict(deviance_=deviance, null_deviance_=null, pearson_residuals_=residuals,
                    pearson_chi2_=chi2, model_d2_=1-deviance/null if null else float(deviance == 0),
                    df_model_=self.coef_.size, df_residuals_=degrees, df_total_=len(y)-1,
                    dispersion_scale_=dispersion, dispersion_scale_sqrt_=np.sqrt(dispersion))
        
    def adjust_params_(self, new_param_summary):
        """Update coefficients and scale with a new posterior summary."""
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(
            self.mcmc_params, new_param_summary, self.extra_params)
        self.scale_ = float(np.exp(self.extra_params_sum_[-1]))
        return self
    
    def fit(self, X, y):
        """Fit the model with finite numeric features and a compatible response."""
        return _fit_bayesian(self, X, y, 'weibull')
    
    def predict(self, X):
        """Return the Weibull mean from the fitted shape and scale."""
        check_is_fitted(self, 'coef_')
        X = validate_data(self, X, reset=False)
        linear = X[:, 1:] if self.has_constant else X
        shape = np.exp(linear @ self.coef_ + self.intercept_)
        return self.scale_*gamma(1+1/shape)
    
    def score(self, X, y, sample_weight=None, scorer='r2'):
        """Return R2, or D2 for the positive-response diagnostic."""
        if scorer == 'D2':
            return self._deviance_dispersion_update(X, y, sample_weight)['model_d2_']
        scorer = get_scorer(scorer) if isinstance(scorer, str) else scorer
        return scorer(self, X, y, sample_weight=sample_weight)
    

class LogReg(ClassifierMixin, BaseEstimator):
    """
    A class for performing Logistic Regression using scipy.optimize.minimize.
    This is not meant to replace skleaarn's implementation, and in fact, it
    is mainly built on sklearn. This was mainly a test for the author to get a
    better understanding of both the internals of sklearn and
    Logistic Regression
    """

    def __init__(self, x0=None, lamb=1, l_norm=1, method=None, jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, has_constant=False):
        self.x0, self.lamb, self.l_norm = x0, lamb, l_norm
        self.method, self.jac, self.hess, self.hessp = method, jac, hess, hessp
        self.bounds, self.constraints, self.tol = bounds, constraints, tol
        self.callback, self.options, self.has_constant = callback, options, has_constant
        
        
    def fit(self, X, y):
        """Minimize the penalized binary logistic negative log likelihood."""
        X, y = validate_data(self, X, y, dtype=float)
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError('The classifier requires exactly two classes')
        design = X if self.has_constant else np.column_stack([np.ones(len(X)), X])
        initial = np.zeros(design.shape[1]) if self.x0 is None else np.asarray(self.x0)
        self.optim_results = minimize(glm_likelihood_bernoulli, initial,
            args=(design, y, self.lamb, self.l_norm), method=self.method, jac=self.jac,
            hess=self.hess, hessp=self.hessp, bounds=self.bounds, constraints=self.constraints,
            tol=self.tol, callback=self.callback, options=self.options)
        self.intercept_, self.coef_ = self.optim_results.x[:1], self.optim_results.x[1:][None, :]
        self.n_iter_ = self.optim_results.nit
        return self
    
    def decision_function(self, X):
        """Return the linear predictor for each row."""
        check_is_fitted(self, 'coef_')
        X = validate_data(self, X, reset=False)
        linear = X[:, 1:] if self.has_constant else X
        return (linear @ self.coef_.ravel() + self.intercept_).ravel()

    def predict(self, X):
        """Return the class with probability at least one half."""
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]
    
    def predict_proba(self, X):
        """Return binary probabilities with the logistic link."""
        positive = expit(self.decision_function(X))
        return np.column_stack([1-positive, positive])
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def score(self, X, y, sample_weight=None):
        """Return classification accuracy."""
        return ClassifierMixin.score(self, X, y, sample_weight=sample_weight)
