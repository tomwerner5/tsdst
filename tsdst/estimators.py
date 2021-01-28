from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import numpy as np

from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as glm_lr
### keeping in case using an older version of sklearn
from statsmodels.genmod.generalized_linear_model import GLM as glm_pois
from statsmodels.genmod.families.family import Poisson
###
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.utils import check_array
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from statsmodels.tools import add_constant
from timeit import default_timer as dt

from .distributions import (ap_logreg_lasso,
                                 posterior_logreg_lasso,
                                 likelihood_bernoulli,
                                 posterior_poisson_lasso,
                                 ap_poisson_lasso,
                                 posterior_poisson_lasso_od,
                                 ap_poisson_lasso_od,
                                 weibull_regression_post)
from .mcmc import adaptive_mcmc, rwm_with_lap, rwm, applyMCMC
from .tmath import mode_histogram, mode_kde
from .utils import print_time


class BayesLogRegClassifier(BaseEstimator, LinearClassifierMixin):
    '''
    A Logistic Regression Classifier that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    '''
    def __init__(self, C=None, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False, scorer=None):
        # TODO: Implement random_state for reporduceability
        '''
        The constructor for the BayesLogRegClassifier

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

        '''
        self.C = C
        self.start = start
        self.niter = niter
        if algo == 'rosenthal':
            self.algo = adaptive_mcmc
        elif algo == 'lap':
            self.algo = rwm_with_lap
        else:
            self.algo = rwm
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
        '''
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

        '''
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
        '''
        This method is intended to update the aggregated paramters with a new 
        summary as defined in new_param_summary. For example, if someone wanted
        to switch from coef_ representing the mean to coef_ representing 
        the median, they could use this function to do so.

        Parameters
        ----------
        new_param_summary : str
            The method of aggregating the MCMC chain parameters (either mean,
            median, mode_kde, mode_histogram, or last_value).

        Returns
        -------
        self
            Updates the coef_, intercept_, and extra_params_sum_ attributes.

        '''
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
        '''
        Fit the model

        Parameters
        ----------
        X : numpy array
            The feature (or design) matrix.
        y : numpy array
            The response variable.

        Returns
        -------
        self
            Updates internal attributes, such as coef_ and intercept_.

        '''
        t0 = dt()
        X = check_array(X, force_all_finite='allow-nan', estimator=self, copy=True)
        if not self.has_constant:
            X = add_constant(X, prepend=True)       
        
        if self.start is not None:
            pass
        else:
            if self.verbose:
                print_time("Initializing Coefficients...", t0, dt(), backsn=True)
            if self.initialize_weights == 'sklearn':
                C = self.C
                if C is None:
                    C = 1
                mod = glm_lr(C=C, solver='liblinear', penalty='l1', fit_intercept=False).fit(X, y)
                self.start = mod.coef_.reshape(-1, )
                if self.extra_params > 0:
                    self.start = np.concatenate((self.start, np.repeat(1, self.extra_params)))
            elif self.initialize_weights == 'ones':
                self.start = np.ones(shape=X.shape[1] + self.extra_params)
            elif self.initialize_weights == 'random':
                self.start = np.random.normal(size=(X.shape[1] + self.extra_params, ))
            else:
                self.start = np.zeros(shape=X.shape[1] + self.extra_params)

        if self.verbose:
            print_time("Beginning MCMC...", t0, dt(), backsn=True)
        
        postArgs = {
            'X': X,
            'Y': y,
            'l_scale': self.C*2
        }
        
        algo_res = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                             algo=self.algo, postArgs=postArgs,
                             algoOpts=self.algo_options, sd=self.retry_sd,
                             max_tries=self.retry_max_tries)
        
        self.mcmc_params = algo_res['parameters']
        self.prev_vals = algo_res['prev_vals']
        
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, self.param_summary,
                                                                                 self.extra_params)
        self.n_iter_ = self.niter
        self.classes_ = np.unique(y)
        
        if self.over_dispersion:
            self.dispersion_estimation_ = self.extra_params_sum_[-1]
        else:
            self.dispersion_estimation_ = None
        
        if self.verbose:
            print_time("Finished MCMC. Stored Coefficients...", t0, dt(), backsn=True)
            
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        scores = self.decision_function(X)
        # note: decision_function returns X.dot(Betas). This code, modified from sklearn,
        # computes the decision based on the sign of the log odds ratio (X.dot(betas) == log(p/(1-p))).
        # When p <= 0.5, decision is 0, when p > 0.5, decision is 1 (threshold of 0.5 is the default for sklearn). 
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        # returning the value in this way allows for someone to define their own classes,
        # for example, -1, 1 instead of 0, 1
        # Note: this is used in sklearn, but this code is not be robust enough for that,
        # so you need to define your classes as 0, 1
        return self.classes_[indices]
    
    def predict_proba(self, X):
        '''
        Generate predicted probability.

        Parameters
        ----------
        X : numpy array
            Feature (or design) matrix.

        Returns
        -------
        numpy array (float)
            The predicted probabilities.

        '''
        check_is_fitted(self, 'coef_')
        
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        
        return softmax(decision_2d, copy=False)
    
    def predict_log_proba(self, X):
        '''
        Generate predicted log probability.

        Parameters
        ----------
        X : numpy array
            Feature (or design) matrix.

        Returns
        -------
        numpy array (float)
            The predicted log probabilities.

        '''
        return np.log(self.predict_proba(X))
    
    def score(self, X, y, sample_weight=None):
        '''
        Scores the model using the scoring method passed, or, the default
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

        '''
        scorer = self.scorer or 'accuracy'
        scorer = get_scorer(scorer)
        
        return scorer(self, X, y, sample_weight=sample_weight)


class BayesPoissonRegressor(BaseEstimator, LinearClassifierMixin):
    '''
    A Poisson Regressor that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    '''
    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False, scorer='D2'):
        '''
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

        '''
        self.C = C
        self.start = start
        self.niter = niter
        if algo == 'rosenthal':
            self.algo = adaptive_mcmc
        elif algo == 'lap':
            self.algo = rwm_with_lap
        else:
            self.algo = rwm
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
        weights = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)
        y_mean = np.average(y, weights=weights)
        deviance_ = np.sum(weights * (2*(xlogy(y, y/y_pred) - y + y_pred)))
        null_deviance_ = np.sum(weights * (2*(xlogy(y, y/y_mean) - y + y_mean)))
        # pearson residual:  (raw residual)/(variance function)
        pearson_residuals_ = (y - y_pred)/np.sqrt(y_pred)
        pearson_chi2_ = np.sum(pearson_residuals_**2)
        model_d2_ = 1 - deviance_/null_deviance_
        # degrees of freedom of the model (all params (including intercept) minus 1)
        df_model_ = X.shape[1]
        # degrees of freedom of residuals ((n_obs - 1) - (nparms - 1)), or
        df_residuals_ = X.shape[0] - X.shape[1] - 1
        # total degrees of freedom
        df_total_ = df_residuals_ + df_model_
        # method of moments estimator for dispersion scale
        dispersion_scale_ = pearson_chi2_/df_residuals_
        dispersion_scale_sqrt_ = np.sqrt(dispersion_scale_)
        results = {'deviance_': deviance_,
                   'null_deviance_': null_deviance_,
                   'pearson_residuals_': pearson_residuals_,
                   'pearson_chi2_': pearson_chi2_,
                   'model_d2_': model_d2_,
                   'df_model_': df_model_,
                   'df_residuals_': df_residuals_,
                   'df_total_': df_total_,
                   'dispersion_scale_': dispersion_scale_,
                   'dispersion_scale_sqrt_': dispersion_scale_sqrt_}
        return results
        
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        '''
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

        '''
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
        '''
        This method is intended to update the aggregated paramters with a new 
        summary as defined in new_param_summary. For example, if someone wanted
        to switch from coef_ representing the mean to coef_ representing 
        the median, they could use this function to do so.

        Parameters
        ----------
        new_param_summary : str
            The method of aggregating the MCMC chain parameters (either mean,
            median, mode_kde, mode_histogram, or last_value).

        Returns
        -------
        self
            Updates the coef_, intercept_, and extra_params_sum_ attributes.

        '''
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        
        return self
    
    def fit(self, X, y):
        '''
        Fit the model

        Parameters
        ----------
        X : numpy array
            The feature (or design) matrix.
        y : numpy array
            The response variable.

        Returns
        -------
        self
            Updates internal attributes, such as coef_ and intercept_.

        '''
        t0 = dt()
        X = check_array(X, force_all_finite='allow-nan', estimator=self, copy=True)
        if not self.has_constant:
            X = add_constant(X, prepend=True)       
        
        if self.start is not None:
            pass
        else:
            if self.verbose:
                print_time("Initializing Coefficients...", t0, dt(), backsn=True)
            if self.initialize_weights == 'sklearn':
                C = self.C
                if C is None:
                    C = 1
                try:
                    from sklearn.linear_model import PoissonRegressor as glm_pois_sk
                    mod = glm_pois_sk(alpha=1/C, fit_intercept=False, max_iter=1000).fit(X, y)
                    self.start = mod.coef_.reshape(-1, )
                except ImportError:
                    print('Older sklearn, no PoissonRegressor. Using statsmodels instead')
                    mod = glm_pois(y, X, family=Poisson()).fit()
                    self.start = mod.params.reshape(-1, )
                if self.extra_params > 0:
                    self.start = np.concatenate((self.start, np.repeat(1, self.extra_params)))
            elif self.initialize_weights == 'ones':
                self.start = np.ones(shape=X.shape[1] + self.extra_params)
            elif self.initialize_weights == 'random':
                self.start = np.random.normal(size=(X.shape[1] + self.extra_params, ))
            else:
                self.start = np.zeros(shape=X.shape[1] + self.extra_params)

        if self.verbose:
            print_time("Beginning MCMC...", t0, dt(), backsn=True)
        
        postArgs = {
            'X': X,
            'Y': y,
            'l_scale': self.C*2
        }
        
        algo_res = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                             algo=self.algo, postArgs=postArgs,
                             algoOpts=self.algo_options, sd=self.retry_sd,
                             max_tries=self.retry_max_tries)
        
        self.mcmc_params = algo_res['parameters']
        self.prev_vals = algo_res['prev_vals']
        
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, self.param_summary,
                                                                                 self.extra_params)
        self.n_iter_ = self.niter
        
        # get model summaries
        if self.over_dispersion:
            self.dispersion_delta_ = self.extra_params_sum_[-1]
            self.dispersion_estimation_ = 1/(1 - self.dispersion_delta_)**2
        else:
            self.dispersion_delta_ = 0
            self.dispersion_estimation_ = None
        
        ddu = self._deviance_dispersion_update(X[:, 1:], y,
                                               sample_weight=None)
        self.deviance_ = ddu['deviance_']
        self.null_deviance_ = ddu['null_deviance_']
        self.pearson_residuals_ = ddu['pearson_residuals_']
        self.pearson_chi2_ = ddu['pearson_chi2_']
        self.model_d2_ = ddu['model_d2_']
        self.df_model_ = ddu['df_model_']
        self.df_residuals_ = ddu['df_residuals_']
        self.df_total_ = ddu['df_total_']
        self.dispersion_scale_ = ddu['dispersion_scale_']
        self.dispersion_scale_sqrt_ = ddu['dispersion_scale_sqrt_']
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=[np.float64, np.float32], ensure_2d=True,
                        allow_nd=False)
        mu = X @ self.coef_ + self.intercept_
        #return np.exp(mu)
        return np.exp(mu)*(1-self.dispersion_delta_)
    
    def score(self, X, y, sample_weight=None):
        if self.scorer == 'D2':
            ddu = self._deviance_dispersion_update(X, y,
                                                   sample_weight=None)
            score = ddu['model_d2_']
        else:
            score = self.scorer(X, y, sample_weight=sample_weight)
        
        return score

    
class BayesWeibullRegressor(BaseEstimator, LinearClassifierMixin):
    '''
    A Weibull Regressor that uses MCMC to evaluate the parameters.
    This objects inherits from sklearn\'s BaseEstimator and
    LinearClassifierMixin.
    '''
    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100,
                 initialize_weights='sklearn', param_summary='mean',
                 has_constant=False, verbose=True,
                 over_dispersion=False):
        '''
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

        '''
        self.C = C
        self.start = start
        self.niter = niter
        if algo == 'rosenthal':
            self.algo = adaptive_mcmc
        elif algo == 'lap':
            self.algo = rwm_with_lap
        else:
            self.algo = rwm
        self.algo_options = algo_options
        self.param_summary = param_summary
        self.initialize_weights = initialize_weights
        self.has_constant = has_constant
        self.verbose = verbose
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        self.lpost = weibull_regression_post
        self.extra_params = 0
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        '''
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

        '''
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
        weights = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)
        y_mean = np.average(y, weights=weights)
        deviance_ = np.sum(weights * (2*(xlogy(y, y/y_pred) - y + y_pred)))
        null_deviance_ = np.sum(weights * (2*(xlogy(y, y/y_mean) - y + y_mean)))
        # pearson residual:  (raw residual)/(variance function)
        # TODO: put correct weibull variance here
        pearson_residuals_ = (y - y_pred)/np.sqrt(y_pred)
        pearson_chi2_ = np.sum(pearson_residuals_**2)
        model_d2_ = 1 - deviance_/null_deviance_
        # degrees of freedom of the model (all params (including intercept) minus 1)
        df_model_ = X.shape[1]
        # degrees of freedom of residuals ((n_obs - 1) - (nparms - 1)), or
        df_residuals_ = X.shape[0] - X.shape[1] - 1
        # total degrees of freedom
        df_total_ = df_residuals_ + df_model_
        # method of moments estimator for dispersion scale
        dispersion_scale_ = pearson_chi2_/df_residuals_
        dispersion_scale_sqrt_ = np.sqrt(dispersion_scale_)
        results = {'deviance_': deviance_,
                   'null_deviance_': null_deviance_,
                   'pearson_residuals_': pearson_residuals_,
                   'pearson_chi2_': pearson_chi2_,
                   'model_d2_': model_d2_,
                   'df_model_': df_model_,
                   'df_residuals_': df_residuals_,
                   'df_total_': df_total_,
                   'dispersion_scale_': dispersion_scale_,
                   'dispersion_scale_sqrt_': dispersion_scale_sqrt_}
        return results
        
    def adjust_params_(self, new_param_summary):
        '''
        This method is intended to update the aggregated paramters with a new 
        summary as defined in new_param_summary. For example, if someone wanted
        to switch from coef_ representing the mean to coef_ representing 
        the median, they could use this function to do so.

        Parameters
        ----------
        new_param_summary : str
            The method of aggregating the MCMC chain parameters (either mean,
            median, mode_kde, mode_histogram, or last_value).

        Returns
        -------
        self
            Updates the coef_, intercept_, and extra_params_sum_ attributes.

        '''
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
        '''
        Fit the model

        Parameters
        ----------
        X : numpy array
            The feature (or design) matrix.
        y : numpy array
            The response variable.

        Returns
        -------
        self
            Updates internal attributes, such as coef_ and intercept_.

        '''
        t0 = dt()
        X = check_array(X, force_all_finite='allow-nan', estimator=self, copy=True)
        if not self.has_constant:
            X = add_constant(X, prepend=True)       
        
        if self.start is not None:
            pass
        else:
            if self.verbose:
                print_time("Initializing Coefficients...", t0, dt(), backsn=True)
            if self.initialize_weights == 'sklearn':
                C = self.C
                if C is None:
                    C = 1
                # TODO: update with Weibull Regression starting values
                ### If using sklearn version 0.23.2, can use this line instead
                #mod = glm_pois(alpha=1/C, fit_intercept=False, max_iter=1000).fit(X, y)
                #self.start = mod.coef_.reshape(-1, )
                ### else, use statsmodels
                mod = glm_pois(y, X, family=Poisson()).fit()
                self.start = mod.params.reshape(-1, )
                if self.extra_params > 0:
                    self.start = np.concatenate((self.start, np.repeat(1, self.extra_params)))
            elif self.initialize_weights == 'ones':
                self.start = np.ones(shape=X.shape[1] + self.extra_params)
            elif self.initialize_weights == 'random':
                self.start = np.random.normal(X.shape[1] + self.extra_params)
            else:
                self.start = np.zeros(shape=X.shape[1] + self.extra_params)

        if self.verbose:
            print_time("Beginning MCMC...", t0, dt(), backsn=True)
        
        postArgs = {
            'X': X,
            'Y': y,
            'l_scale': self.C*2
        }
        
        algo_res = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                             algo=self.algo, postArgs=postArgs,
                             algoOpts=self.algo_options, sd=self.retry_sd,
                             max_tries=self.retry_max_tries)
        
        self.mcmc_params = algo_res['parameters']
        self.prev_vals = algo_res['prev_vals']
        
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, self.param_summary,
                                                                                 self.extra_params)
        self.n_iter_ = self.niter
        
        #get model summaries
        weights = _check_sample_weight(None, X)
        y_pred = self.predict(X[:, 1:])
        y_mean = np.average(y, weights=weights)
        dev = np.sum(weights * (2*(xlogy(y, y/y_pred) - y + y_pred)))
        dev_null = np.sum(weights * (2*(xlogy(y, y/y_mean) - y + y_mean)))
        self.deviance_ = dev
        self.null_deviance_ = dev_null
        self.pearson_residuals_ = (y - y_pred)/np.sqrt(y_pred)
        self.pearson_chi2_ = np.sum(self.pearson_residuals_**2)
        self.model_d2_ = 1 - dev/dev_null
        self.df_model_ = X.shape[1] - 1
        self.df_residuals_ = X.shape[0] - X.shape[1]
        self.dispersion_scale_ = self.pearson_chi2_/self.df_residuals_
        self.dispersion_scale_sqrt_ = np.sqrt(self.dispersion_scale_)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=[np.float64, np.float32], ensure_2d=True,
                        allow_nd=False)
        mu = X @ self.coef_ + self.intercept_
        return np.exp(mu)
    
    def score(self, X, y, sample_weight=None, scorer='D2'):
        if scorer == 'D2':
            ddu = self._deviance_dispersion_update(X[:, 1:], y,
                                                   sample_weight=None)
            score = 1 - ddu['deviance_']/ddu['null_deviance_']
        else:
            score = scorer(X, y, sample_weight=sample_weight)
        
        return score
    

class LogReg(BaseEstimator, LinearClassifierMixin):
    '''
    A class for performing Logistic Regression using scipy.optimize.minimize.
    This is not meant to replace skleaarn's implementation, and in fact, it
    is mainly built on sklearn. This was mainly a test for the author to get a
    better understanding of both the internals of sklearn and
    Logistic Regression
    '''
    def __init__(self, x0=None, lamb=1, l_norm=1, method=None, jac=None,
                 hess=None, hessp=None, bounds=None, constraints=(), tol=None,
                 callback=None, options=None, has_constant=False):
        
        self.optimize_args = {'fun': likelihood_bernoulli,
                              'x0': x0,
                              'method': method,
                              'jac': jac,
                              'hess': hess,
                              'hessp': hessp,
                              'bounds': bounds,
                              'constraints': constraints,
                              'tol': tol,
                              'callback': callback,
                              'options': options
                             }
        self.lamb = lamb
        self.l_norm = l_norm
        
        
    def fit(self, X, y):
        '''
        Fit the model

        Parameters
        ----------
        X : numpy array
            The feature (or design) matrix.
        y : numpy array
            The response variable.

        Returns
        -------
        self
            Updates internal attributes, such as coef_ and intercept_.

        '''
        
        if not self.has_constant:
            X = add_constant(X, prepend=True)
        
        if self.optimize_args['x0'] is None:
            self.optimize_args['x0'] = np.zeros(shape=X.shape[1])
            #self.optimize_args['x0'] = np.random.normal(size=X.shape[1])
        
        args = (X, y, self.lamb, self.l_norm)
        
        self.optimize_args.update({'args': args})
        self.optim_results = minimize(**self.optimize_args)
        
        self.intercept_ = self.optim_results.x[0].reshape(1, )
        self.coef_ = self.optim_results.x[1:].reshape(1, -1)
        self.n_iter_ = self.optim_results.nit
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        scores = self.decision_function(X)
        # note: decision_function returns X.dot(Betas). This code, modified from sklearn,
        # computes the decision based on the sign of the log odds ratio (X.dot(betas) == log(p/(1-p))).
        # When p <= 0.5, decision is 0, when p > 0.5, decision is 1 (threshold of 0.5 is the default for sklearn). 
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        # returning the value in this way allows for someone to define their own classes,
        # for example, -1, 1 instead of 0, 1
        # Note: this is used in sklearn, but this code is not be robust enough for that,
        # so you need to define your classes as 0, 1
        return self.classes_[indices]
    
    def predict_proba(self, X):
        '''
        Generate predicted probability.

        Parameters
        ----------
        X : numpy array
            Feature (or design) matrix.

        Returns
        -------
        numpy array (float)
            The predicted probabilities.

        '''
        #check_is_fitted(self)
        
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        
        return softmax(decision_2d, copy=False)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def score(self, X, y, sample_weight=None):
        #TODO: Test default scoring
        scoring = self.scoring or 'accuracy'
        scoring = get_scorer(scoring)
        
        return scoring(self, X, y, sample_weight=sample_weight)