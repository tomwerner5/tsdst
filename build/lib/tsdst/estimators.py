from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import numpy as np

from scipy.special import xlogy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as glm_lr#, PoissonRegressor as glm_pois
### remove once sklearn gets updated
from statsmodels.genmod.generalized_linear_model import GLM as glm_pois
from statsmodels.genmod.families.family import Poisson
###
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from statsmodels.tools import add_constant
from timeit import default_timer as dt

from tsdst.distributions import (adaptive_posterior_logreg_lasso, posterior_logreg_lasso,
                                 negloglike_logreg, poisson_regression, adap_poisson_regression,
                                 poisson_regression_od, adap_poisson_regression_od, weibull_regression)
from tsdst.mcmc import adaptive_mcmc, rwm_with_lap, rwm, applyMCMC
from tsdst.tmath import histogram_mode, mode_kde
from tsdst.utils import print_time


class BayesLogRegClassifier(BaseEstimator, LinearClassifierMixin):
    def __init__(self, C=None, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100, initialize_weights='sklearn',
                 param_summary='mean', has_constant=False, verbose=True,
                 keep_nzc_only=True, over_dispersion=False):
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
        self.keep_nzc_only = keep_nzc_only
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        if C is None:
            if over_dispersion:
                self.lpost = adaptive_posterior_logreg_lasso_od
                self.extra_params = 2
            else:
                self.lpost = adaptive_posterior_logreg_lasso
                self.extra_params = 1
        else:
            if over_dispersion:
                self.lpost = posterior_logreg_lasso_od
                self.extra_params = 1
            else:
                self.lpost = posterior_logreg_lasso
                self.extra_params = 0
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([histogram_mode(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
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
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
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
                self.start = np.random.normal(X.shape[1] + self.extra_params)
            else:
                self.start = np.zeros(shape=X.shape[1] + self.extra_params)

        if self.verbose:
            print_time("Beginning MCMC...", t0, dt(), backsn=True)
        
        postArgs = {
            'X': X,
            'Y': y,
            'l_scale': self.C
        }
        
        self.mcmc_params, self.prev_vals = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                                                     algo=self.algo, postArgs=postArgs,
                                                     algoOpts=self.algo_options, sd=self.retry_sd,
                                                     max_tries=self.retry_max_tries)
        
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, self.param_summary,
                                                                                 self.extra_params)
        self.n_iter_ = self.niter
        self.classes_ = np.unique(y)
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
        check_is_fitted(self, 'coef_')
        
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        
        return softmax(decision_2d, copy=False)
    
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def score(self, X, y, sample_weight=None):
        scoring = self.scoring or 'accuracy'
        scoring = get_scorer(scoring)
        
        return scoring(self, X, y, sample_weight=sample_weight)


class BayesPoissonRegressor(BaseEstimator, LinearClassifierMixin):
    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100, initialize_weights='sklearn',
                 param_summary='mean', has_constant=False, verbose=True,
                 keep_nzc_only=True, over_dispersion=False):
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
        self.keep_nzc_only = keep_nzc_only
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        if C is None:
            if over_dispersion:
                self.lpost = adap_poisson_regression_od
                self.extra_params = 2
            else:
                self.lpost = adap_poisson_regression
                self.extra_params = 1
        else:
            if over_dispersion:
                self.lpost = poisson_regression_od
                self.extra_params = 1
            else:
                self.lpost = poisson_regression
                self.extra_params = 0
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([histogram_mode(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
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
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
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
            'l_scale': self.C
        }
        
        self.mcmc_params, self.prev_vals = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                                                     algo=self.algo, postArgs=postArgs,
                                                     algoOpts=self.algo_options, sd=self.retry_sd,
                                                     max_tries=self.retry_max_tries)
        
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
            weights = _check_sample_weight(sample_weight, X)
            y_pred = self.predict(X)
            y_mean = np.average(y, weights=weights)
            dev = np.sum(weights * (2*(xlogy(y, y/y_pred) - y + y_pred)))
            dev_null = np.sum(weights * (2*(xlogy(y, y/y_mean) - y + y_mean)))
            self.score_deviance_ = dev
            self.score_null_deviance_ = dev_null
            self.score_pearson_residuals_ = (y - y_pred)/np.sqrt(y_pred)
            self.score_pearson_chi2_ = np.sum(self.score_pearson_residuals_**2)
            
            score = 1 - dev/dev_null
        else:
            score = scorer(X, y, sample_weight=sample_weight)
        
        return score

    
class BayesWeibullRegressor(BaseEstimator, LinearClassifierMixin):
    def __init__(self, C=1, start=None, niter=10000, algo='rosenthal',
                 algo_options=None, retry_sd=0.02, retry_max_tries=100, initialize_weights='sklearn',
                 param_summary='mean', has_constant=False, verbose=True,
                 keep_nzc_only=True, over_dispersion=False):
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
        self.keep_nzc_only = keep_nzc_only
        self.retry_sd = retry_sd
        self.retry_max_tries = retry_max_tries
        self.lpost = weibull_regression
        self.extra_params = 0
            
    def _create_coefs(self, mcmc_params, param_summary, extra_params):
        sum_parms = None
        if param_summary == 'mean':
            sum_parms = mcmc_params.mean(axis=0)
        elif param_summary == 'median':
            sum_parms = np.median(mcmc_params, axis=0)
        elif param_summary == 'mode_histogram':
            sum_parms = np.array([histogram_mode(mcmc_params[:, i]) for i in range(mcmc_params.shape[1])])
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
        self.coef_, self.intercept_, self.extra_params_sum_ = self._create_coefs(self.mcmc_params, new_param_summary,
                                                                                 self.extra_params)
        return self
    
    def fit(self, X, y):
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
            'l_scale': self.C
        }
        
        self.mcmc_params, self.prev_vals = applyMCMC(st=self.start, ni=self.niter, lp=self.lpost,
                                                     algo=self.algo, postArgs=postArgs,
                                                     algoOpts=self.algo_options, sd=self.retry_sd,
                                                     max_tries=self.retry_max_tries)
        
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
            weights = _check_sample_weight(sample_weight, X)
            y_pred = self.predict(X)
            y_mean = np.average(y, weights=weights)
            dev = np.sum(weights * (2*(xlogy(y, y/y_pred) - y + y_pred)))
            dev_null = np.sum(weights * (2*(xlogy(y, y/y_mean) - y + y_mean)))
            self.score_deviance_ = dev
            self.score_null_deviance_ = dev_null
            self.score_pearson_residuals_ = (y - y_pred)/np.sqrt(y_pred)
            self.score_pearson_chi2_ = np.sum(self.score_pearson_residuals_**2)
            
            score = 1 - dev/dev_null
        else:
            score = scorer(X, y, sample_weight=sample_weight)
        
        return score
    

class LogReg(BaseEstimator, LinearClassifierMixin):
    def __init__(self, x0=None, lamb=1, l_norm=1, method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        
        self.optimize_args = {'fun': negloglike_logreg,
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
        
        
    def fit(self, X, y, has_constant=False):
        
        if not has_constant:
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