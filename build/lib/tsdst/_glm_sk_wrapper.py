from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from statsmodels.tools import add_constant


def norm(data, p):
    if p == 1:
        return np.sum(np.abs(data))
    elif p == 2:
        return np.sum(data*data)**0.5
    elif p >= 3:
        if p % 2 == 0:
            return np.sum(data**p)**(1/p)
        else:
            return np.sum(np.abs(data)**p)**(1/p)
    else:
        raise('Not a valid norm')


def negloglike_logreg(parms, X, Y, lamb=1, l_norm=1):

    intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    Ypred = 1.0/(1.0 + np.exp(-mu))
    #Ypred = np.sum([Ypred >= 0.5], axis=0)
    loglike = np.sum(xlogy(Y, Ypred) + xlogy(1.0 - Y, 1.0 - Ypred)) - lamb*norm(betas, l_norm)

    return -loglike


def posterior_logreg(parms, X, Y, l_scale=0.5):
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    # a common choice is scale = 2/lambda, or scale = 2*C. I find that when scale == C, you get similar results. This
    # parameterization is similar to scale = stddev/lambda or scale = stddev*C, where I set stddev to 1
    
    intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    Ypred = 1/(1 + np.exp(-mu))
    like = np.sum(Y*np.log(Ypred) + (1 - Y)*np.log(1 - Ypred))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2*(n_sigma**2))))
    # Laplace prior 
    parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    post = like + parm_prior + int_prior 
    return post


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
        
   
        
        
        
        