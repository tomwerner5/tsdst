"""
Statistical Distribution Functions
"""

from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import numpy as np

from scipy.stats import norm as normal, lognorm, gamma
from scipy.special import xlogy, loggamma

from tsdst.tmath import norm


# #########################################
# ########## General Functions ############
# #########################################


# This is a faster approximation of the normal quantile function.
# Accurate to 2 or 3 digits. For exact solutions, use dwrap or scipy
def qnorm_aprox(p, mu=0, sigma=1, lt=True):
    '''
    This is a faster approximation of the normal quantile function (At least
    this was true at the time I last benchmarked it). Accurate to 2 or 3
    digits. For exact solutions, use dwrap or scipy.

    Parameters
    ----------
    p : float or array-like
        percentile (or probability) of interest.
    mu : float, optional
        Mean of the normal distribution. The default is 0.
    sigma : float, optional
        Standard deviation of the normal distribution. The default is 1.
    lt : bool, optional
        Lower tail of the distribution. The default is True.

    Returns
    -------
    quant : float or numpy array
        Return quantile of interest.

    '''
    p = np.array(p)
    if not lt:
        p = 1 - p
    a = 0.147
    x = 2.0*p - 1.0
    b = (2.0/(np.pi*a)) + (np.log(1.0 - x**2)/2.0)
    c = (np.log(1.0 - x**2)/a)
    i_erf = np.sign(x)*np.sqrt(np.sqrt(b**2 - c) - b)
    quant = mu + sigma*np.sqrt(2)*i_erf
    return quant


def dwrap(data, params, disttype, funct, log=False):
    '''
    This function is meant to be similar to the R distribution functions,
    such as dnorm, pnorm, qnorm, etc. It calculates variations of the 
    cdf or pdf depending on the funct selected.
    
    I have found that writing the distributions in plain math is sometimes 
    faster in python than using the scipy implementations, which is why
    this function originally existed. I wrote my own versions of the
    distributions, except where the distributions were complicated and it
    wasn't worth it at the time (and then this function uses scipy). However,
    as time goes on, that will probably change. Also, the scipy.stats
    implementations and documentations are quite complete, so unless you're
    feeling adventurous, it's probably a good idea to just use scipy.

    Parameters
    ----------
    data : numpy array or pandas dataframe/series
        Numeric values representing the data of interest, either a random
        variable, probability, or quantile.
    params : numpy array or pandas dataframe/series
        the parameters of the function of interest (shape, scale, etc.)
        - weibull: (shape, scale)
        - 
    disttype : str
        the distribution type, which currently includes pdf, cdf, inv-cdf, sf,
        left-truncated-cdf, left-truncated-inv-cdf. (Note: not all of these
        options may be available for all funct options)
    funct : str
        the distribution function, which currently includes weibull, 
        exponential, log-normal (as lnorm), normal, and gamma
    log : bool, optional
        Whether to use the log of the distribution or not. The default is
        False.

    Raises
    ------
    ValueError
        Raised when invalid distribution type or function is chosen.

    Returns
    -------
    numpy array 
        an array containing the evaluation of the distribution.

    '''
    try:
        num_parms = params.shape[1]
        params = np.array(params).reshape(-1, num_parms)
    except (IndexError, AttributeError):
        params = np.array(params).reshape(1, -1)

    if funct == "weibull":
        shape = params[:, 0]
        scale = params[:, 1]
        if disttype == "pdf":
            if log:
                return (np.log(shape) - np.log(scale) +
                        xlogy(shape - 1.0, data/scale) -
                        (data/scale)**shape)
            else:
                return ((shape/scale)*((data/scale)**(shape - 1.0)) *
                        np.exp(-(data/scale)**shape))
        elif disttype == "cdf":
            if log:
                return np.log((1.0 - np.exp(-(data/scale)**shape)))
            else:
                return (1.0 - np.exp(-(data/scale)**shape))
        elif disttype == "sf":
            if log:
                return -(data/scale)**shape
            else:
                return np.exp(-(data/scale)**shape)
        elif disttype == "inv-cdf":
            if log:
                return np.log(scale) + ((1/shape)*np.log(-np.log(1 - data)))
            else:
                return scale*(-np.log(1 - data))**(1/shape)
        # The left-truncted distribution is useful for evaluating the
        # censored items (probability of failing given it lived this long)
        elif disttype == "left-truncated-cdf":
            a = params[:, 2]
            if log:
                return np.log(1 - (np.exp(((a/scale)**shape) - ((data/scale)**shape))))
            else:
                return 1 - (np.exp(((a/scale)**shape) - ((data/scale)**shape)))
        elif disttype == "left-truncated-inv-cdf":
            a = params[:, 2]
            if log:
                return np.log(scale) + (1/shape)*np.log((((a/scale)**shape) - np.log(1 - data)))
            else:
                return scale*(((a/scale)**shape) - np.log(1 - data))**(1/shape)
        else:
            raise ValueError("Not a valid distribution type")
    elif funct == "exponential":
        rate = params[:, 0]

        if disttype == "pdf":
            if log:
                return np.log(rate) - rate*data
            else:
                return rate * np.exp(-rate*data)
        elif disttype == "cdf":
            if log:
                return np.log(1.0 - np.exp(-rate*data))
            else:
                return 1.0 - np.exp(-rate*data)
        elif disttype == "sf":
            if log:
                return -rate*data
            else:
                return np.exp(-rate*data)
        else:
            raise ValueError("Not a valid distribution type")
    elif funct == "lnorm":
        mu = params[:, 0]
        sigma = params[:, 1]
        if disttype == "pdf":
            if log:
                return (-(np.log(data) + np.log(sigma) +
                          0.5*np.log(2*np.pi)) -
                        (((np.log(data) - mu)**2)/(2*sigma**2)))
            else:
                return ((1/(data*sigma*np.sqrt(2*np.pi))) *
                        np.exp(-((np.log(data) - mu)**2)/(2*sigma**2)))
        elif disttype == "cdf":
            if log:
                return lognorm.logcdf(x=data, scale=np.exp(mu), s=sigma)
            else:
                return lognorm.cdf(x=data, scale=np.exp(mu), s=sigma)
        elif disttype == "sf":
            if log:
                return lognorm.logsf(x=data, scale=np.exp(mu), s=sigma)
            else:
                return lognorm.sf(x=data, scale=np.exp(mu), s=sigma)
        else:
            raise ValueError("Not a valid distribution type")
    elif funct == "normal":
        mu = params[:, 0]
        sigma = params[:, 1]
        if disttype == "pdf":
            if log:
                return (-0.5*np.log(2*np.pi) - np.log(sigma) -
                        (((data - mu)**2) / (2*(sigma**2))))
            else:
                return ((1/(np.sqrt(2*np.pi)*sigma)) *
                        (np.exp(-((data - mu)**2)/(2*(sigma**2)))))
        elif disttype == "cdf":
            if log:
                return normal.logcdf(x=data, loc=mu, scale=sigma)
            else:
                return normal.cdf(x=data, loc=mu, scale=sigma)
        elif disttype == "sf":
            if log:
                return normal.logsf(x=data, scale=mu, s=sigma)
            else:
                return normal.sf(x=data, scale=mu, s=sigma)
        else:
            raise ValueError("Not a valid distribution type")
    elif funct == "gamma":
        param1 = params[:, 0]
        param2 = params[:, 1]
        if disttype == "pdf":
            if log:
                return gamma.logpdf(x=data, a=param1, scale=param2)
            else:
                return gamma.pdf(x=data, a=param1, scale=param2)
        elif disttype == "cdf":
            if log:
                return gamma.logcdf(x=data, a=param1, scale=param2)
            else:
                return gamma.cdf(x=data, a=param1, scale=param2)
        elif disttype == "sf":
            if log:
                return gamma.logsf(x=data, a=param1, scale=param2)
            else:
                return gamma.sf(x=data, a=param1, scale=param2)
        else:
            raise ValueError("Not a valid distribution type")
    else:
        raise ValueError("Not a valid distribution")


# #########################################
# ######## Likelihood Functions ###########
# #########################################


def negloglike_logreg(parms, X, Y, lamb=1, l_norm=1):
    '''
    The negative log-likelihood for a logistic regression model with a penalty
    term (can accept any norm, default is 1 for L1)
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The coefficients (including intercept, which is first)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    lamb : int, optional
        The size of the penalty (lambda). Note this is the inverse of the
        common sklearn parameter C (i.e. C=1/lambda. The default is 1.
    l_norm : int, optional
        The mathmatical norm to be applied to the coefficients.
        The default is 1, representing an L1 penalty.

    Returns
    -------
    float
        The negative log-likelihood.

    '''
    #intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    Ypred = 1.0/(1.0 + np.exp(-mu))
    #Ypred = np.sum([Ypred >= 0.5], axis=0)
    loglike = np.sum(xlogy(Y, Ypred) + xlogy(1.0 - Y, 1.0 - Ypred)) - lamb*norm(betas, l_norm)

    return -loglike


# #########################################
# ########## Posterior Functions ##########
# #########################################


def posterior_logreg_lasso(parms, X, Y, l_scale=0.5):
    '''
    The posterior likelihood for a logistic regression model with an L1 penalty
    term.
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The coefficients (including intercept, which is first)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)

    '''
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    
    intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    Ypred = 1.0/(1.0 + np.exp(-mu))
    like = np.sum(Y*np.log(Ypred) + (1.0 - Y)*np.log(1.0 - Ypred))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    post = like + parm_prior + int_prior 
    return post


def adaptive_posterior_logreg_lasso(parms, X, Y, l_scale=None):
    '''
    The posterior likelihood for a logistic regression model with an L1 penalty
    term. However, unlike the posterior_logreg_lasso function, this is made to
    addaptively learn the optimal L1 penalty. Therefore, the L1 penalty is
    in the parms variable, at the end of the array
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        ---THIS IS NOT USED. ONLY HERE FOR CONVENIENCE OF THE USER WHEN
        EXPERIMENTING BETWEEN THE ADAPTIVE AND NON-ADAPTIVE VERSIONS--- 
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_mu = 0
    n_sigma = 1
    l_loc = 0
    # assuming an exponential prior for the scale parameter of the laplace
    # mean/scale of 0.38 seemed to work well for these experiments
    l_scale_rate = 1/0.38
    
    intercept = parms[0]
    betas = parms[1:-1]
    l_scale = np.exp(parms[-1])
    
    mu = X.dot(parms[:-1].reshape(-1, )).reshape(-1, )
    Ypred = 1/(1 + np.exp(-mu))
    like = np.sum(Y*np.log(Ypred) + (1 - Y)*np.log(1 - Ypred))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2*(n_sigma**2))))
    # Laplace prior for Coefficient
    parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # exponential prior for scale parameter
    scale_prior = np.log(l_scale_rate) - l_scale_rate * l_scale
    # post = likelihood + laplace prior + jacobian for laplace scale + laplace scale prior
    post = like + int_prior + parm_prior + parms[-1] + scale_prior
    return post


def adaptive_poisson_regression(parms, X, Y, l_scale=None):
    '''
    The posterior likelihood for a poisson regression model with an L1 penalty
    term. However, unlike the poisson_regression function, this is made to
    addaptively learn the optimal L1 penalty. Therefore, the L1 penalty is
    in the parms variable, at the end of the array
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        ---THIS IS NOT USED. ONLY HERE FOR CONVENIENCE OF THE USER WHEN
        EXPERIMENTING BETWEEN THE ADAPTIVE AND NON-ADAPTIVE VERSIONS--- 
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_sigma = 10
    l_loc = 0
    # assuming an exponential prior for the scale parameter of the laplace
    # mean/scale of 0.38 seemed to work well for these experiments
    l_scale_rate = 1/0.38
    
    intercept = parms[0]
    betas = parms[1:-1]
    l_scale = np.exp(parms[-1])
    
    mu = X.dot(parms)
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - l_loc)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - l_loc)**2) / (2.0*(l_scale**2))))
    
    # exponential prior for scale parameter
    scale_prior = np.log(l_scale_rate) - l_scale_rate * l_scale
    # post = likelihood + laplace prior + jacobian for laplace scale + laplace scale prior
    
    post = like + parm_prior + int_prior + parms[-1] + scale_prior
    return post


def adaptive_poisson_regression_od(parms, X, Y, l_scale=None):
    '''
    The posterior likelihood for a poisson regression model with an L1 penalty
    term. However, unlike the poisson_regression function, this is made to
    addaptively learn the optimal L1 penalty. Therefore, the L1 penalty is
    in the parms variable, at the end of the array. This function also attempts
    to adaptively account for overdispersion.
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        ---THIS IS NOT USED. ONLY HERE FOR CONVENIENCE OF THE USER WHEN
        EXPERIMENTING BETWEEN THE ADAPTIVE AND NON-ADAPTIVE VERSIONS--- 
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_sigma = 10
    # I think in pretty much every case you want this zero, so l_loc applies
    # to both the normal mu parameter and the laplace location parameter
    l_loc = 0
    # assuming an exponential prior for the scale parameter of the laplace
    # mean/scale of 0.38 seemed to work well for these experiments
    l_scale_rate = 1/0.38
    
    intercept = parms[0]
    betas = parms[1:-2]
    l_scale = np.exp(parms[-2])
    
    mu = X.dot(parms[:-2]) - parms[-1]
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    #like = like/np.exp(parms[-1])
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - l_loc)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - l_loc)**2) / (2.0*(l_scale**2))))
    
    # exponential prior for scale parameter
    scale_prior = np.log(l_scale_rate) - l_scale_rate * l_scale
    # post = likelihood + laplace prior + jacobian for laplace scale + laplace scale prior
    
    post = like + parm_prior + int_prior + parms[-2] + scale_prior
    return post


def poisson_regression(parms, X, Y, l_scale=1):
    '''
    The posterior likelihood for a poisson regression model with an L1 penalty
    term.
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_sigma = 10
    # I think in pretty much every case you want this zero, so l_loc applies
    # to both the normal mu parameter and the laplace location parameter
    l_loc = 0
    
    intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - l_loc)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - l_loc)**2) / (2.0*(l_scale**2))))
    
    post = like + parm_prior + int_prior 
    return post


def poisson_regression_od(parms, X, Y, l_scale=1):
    '''
    The posterior likelihood for a poisson regression model with an L1 penalty
    term. This function also attempts to adaptively account for overdispersion.
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    l_scale : int, optional
        ---THIS IS NOT USED. ONLY HERE FOR CONVENIENCE OF THE USER WHEN
        EXPERIMENTING BETWEEN THE ADAPTIVE AND NON-ADAPTIVE VERSIONS--- 
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_sigma = 10
    # I think in pretty much every case you want this zero, so l_loc applies
    # to both the normal mu parameter and the laplace location parameter
    l_loc = 0
    
    intercept = parms[0]
    betas = parms[1:-1]
    
    mu = X.dot(parms[:-1]) - parms[-1]
    #mu = mu/np.exp(parms[-1])
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - l_loc)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - l_loc)**2) / (2.0*(l_scale**2))))
    # assuming uniform for over dispersion prior
    post = like + parm_prior + int_prior #+ parms[-1]
    return post


def weibull_regression_post(parms, X, Y, status=None, l_scale=1):
    '''
    The posterior likelihood for a weibull regression model with an L1 penalty
    term. 
    
    Parameters
    ----------
    parms : numpy array (numeric)
        The model coefficients (including intercept, which is first, and the
        scale of the laplace distribution, which is last)
    X : numpy array (numeric)
        The independent variables (or feature matrix), where the first column
        is a dummy column of 1's (for the intercept).
    Y : numpy array or pandas dataframe
        The response value (should be 0 or 1, but could be float as well if 
        you're willing to deal with those consequences).
    status : numpy array (int, bool) or None
        Indicates status used for censoring, in this case, right censoring.
        An array of 1's and 0's, such that a 1 indicates an event, and 0 is the
        censored observation. If None, it will assume there are no censored
        events. Default is None.
    l_scale : int, optional
        The size of the scale parameter in the Laplace distribution.
        A common choice for the laplace prior is scale = 2/lambda, where
        lambda is the L1 penalty, or scale = 2*C (where C is the penalty term
        in sklearn). I find that when scale == C, you get more similar results.
        to LogisticRegression output in sklearn. This
        parameterization is similar to scale = stddev/lambda or
        scale = stddev*C, where I set stddev to 1 instead of 2, as is common.
        The default value is 1.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density)
    '''
    n_mu = 0
    n_sigma = 10
    
    intercept = parms[0]
    betas = parms[1:-1]
    
    scale = np.exp(parms[-1])
    logshape = X.dot(parms[:-1])
    shape = np.exp(logshape)
    
    if status is None:
        status = np.repeat(1, X.shape[0])

    fails = Y[status == 1]
    cens = Y[status == 0]
    gshape = 0.01
    gscale = 100.0
    
    xly_1 = (shape - 1.0)*np.log(fails/scale)
    xly_2 = (gshape - 1.0)*np.log(scale)
    if (shape - 1.0) == 0:
        xly_1[np.where(np.isnan(xly_1))] = 0
    
    fail_like = (np.log(shape) - np.log(scale) + xly_1 - (fails/scale)**shape)
    cens_like = -(cens/scale)**shape
    
    #scale prior
    scale_prior = (-(loggamma(gshape) + gshape*np.log(gscale)) + xly_2 - (scale/gscale))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - n_mu)**2) / (2.0*(l_scale**2))))
    # assuming uniform for over dispersion prior
    post = np.sum(fail_like) + np.sum(cens_like) + parm_prior + int_prior + scale_prior + parms[-1]
    return post


def pois_uniform(param, count):
    '''
    Posterior Likelihood for the Poisson distribution (assuming uniform prior)

    Parameters
    ----------
    param : float
        log of the Poisson mean (lambda).
    count : int, or array of ints
        count of events (or poisson counts)

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density).

    '''
    lmbda = np.exp(param)
    like = np.sum(count * param - lmbda - loggamma(count + 1.0))
    jac = param
    return like + jac


def pois_gamma(param, count, gshape=0.01, gscale=100):
    '''
    Posterior Likelihood for the Poisson distribution (assuming gamma prior)

    Parameters
    ----------
    param : float
        Log of the Poisson mean (lambda).
    count : int, or array of ints
        count of events (or poisson counts)
    gshape : float
        shape parameter of the gamma prior
    gscale : float
        scale parameter of the gamma prior

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density).

    '''
    lmbda = np.exp(param)
    like = np.sum(count * param - lmbda - loggamma(count + 1.0))
    prior = (-(loggamma(gshape) + gshape*np.log(gscale)) +
             xlogy(gshape-1, lmbda) - (lmbda/gscale))
    jac = param
    return like + prior + jac


def pois_gamma_ada(param, count):
    '''
    Posterior Likelihood for the Poisson distribution (assuming gamma prior),
    learns the gamma shape/scale as part of mcmc

    Parameters
    ----------
    param : numpy array (float)
        The log of the Poisson mean (lambda), and the shape/scale
        of the gamma prior.
    count : int, or array of ints
        count of events (or poisson counts)

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density).

    '''
    lmbda = np.exp(param[0])
    gshape = np.exp(param[1])
    gscale = np.exp(param[2])
    like = np.sum(count * param[0] - lmbda - loggamma(count + 1.0))
    prior = (-(loggamma(gshape) + gshape*np.log(gscale)) +
             xlogy(gshape-1, lmbda) - (lmbda/gscale))
    jac = np.sum(param)
    return like + prior + jac


def weibull_lpost(param, data, status=None, neg=False):
    '''
    The posterior likelihood for a weibull distribution with a gamma prior

    Parameters
    ----------
    param : numpy array (float)
        The log shape and scale parameters
    data : numpy array (float)
        The data being fit
    status : numpy array (int, or bool), optional
        Indicates status used for censoring, in this case, right censoring.
        An array of 1's and 0's, such that a 1 indicates an event, and 0 is the
        censored observation. If None, it will assume there are no censored
        events. The default is None.
    neg : bool, optional
        Whether to return the negative of the posterior or not. The default
        is False.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density).

    '''
    shape = np.exp(param[0])
    scale = np.exp(param[1])

    if status is None:
        status = np.repeat(1, data.shape[0])

    fails = data[status == 1]
    cens = data[status == 0]
    gshape = 0.01
    gscale = 100.0

    xly_1 = (shape - 1.0)*np.log(fails/scale)
    xly_2 = (gshape - 1.0)*np.log(scale)
    if (shape - 1.0) == 0:
        xly_1[np.where(np.isnan(xly_1))] = 0

    fail_like = (np.log(shape) - np.log(scale) +
                 xly_1 - (fails/scale)**shape)
    cens_like = -(cens/scale)**shape

    prior = (-(loggamma(gshape) + gshape*np.log(gscale)) +
             xly_2 - (scale/gscale))

    post = np.sum(fail_like) + np.sum(cens_like) + prior + param[0] + param[1]
    if neg:
        return -post
    else:
        return post


def NHPP_lpost_Math(lparm, tints, tbar, obs, a, b, mu, sigma, neg=False):
    '''
    The posterior likelihood of the Non-Homogenous Poisson Distribution with
    a power law process. 

    Parameters
    ----------
    lparm : numpy array (float)
        eta and phi (shape and scale with a variable change).
    tints : numpy array (int)
        The sequence of time intervals (1,2,3, etc...) for the poisson counts.
    tbar : int
        time between tints (usually 1).
    obs : numpy array (int)
        The data (usually count) for the poisson process (equal to length as tints).
    a : float
        lower bound of the uniform prior on phi.
    b : float
        upper bound of the uniform prior on phi.
    mu : float
        This is a paramter that is part of the variable change on the gamma
        prior for the poisson likelihood.
    sigma : float
        This is a paramter that is part of the variable change on the gamma
        prior for the poisson likelihood.
    neg : bool, optional
        Whether to return the negative of the posterior or not. The default
        is False.

    Returns
    -------
    float
        The posterior likelihood (height of the posterior density).

    '''
    # The Log postirior for NHPP with rate = (phi/eta)*(t/eta)**(phi-1) using
    # only Math no premade distributions.

    # Preform variable change.
    shape = (np.exp(lparm[0]) * b + a) / (1 + np.exp(lparm[0]))
    scale = np.exp(lparm[1])

    # Extrace interval start and end times from time series.
    int_start = tints[0:len(tints)-1]
    int_end = tints[1:len(tints)]

    # Clculate log prior
    # This prior is a Gamma distribution with shape = (mu/sigma)^2,
    # Rate = (mu/sigma^2), a verible change of x = (t_bar/eta)^phi was made
    # with variable eta and phi and t_bar as constants. So the end prior is
    # Gamma((t_bar/eta)^phi,(mu/sigma)^2,(mu/sigma^2))*(dx/deta)*Uniform(phi,a,b)
    lp = (np.log(shape) + (mu/sigma)*(mu/sigma)*np.log(mu/(sigma*sigma)) +
          (shape*(mu/sigma)*(mu/sigma))*(np.log(tbar)-np.log(scale)) -
          np.log(scale) - (mu/(sigma*sigma))*((tbar/scale)**shape) -
          loggamma((mu/sigma)*(mu/sigma)))
    lp_unif = 0.0
    if a >= b:
        lp_unif = np.nan
    elif a <= shape <= b:
        lp_unif = -np.log(b - a)

    lprior = lp + lp_unif

    # Calulate log Liklihood
    llik = np.sum(
            obs*(np.log(((int_end/scale)**shape) -
                        ((int_start/scale)**shape))) -
                (((int_end/scale)**shape) -
                 ((int_start/scale)**shape)) - loggamma(obs + 1))

    # Calulate Log Jacobian from the above variable change.
    # the term np.log(b-a) was removed because it's just a constant.
    ljac = (lparm[0] + lparm[1] - 2.0*np.log(1.0 + np.exp(lparm[0])))

    # Sum and return
    if neg:
        return -(llik + lprior + ljac)
    else:
        return llik + lprior + ljac

    