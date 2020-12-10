"""
Statistical Distribution Functions
"""

from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import numpy as np

from scipy.stats import norm, lognorm, gamma
from scipy.special import xlogy, loggamma

from tsdst.tmath import norm


# #########################################
# ########## General Functions ############
# #########################################


def dwrap(data, params, disttype, funct, log=False):
    
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
                        t_xlogy(shape - 1.0, data/scale) -
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
    elif funct == "norm":
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
                return norm.logcdf(x=data, loc=mu, scale=sigma)
            else:
                return norm.cdf(x=data, loc=mu, scale=sigma)
        elif disttype == "sf":
            if log:
                return norm.logsf(x=data, scale=mu, s=sigma)
            else:
                return norm.sf(x=data, scale=mu, s=sigma)
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

    intercept = parms[0]
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
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    # a common choice for the laplace prior is scale = 2/lambda, where lambda is the L1 penalty,
    # or scale = 2*C. I find that when scale == C, you get similar results. This
    # parameterization is similar to scale = stddev/lambda or scale = stddev*C, where I set stddev to 1
    
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


def adap_poisson_regression(parms, X, Y, l_scale=None):
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    # assuming an exponential prior for the scale parameter of the laplace
    # mean/scale of 0.38 seemed to work well for these experiments
    l_scale_rate = 1/0.38
    
    intercept = parms[0]
    betas = parms[1:-1]
    l_scale = np.exp(params[-1])
    
    mu = X.dot(parms)
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - n_mu)**2) / (2.0*(l_scale**2))))
    
    # exponential prior for scale parameter
    scale_prior = np.log(l_scale_rate) - l_scale_rate * l_scale
    # post = likelihood + laplace prior + jacobian for laplace scale + laplace scale prior
    
    post = like + parm_prior + int_prior + parms[-1] + scale_prior
    return post


def adap_poisson_regression_od(parms, X, Y, l_scale=None):
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    # assuming an exponential prior for the scale parameter of the laplace
    # mean/scale of 0.38 seemed to work well for these experiments
    l_scale_rate = 1/0.38
    
    intercept = parms[0]
    betas = parms[1:-2]
    l_scale = np.exp(params[-2])
    
    mu = X.dot(parms[:-2]) - parms[-1]
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    #like = like/np.exp(parms[-1])
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - n_mu)**2) / (2.0*(l_scale**2))))
    
    # exponential prior for scale parameter
    scale_prior = np.log(l_scale_rate) - l_scale_rate * l_scale
    # post = likelihood + laplace prior + jacobian for laplace scale + laplace scale prior
    
    post = like + parm_prior + int_prior + parms[-2] + scale_prior
    return post


def poisson_regression(parms, X, Y, l_scale=1):
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    
    intercept = parms[0]
    betas = parms[1:]
    
    mu = X.dot(parms)
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - n_mu)**2) / (2.0*(l_scale**2))))
    
    post = like + parm_prior + int_prior 
    return post


def poisson_regression_od(parms, X, Y, l_scale=1):
    n_mu = 0
    n_sigma = 10
    l_loc = 0
    
    intercept = parms[0]
    betas = parms[1:-1]
    
    mu = X.dot(parms[:-1]) - parms[-1]
    #mu = mu/np.exp(parms[-1])
    
    like = np.sum(Y*mu - np.exp(mu) - loggamma(Y + 1))
    # normal prior on the intercept
    int_prior = (-0.5*np.log(2.0*np.pi) - np.log(n_sigma) - (((intercept - n_mu)**2) / (2.0*(n_sigma**2))))
    # Laplace prior 
    #parm_prior = np.sum(-np.log(2*l_scale) - (np.abs(betas - l_loc)/l_scale))
    # Normal prior
    parm_prior = np.sum(-0.5*np.log(2.0*np.pi) - np.log(l_scale) - (((betas - n_mu)**2) / (2.0*(l_scale**2))))
    # assuming uniform for over dispersion prior
    post = like + parm_prior + int_prior #+ parms[-1]
    return post


def weibull_regression_post(parms, X, Y, status=None, l_scale=1):
    n_mu = 0
    n_sigma = 10
    
    intercept = parms[0]
    betas = parms[1:-1]
    
    scale = np.exp(parms[-1])
    logshape = X.dot(parms[:-1])
    
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
    post = np.sum(fail_like) + np.sum(cens_like) + parm_prior + int_prior + parms[-1]
    return post


def pois_uniform(param, count):
    lmbda = np.exp(param)
    like = np.sum(count * param - lmbda - loggamma(count + 1.0))
    jac = param
    return like + jac


def pois_gamma(param, count, gshape=0.01, gscale=100):
    lmbda = np.exp(param)
    like = np.sum(count * param - lmbda - loggamma(count + 1.0))
    prior = (-(loggamma(gshape) + gshape*np.log(gscale)) +
             xlogy(gshape-1, lmbda) - (lmbda/gscale))
    jac = param
    return like + prior + jac


def pois_gamma_ada(param, count):
    lmbda = np.exp(param[0])
    gshape = np.exp(param[1])
    gscale = np.exp(param[2])
    like = np.sum(count * param[0] - lmbda - loggamma(count + 1.0))
    prior = (-(loggamma(gshape) + gshape*np.log(gscale)) +
             xlogy(gshape-1, lmbda) - (lmbda/gscale))
    jac = np.sum(param)
    return like + prior + jac


def weibull_lpost(param, time, status=None, neg=False):
    shape = np.exp(param[0])
    scale = np.exp(param[1])

    if status is None:
        status = np.repeat(1, X.shape[0])

    fails = time[status == 1]
    cens = time[status == 0]
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
    

def weibull_lpost_optim(param, fails, cens):

    shape = np.exp(param[0])
    scale = np.exp(param[1])

    # note: this particular calculation assumes gshape (or the
    # prior shape for the gamma distribution) as 0.01 and gscale as 100.0

    xly_1 = (shape - 1.0)*(np.log(fails) - param[1])
    xly_2 = -0.99*param[1]
    if (shape - 1.0) == 0:
        xly_1[np.where(np.isnan(xly_1))] = 0

    fail_like = (param[0] - param[1] +
                 xly_1 - (fails/scale)**shape)
    cens_like = -(cens/scale)**shape

    prior = (-4.645531579901903 +
             xly_2 - (scale*0.01))

    post = np.sum(fail_like) + np.sum(cens_like) + prior + param[0] + param[1]

    return post


def NHPP_lpost_Math(lparm, tints, tbar, obs, a, b, mu, sigma, neg=False):
    # The Log postirior for NHPP with rate = (phi/eta)*(t/eta)**(phi-1) using
    # only Math no premade distributions.

    # Preform variable change.
    shape = (np.exp(lparm[0]) * b + a) / (1 + np.exp(lparm[0]))
    scale = np.exp(lparm[1])

    # Extrace interval start and end time staps from time series.
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

    