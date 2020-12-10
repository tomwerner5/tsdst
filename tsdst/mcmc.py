from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
import random
import seaborn as sns
import sys
import warnings

from numba import jit
from numpy import linalg as la
from scipy.special import loggamma
from scipy.stats import chi2, f
from scipy.linalg import toeplitz, solve
from sklearn.preprocessing import scale
from timeit import default_timer as dt

from tsdst.utils import pretty_print_time
from tsdst.distributions import qnorm_aprox


def updateProgBarMCMC(curIter, totalIter, t0, ar, barLength=20):
    '''
    Custom progress bar to output MCMC chain progress.

    Parameters
    ----------
    curIter : int
        Current iteration.
    totalIter : int
        Total iterations.
    t0 : float
        Timestamp of when the process started (timestamp as float).
    ar : float
        Acceptance Ratio.
    barLength : int, optional
        The character length of the progress bar. The default is 20.

    Returns
    -------
    None.

    '''
    status = "Working..."
    progress = float(curIter)/float(totalIter)
    if isinstance(progress, int):
        progress = float(progress)
    if progress >= 1:
        progress = 1
        status = "Finished!..."
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% iter: {2}/{3}, {4} Elapsed: {5}, Est: {6}, Accept. Rate: {7}".format(
        "#"*block + "-"*(barLength - block), 
        round(progress*100.0, 2), curIter, totalIter, status, pretty_print_time(t0, dt()),
        pretty_print_time((dt()-t0)/curIter * (totalIter - curIter)), np.round(ar, 3))
    if progress >= 1:
        sys.stdout.write(text + "\r\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
        

def applyMCMC(st, ni, lp, algo, algoOpts=None, postArgs={},
              sd=0.02, max_tries=100):
    '''
    This function iteratively applies the MCMC initialization. Since the MCMC
    algorithms used here involve a cholesky decomposition, the methods
    sometimes get stuck with a covaraince matrix that is not positive definite.
    This will attempt to jitter the covariance matrix until it can initialize
    properly.

    Parameters
    ----------
    st : numpy array
        An array of the parameter starting values.
    ni : int
        NUmber of MCMC iterations.
    lp : function
        Function for the log posterior.
    algo : function
        MCMC algorithm to be performed.
    algoOpts : dict, optional
        Specific options for the MCMC algorithm. The default is None.
    postArgs : dict, optional
        Specific options for the posterior function. The default is None.
    sd : float, optional
        The standard deviation of the normal distribution used to draw the
        jitter amount from. In other words, the jittered covariance is the 
        covaraince matrix plus a random draw X, where X~N(0, sd). 
        The default is 0.02.
    max_tries : int, optional
        The max number of times to try and jitter before admitting defeat.
        If the jitter fails, the reason or the covaraince matrix not being
        positive definite may not be due to randomness, and may require
        a re-evaluation of the problem space. The default is 100.

    Raises
    ------
    ValueError
        Raised when cholesky decomposition fails after max_tries.

    Returns
    -------
    res : tuple
        Returns tuple containing the MCMC results.

    '''
    try_num = 1
    not_successful = True
    res = None
    lns = st.size
    while not_successful:
        if try_num % 5 == 0:
            st = st + np.random.normal(size=lns, scale=sd)
        try:
            res = algo(start=st, niter=ni, lpost=lp, postArgs=postArgs,
                       options=algoOpts)
            not_successful = False
        except np.linalg.LinAlgError:
            try_num += 1
        
        if try_num >= max_tries:
            raise ValueError("Cholesky Decomposition was not successful after " + str(max_tries) + " tries. Try new starting values")
    print("Number of Cholesky tries: " + str(try_num))
    return res              


# For upper triangle rank one update
@jit
def cholupdate(L, x, update=True):
    '''
    Upper triangle, rank one update for cholesky decomposed matrix. 

    Parameters
    ----------
    L : numpy array (float)
        The upper-triangular decomposed matrix, shape=(N, N).
    x : numpy array (float)
        The values being added to L, shape=(N, ).
    update : bool, optional
        Perform an update (as opposed to a downdate). The default is True.

    Returns
    -------
    L : numpy array
        Return updated L matrix.

    '''
    p = len(x)
    for k in range(p):
        if update:
            r = np.sqrt((L[k, k]**2) + (x[k]**2))
        else:
            r = np.sqrt((L[k, k]**2) - (x[k]**2))
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < (p - 1):
            if update:
                L[k, (k + 1):p] = (L[k, (k + 1):p] + s * x[(k + 1):p]) / c
            else:
                L[k, (k + 1):p] = (L[k, (k + 1):p] - s * x[(k + 1):p]) / c
            x[(k + 1):p] = c * x[(k + 1):p] - s * L[k, (k + 1):p]
    return L


def adaptive_mcmc(start, niter, lpost, postArgs={}, options=None):
    '''
    A random walk metropolis algorithm that adaptively tunes the covaraince 
    matrix. Based on methods by Rosenthal and Haario.

    Parameters
    ----------
    start : numpy array
        Starting values for the MCMC.
    niter : int
        Number of iterations.
    lpost : function
        Log posterior function.
    postArgs : dict
        Extra arguments for the log posterior function. If there are none, pass
        an empty dictionary.
    options : dict, optional
        Extra arguments for the MCMC algorithm, namely:
            beta : float
                Between 0 and 1. Decides the proportion to sample for each
                section of the mixture distribution.
                
                A mixture distribution is essentially like adding two 
                distributions together. However, to avoid some complicated
                math, one way to sample from a mixture of two distributions is
                to use a trick, namely, to first sample from a uniform
                distribution between 0, 1, and then evaluate whether that value
                is above some threshold (beta in this case). If it is, sample 
                from the first distribution, otherwise, sample from the second.
            progress : bool
                Whether to display progress bar
            prev_vals : dict
                The previous values of the last run, namely:
                    chol2 : numpy array
                        the decomposed covariance matrix of the parameters
                    sumx : numpy array
                        the current sum of the parameter value (for each
                                                                parameter) 
                    prev_i : int or float
                        the number of samples represented in sumx.
                        Used in averaging sumx
        The default is None.

    Returns
    -------
    parm : numpy array
        MCMC samples.
    prev_vals : dict
        The ending values of the MCMC algorithm. Useful when you want to
        continue where you left off.

    '''
    
    beta = 0.05
    progress = True
    prev_vals = {'chol2': None, 'sumx': 0.0, 'prev_i': 0.0}
    if options is not None:
        keys = list(options.keys())
        if 'beta' in keys:
            beta = options['beta']
        if 'progress' in keys:
            progress = options['progress']
        if 'prev_vals' in keys:
            prev_vals = options['prev_vals']
    
    numParams = start.size
    sqrtNumParams = np.sqrt(numParams)
    parm = np.zeros(shape=(niter, numParams))
    parm[0, ] = start
    sumx = start + prev_vals['sumx']
    accept = 0
    post_old = lpost(start, **postArgs)
    
    prop_dist_var = (0.1**2) * np.diag(np.repeat(1, numParams)) / numParams
    chol1 = la.cholesky(prop_dist_var)
    chol2 = prev_vals['chol2']
    acceptDraw = False
    loop = range(1, niter)
    
    sumi = 1.0 + prev_vals['prev_i']
    t0 = dt()
    for i in loop:
        parm[i, ] = parm[i - 1, ]
        
        if i <= ((2 * numParams) - 1):
            tune = chol1
        else:
            if chol2 is None:
                XXt = parm[0:i, ].T.dot(parm[0:i, ])
                chol2 = la.cholesky(XXt).T
            else:
                chol2 = cholupdate(chol2, np.array(parm[i - 1, ]))
            
            if random.random() < beta:
                tune = chol1
            else:
                tune = (2.38*cholupdate(chol2 / np.sqrt(sumi), sumx/sumi, update=False) / sqrtNumParams * np.sqrt(sumi / (sumi - 1)))
        
        if np.any(np.isnan(tune)):
            tune = chol1
        cand = np.random.normal(size=numParams).dot(tune) + parm[i - 1, ]
        post_new = lpost(cand, **postArgs)
        
        if (post_new - post_old) > np.log(random.random()):
            acceptDraw = True
        
        if acceptDraw:
            parm[i, ] = cand
            post_old = post_new
            accept += 1
        
        sumx = sumx + parm[i, ]
        sumi += 1.0
        acceptDraw = False
        if progress:
            updateProgBarMCMC(i + 1, niter, t0, float(accept) / float(i))
        
    prev_vals = {'chol2': chol2, 'prev_i': sumi - 1, 'sumx': sumx}
    print("Acceptance Rate: ", float(accept) / float(niter))
    return parm, prev_vals


def rwm_with_lap(start, niter, lpost, postArgs={}, options=None):
    '''
    A random walk metropolis algorithm that adaptively tunes the covaraince 
    matrix.

    Parameters
    ----------
    start : numpy array
        Starting values for the MCMC.
    niter : int
        Number of iterations.
    lpost : function
        Log posterior function.
    postArgs : dict
        Extra arguments for the log posterior function. The default is
        an empty dictionary.
    options : dict, optional
        Extra arguments for the MCMC algorithm, namely:
            k : int
                The number of MCMC samples to generate for each evaluation.
            c0 : float
                TODO: add DESCRIPTION.
            c1 : float
                TODO: add DESCRIPTION.
            progress : bool
                Whether to display progress bar
            prev_vals : dict
                The previous values of the last run, namely:
                    E_0 : numpy array
                        the final covaraince matrix
                    sigma_2 : float
                        the standard deviation multiplier in the algorithm
                    t : int
                        the current iteration number
        The default is None.

    Returns
    -------
    parm : numpy array
        MCMC samples.
    prev_vals : dict
        The ending values of the MCMC algorithm. Useful when you want to
        continue where you left off.

    '''
    k = 20
    c_0 = 1.0
    c_1 = 0.8
    progress = True
    prev_vals = {'E_0': None, 'sigma_2': None, 't': 0.0}
    if options is not None:
        keys = list(options.keys())
        if 'k' in keys:
            k = options['k']
        if 'c_0' in keys:
            c_0 = options['c_0']
        if 'c_1' in keys:
            c_1 = options['c_1']
        if 'progress' in keys:
            progress = options['progress']
        if 'prev_vals' in keys:
            prev_vals = options['prev_vals']
    
    numParams = start.size
    optimal = 0.444
    if numParams >= 2:
        optimal = 0.234
    T_iter = np.ceil(niter/float(k))
    niter = int(T_iter * k)
    parm = np.zeros(shape=(niter, numParams))
    parm[0, ] = start
    
    total_accept = k_accept = 0
    post_old = lpost(start, **postArgs)
    
    sigma_2 = (2.38**2)/numParams
    if prev_vals['sigma_2'] is not None:
        sigma_2 = prev_vals["sigma_2"]
    
    E_0 = np.diag(np.repeat(1, numParams))
    if prev_vals['E_0'] is not None:
        E_0 = prev_vals["E_0"]

    chol = la.cholesky((sigma_2)*E_0)
    chol_i = np.array(chol)
    
    t = 1 + prev_vals['t']
    
    acceptDraw = False
    loop = range(1, niter)
    
    t0 = dt()
    for i in loop:
        parm[i, ] = parm[i - 1, ]
        cand = np.random.normal(size=numParams).dot(chol) + parm[i - 1, ]
        
        post_new = lpost(cand, **postArgs)
        
        if (post_new - post_old) > np.log(random.random()):
            acceptDraw = True
        
        if acceptDraw:
            parm[i, ] = cand
            post_old = post_new
            k_accept += 1
            total_accept += 1
            
        acceptDraw = False
        if progress:
            updateProgBarMCMC(i + 1, niter, t0, float(total_accept) / float(i))
        
        if (i + 1) % k == 0:
            X = parm[(i + 1 - k):(i + 1), :]
            mean_X = np.mean(X, axis=0)
            
            r_t = k_accept / float(k)
            Ehat_0 = (1.0 / (k - 1.0)) * ((X - mean_X).T.dot((X - mean_X)))
            gamma_1 = 1/(t**c_1)
            gamma_2 = c_0 * gamma_1
            sigma_2 = np.exp(np.log(sigma_2) + (gamma_2 * (r_t - optimal)))
            E_0 = E_0 + gamma_1*(Ehat_0 - E_0)
            
            if np.any(np.isnan(E_0)) or not np.all(np.isfinite(E_0)):
                chol = chol_i
            else:
                try:
                    chol = la.cholesky(sigma_2*E_0)
                #except la.LinAlgError:
                #    chol = sla.sqrtm(sigma_2*E_0)
                except:
                    chol = chol_i
                    
            t += 1
            k_accept = 0
            
    prev_vals = {'E_0': E_0, 'sigma_2': sigma_2, 't': t}
    print("Acceptance Rate: ", float(total_accept) / float(niter))
    return parm, prev_vals


def rwm(start, niter, lpost, postArgs={}, options=None):
    '''
    A random walk metropolis algorithm.

    Parameters
    ----------
    start : numpy array
        Starting values for the MCMC.
    niter : int
        Number of iterations.
    lpost : function
        Log posterior function.
    postArgs : dict
        Extra arguments for the log posterior function. The default is
        an empty dictionary.
    options : dict, optional
        Extra arguments for the MCMC algorithm, namely:
            E : numpy array
                The covariance matrix
            progress : bool
                Whether to display progress bar
            prev_vals : dict
                The previous values of the last run, namely:
                    E_0 : numpy array
                        the final covaraince matrix
        The default is None.

    Returns
    -------
    parm : numpy array
        MCMC samples.
    prev_vals : dict
        The ending values of the MCMC algorithm. Useful when you want to
        continue where you left off.

    '''
    prev_vals = E = None
    progress = True
    if options is not None:
        keys = list(options.keys())
        if 'E' in keys:
            E = options['E']
        if 'progress' in keys:
            progress = options['progress']
        if 'prev_vals' in keys:
            prev_vals = options['prev_vals']
    
    numParams = start.size
    parm = np.zeros(shape=(niter, numParams))
    parm[0, ] = start
    
    accept = 0
    post_old = lpost(start, **postArgs)
    
    if E is None:
        E = ((2.38**2)/numParams)*np.diag(np.repeat(1, numParams))
    chol = la.cholesky(E)
    
    acceptDraw = False
    loop = range(1, niter)
    
    t0 = dt()
    for i in loop:
        parm[i, ] = parm[i - 1, ]
        cand = np.random.normal(size=numParams).dot(chol) + parm[i - 1, ]
        
        post_new = lpost(cand, **postArgs)
        
        if (post_new - post_old) > np.log(random.random()):
            acceptDraw = True
        
        if acceptDraw:
            parm[i, ] = cand
            post_old = post_new
            accept += 1
        
        acceptDraw = False
        if progress:
            updateProgBarMCMC(i + 1, niter, t0, float(accept) / float(i))
    
    prev_vals = {'E_O': E}
    print("Acceptance Rate: ", float(accept) / float(niter))
    return parm, prev_vals


def samp_size_calc_raftery(q=0.025, r=0.005, s=0.95):
    '''
    Calculate needed sample size for Raftery evaluation.

    Parameters
    ----------
    q : float, optional
        Quantile of interest (in terms of percentile, i.e. between 0 and 1).
        The default is 0.025.
    r : float, optional
        Accuracy. The default is 0.005.
    s : float, optional
        Probability. The default is 0.95.

    Returns
    -------
    phi : float
        Phi parameter in Raftery evaluation.
    nmin : int
        Minimum number of samples needed.

    '''
    phi = qnorm_aprox(0.5 * (1.0 + s))
    nmin = np.ceil((q * (1.0 - q) * phi**2)/r**2).astype(int)
    return phi, nmin


def mbmc(chain, b):
    '''
    Translated from R's mcmcse package

    Parameters
    ----------
    chain : numpy array
        MCMC chain.
    b : int
        Number of blocks.

    Returns
    -------
    numpy array
        DESCRIPTION.

    '''
    n = chain.shape[0]
    chain = chain.reshape(n, -1)
    p = chain.shape[1]
    a = int(np.floor(n/b))
    y_mean = np.zeros(p)
    out = np.zeros((p, p))
    block_means = np.zeros((a, p))
    mean_mat = np.zeros((a, p))
    idx = np.arange(a) * int(b)

    for i in range(int(b)):
        block_means += chain[idx, :]
        idx += 1
    block_means = block_means/b

    y_mean = np.mean(chain, axis=0)
    for i in range(a):
        mean_mat[i, :] = y_mean

    out = (block_means - mean_mat).T.dot(block_means - mean_mat)
    return (out*b/(a - 1))


def lag(s, b, method):
    if method == "bartlett":
        return 1 - (s/b)
    else:
        return ((1 + np.cos(np.pi * s/b))/2)    


def msvec(chain, b, method="bartlett"):
    n = chain.shape[0]
    chain = chain.reshape(n, -1)
    p = chain.shape[1]
    tchain = chain.T
    out = np.zeros((p, p))
    dummy = np.zeros((p, p))

    for s in range(int(b)):
        dummy = tchain[:, 0:(n-s-1)].dot(chain[s:n-1, :])
        out += lag(s, b, method)*(dummy + dummy.T)

    out += tchain.dot(chain)
    return out/n


def mcse_multi(chain, method="bm", r=3, size="sqroot", g=None, level=0.95,
               adjust=True, blather=False):
    method_used = method
    if method == "lug":
        method = "bm"
        r = 3
    if r > 5:
        warnings.warn("""It's recommended to use r <= 5. Also, r cannot be
                      negative""")
    if r < 0:
        raise ValueError("r cannot be negative.")
    if g is not None:
        chain = np.array([g(chain[i, :]) for i in range(chain.shape[0])])
    n = chain.shape[0]
    chain = chain.reshape(n, -1)
    p = chain.shape[1]
    # TODO: include batchSize function as option in the ifelse below (see 
    # mcmcse R docs)
    #if size is None:
    #    batchSize(chain=chain, method=method, g=g)
    if size == "sqroot":
        b = np.floor(np.sqrt(n))
    elif size == "cuberoot":
        b = np.floor(n**(1/3))
    else:
        if size < 1 or size >= n or np.floor(n/size) <= 1:
            raise ValueError("""'size' must be a numeric quantity not
                             larger than n.""")
        b = np.floor(size)

    a = np.floor(n/b)
    mu_hat = np.mean(chain, axis=0)
    m = 0
    # sig_sum = np.zeros(0, (p, p))
    if method != "bm" and method != "bartlett" and method != "tukey":
        raise ValueError("No such method available")
    if method == "bm":
        sig_mat = mbmc(chain, b)
        m = a - 1
    if method == "bartlett":
        chain = scale(chain, with_mean=True, with_std=False)
        sig_mat = msvec(chain, b, "bartlett")
        m = n - b
    if method == "tukey":
        chain = scale(chain, with_mean=True, with_std=False)
        sig_mat = msvec(chain, b, "tukey")
        m = n - b
    if (m - p + 1 <= 0):
        warnings.warn("Not enough samples. Estimate is not positive definite")
        pth_vol = np.nan
    else:
        sig_eigen = np.linalg.eigvals(sig_mat)
        if (min(sig_eigen) <= 0):
            warnings.warn("""You either need more samples or x is not
                          full column rank""")
        log_dethalf_pth = (1/(2 * p)) * np.sum(np.log(sig_eigen))
        crit = chi2.ppf(level, df=p)/n if large else np.exp(np.log(p) + np.log(m) - np.log(n) - np.log(m - p + 1) + np.log(f.ppf(level, p, m - p + 1)))
        foo = (1/p) * np.log(2) + (1/2) * np.log(np.pi * crit) - (1/p) * np.log(p) - (1/p) * loggamma(p/2)
        pth_vol = np.exp(foo + log_dethalf_pth)

    return {'cov': sig_mat, 'vol': pth_vol, 'est': mu_hat, 'nsim': n,
            'large': large, 'size': b, 'adjust_used': adjust_used,
            'method': method, 'method_used': method_used}


def minESS(p, alpha=0.05, eps=0.05, ess=None):
    '''
    Translated from the R mcmcse package.
    
    Calculates the minimum Effective Sample Size of an MCMC chain for the
    given parameters. `alpha` is the confidence level, `eps` is the
    tolerance level (ignored when `ess is not None`), and `ess` is the
    effective sample size. When `ess is not None`, the function returns
    the confidence level needed to obtain that ESS.

    see mulitESS for more information.

    Parameters
    ----------
    p : int
        The dimension of the estimation problem.
    alpha : float, optional
        Confidence level. The default is 0.05.
    eps : float, optional
        tolerance level. The default is 0.05.
    ess : int, optional
        Estimated effective sample size. The default is None.

    Returns
    -------
    int
        The minimum effective sample required for a given eps tolerance.
        If ess is specified, then the value returned is the eps corresponding
        to that ess.

    '''
    crit = chi2.ppf(1 - alpha, p)
    p2 = 2/p
    if ess is None:
        logminESS = p2 * np.log(2) + np.log(np.pi) - p2 * np.log(p) - p2 * loggamma(p/2) - 2 * np.log(eps) + np.log(crit)
        return np.round(np.exp(logminESS))
    else:
        logEPS = 0.5 * p2 * np.log(2) + 0.5 * np.log(np.pi) - 0.5 * p2 * np.log(p) - 0.5 * p2 * loggamma(p/2) - 0.5 * np.log(ess) + 0.5 * np.log(crit)
        return np.exp(logEPS)


def multiESS(chain, covmat=None, g=None, mcse_multi_args={}):
    '''
    This function computes the Effective Sample Size of an MCMC chain. Due to
    correlation between MCMC samples, it is sometimes unclear how much
    information about the parameters has been obtained. If all of the MCMC
    samples were independent, we would need less samples to get accurate
    information about the posterior than when the samples are correlated.
    ESS measures the amount of independent samples that have actually been
    obtained in the MCMC chain, and mESS is a special case for multivariate
    posteriors. In other words, this method is a way to test if your chain
    has gone far enough.
    
    This information can used in conjunction with minESS, such that the chain
    has sampled enough when multiESS >= minESS.
    
    For more information regarding these functions, see the R documentation.

    Parameters
    ----------
    chain : numpy array
        The MCMC chain.
    covmat : numpy array, optional
        The covaraince matrix for the parameters, if available. If None,
        matrix is obtained from mcse_multi. The default is None.
    g : function, optional
        --NOT YET IMPLEMENTED--. A function that represents features of
        interest. `g` is applied to each row of x, and should thus take a
        vector input only. If g is none, g is set to be identity, which is
        estimation of the mean of the target density. The default is None.
    mcse_multi_args : dict
        Arguments for mcse_multi function. Don't use this if a suitable matrix
        estimate from mcse_multi or mcse_initseq is already obtained. The
        default is an empty dictionary

    Returns
    -------
    ess : int
        The estimated effective sample size.

    '''
    # if g is not None:
    #     chain = np.array([g(chain[i, :]) for i in range(chain.shape[0])])
    
    n = chain.shape[0]
    chain = chain.reshape(n, -1)
    p = chain.shape[1]
    var_mat = np.cov(chain.T, ddof=1)
    if covmat is None:
        covmat = mcse_multi(chain, **mcse_multi_args)['cov']
    det_var_p = np.prod(np.linalg.eigvals(var_mat)**(1/p))
    det_covmat_p = np.prod(np.linalg.eigvals(covmat)**(1/p))
    ess = n * (det_var_p/det_covmat_p)
    return ess


def _raftery(chain, q=0.025, r=0.005, s=0.95, converge_eps=0.001,
             thin=1, print_=False):
    '''
    

    Parameters
    ----------
    chain : numpy array
        MCMC chain.
    q : float, optional
        Quantile of interest (in terms of percentile, i.e. between 0 and 1).
        The default is 0.025.
    r : float, optional
        Accuracy. The default is 0.005.
    s : float, optional
        Probability. The default is 0.95.
    converge_eps : float, optional
        Convergence threshold (epsilon). The default is 0.001.
    thin : int, optional
        Thining amount. The default is 1.
    print_ : bool, optional
        Print results. The default is False.

    Raises
    ------
    ValueError
        Raised if there are not enough samples in the chain, given the q,r,s
        values, or if there is an invalid selection of q.

    Returns
    -------
    None.

    '''

    if not q > 0 or not q < 1:
        raise ValueError("q must be between 0 and 1")
    else:
        # forces chain to be shape (:,1) if it is a single parameter chain
        chain = chain.reshape(chain.shape[0], -1)
        niter, nvar = chain.shape
        columns = ["Burn-in (M)", "Total (N)", "Lower Bound (Nmin)",
                   "Dependence Factor (I)", "Thinning (k)"]
        resmatrix = np.empty(shape=(nvar, len(columns)))
        resmatrix[:] = np.nan
        # nmin based on sample size calculation for proportions
        phi = qnorm_aprox(0.5 * (1 + s))
        nmin = np.ceil((q * (1 - q) * phi**2)/r**2).astype(int)
        if (nmin > niter):
            raise ValueError("Error: You either need to adjust q, r, or " +
                             " s, or you need a longer chain (at least " +
                             str(nmin) + " iterations)" +
                             "\r\nInput Parameters: \r\n" +
                             "\tQuantile (q): " + str(q) + "\r\n" +
                             "\tAccuracy (r): +/- " + str(r) + "\r\n" +
                             "\tProbability (s): " + str(s)
                             )
        else:
            for i in range(nvar):
                # U_t = value of the parameter at iteration t
                # u = value of parameter at defined quantile q
                quant = np.percentile(chain[:, i], q=q*100)
                dichot = chain[:, i] <= quant

                kthin = 0
                bic = 1
                testres = None
                newdim = 0
                # To find k ...
                while bic >= 0:
                    kthin = kthin + thin
                    # Z_t, the indicater function, or a dichotomus
                    # variable, representing where U_t <= u
                    testres = dichot[::kthin]
                    newdim = len(testres)

                    testres = pd.Categorical(testres,
                                             categories=[False, True])
                    testtran = pd.crosstab(index=testres[0:(newdim - 2)],
                                           columns=[testres[2:(newdim)],
                                           testres[1:(newdim - 1)]],
                                           margins=False, dropna=False)
                    g2 = 0

                    # First order MC vs. second order MC test (log
                    # likelihood ratio statistic, Bishop, Fienberg and
                    # Holland (1975))
                    for i1 in range(2):
                        for i2 in range(2):
                            for i3 in range(2):
                                if testtran[i1][i2][i3] != 0:
                                    fitted = (float(np.sum(testtran[i1][i2][:]))
                                              * np.sum(testtran[:][i2][i3])
                                              ) / (np.sum(
                                                     np.sum(testtran[:][i2][:])
                                                     ))
                                    g2 = g2 + testtran[i1][i2][i3] * np.log(
                                                 testtran[i1][i2][i3]/fitted
                                                 ) * 2.0
                    bic = g2 - np.log(newdim - 2.0) * 2.0

                finaltran = pd.crosstab(testres[0:(newdim - 1)],
                                        testres[1:(newdim)], dropna=False)
                alpha = finaltran[1][0]/float(finaltran[0][0] + finaltran[1][0])
                beta = finaltran[0][1]/float(finaltran[0][1] + finaltran[1][1])
                tempburn = (np.log((converge_eps * (alpha + beta))
                            / max([alpha, beta]))
                            / (np.log(np.abs(1.0 - alpha - beta)))
                            )

                # M = M * k
                nburn = np.ceil(tempburn) * float(kthin)
                tempprec = (((2.0 - alpha - beta) * alpha *
                             beta * phi**2) /
                            (((alpha + beta)**3) * r**2))
                nkeep = np.ceil(tempprec) * kthin
                # (M+N) / Nmin, which is the increase in the number of
                # iterations due to dependence in the sequence.
                # If I > 1 by a large amount, there is a high level of
                # dependence (rule of thumb: > 5 indicate problems)
                # Problems could be due to bad starting values, high
                # posterior correlations (which are remedied by
                # transformations that remove correlations), or
                # "stickiness" in the chain ( could be resolved by changing
                # MCMC algorithm)
                iratio = (nburn + nkeep)/nmin
                resmatrix[i, 0] = nburn
                resmatrix[i, 1] = nkeep + nburn
                resmatrix[i, 2] = nmin
                resmatrix[i, 3] = np.round(iratio, 2)
                resmatrix[i, 4] = kthin
        df = pd.DataFrame(data=resmatrix, columns=columns)
        inputs = {"r": r, "s": s, "q": q}

        if print_:
            print("\r\nInput Parameters: ", "\r\n",
                  "\tQuantile (q): ", inputs["q"], "\r\n",
                  "\tAccuracy (r): +/-", inputs["r"], "\r\n",
                  "\tProbability (s): ", inputs["s"], "\r\n",
                  "\r\n", df)
        return(df)


class mcmcObject(object):
    '''
    An object to hold MCMC chains, and to store/compute useful metrics on them.
    Also has some common plotting functionality.
    
    To get a chain up and running, run the mcmcWithRaftery method after
    instantiating the mcmcObject class.
    
    '''
    def __init__(self, name="MCMC Object"):
        '''
        Constructor for MCMC object

        Parameters
        ----------
        name : str, optional
            The name of the object (in case you instantiate multiple objects).
            The default is "MCMC Object".

        Returns
        -------
        None.

        '''
        self.name = name
        self.chains = {}
        self.diagnostic_results = {}
        self.previous_values = {}

    def addChain(self, newChain, chainName=None, concat=False):
        '''
        Adds a chain to your collection of chains.

        Parameters
        ----------
        newChain : numpy array
            New chain you would like to add to the collection.
        chainName : str, optional
            The name of the added chain, used to seperate it from others in the
            collection. If None, one will be selected for you.
            The default is None.
        concat : bool, optional
            Whether or not to append the new chain to an existing chain.
            The default is False.

        Returns
        -------
        None.

        '''
        if chainName is None:
            chainName = ''.join(("Chain_", str(len(self.chains) + 1)))

        if not isinstance(newChain, np.ndarray):
            try:
                newChain = np.array(newChain)
            except (ValueError, IndexError, KeyError, TypeError):
                print("Error: Please convert new chain object to Numpy Array")

        if not concat:
            self.chains[chainName] = newChain
        else:
            try:
                self.chains[chainName] = np.concatenate((self.chains[chainName],
                                                         newChain))
            except (NameError, KeyError):
                warnings.warn('''Failed to concatenate chains. Created new
                              chain instead. Check list of chain keys.''')
                self.chains[chainName] = newChain

    def removeChain(self, chainName, print_=True):
        '''
        Remove a chain from the collection.

        Parameters
        ----------
        chainName : str
            Chain to remove.
        print_ : bool, optional
            Print a message displaying what was dropped. The default is True.

        Returns
        -------
        None.

        '''
        try:
            del self.chains[chainName]
            if print_:
                print("Chain called " + chainName + " removed")
        except KeyError:
            print("No chain named ", chainName)

    def showChain(self, chainName):
        '''
        Display a chain from the collection.

        Parameters
        ----------
        chainName : str
            The name of the chain from the collection to display.

        Returns
        -------
        None.

        '''
        try:
            print(self.chains[chainName])
        except KeyError:
            print("No chain named ", chainName)

    def burnin(self, chainName, burninVal=3000, replace=False):
        '''
        Remove values from chain through burnin process (i.e. remove frist
        burninVal samples)

        Parameters
        ----------
        chainName : str
            The name of the chain from the collection.
        burninVal : int, optional
            The number of samples to remove. The default is 3000.
        replace : bool
            If True, replace the current chain rather than create a new one
            without the burnin samples. If a new chain is created, it will
            be called chainName + '_burnin' + burnunVal, and will be available
            in the collection.

        Returns
        -------
        None.

        '''
        if replace:
            self.chains[chainName] = self.chains[chainName][burninVal:, :]
        else:
            self.chains[chainName+'_burnin'+str(burninVal)] = self.chains[chainName][burninVal:, :]

    def bestRaftery(self, chainName, q=[0.025, 0.5, 0.975],
                    r=0.005, s=0.90, converge_eps=0.001, thin=1,
                    print_each=False, print_final=False):
        '''
        Run multiple Raftery evaluations and compare. The final raftery output
        is the maximum value for that criteria obtained from all Raftery
        evaluations.

        Parameters
        ----------
        chainName : str
            The name of the chain from the collection.
        q : float, optional
            Quantiles of interest (in terms of percentiles, i.e. between 0 and
                                  1).
            The default is [0.025, 0.5, 0.975].
        r : float, optional
            Accuracy. The default is 0.005.
        s : float, optional
            Probability. The default is 0.95.
        converge_eps : float, optional
            Convergence threshold (epsilon). The default is 0.001.
        thin : int, optional
            Thining amount. The default is 1.
        print_each : bool, optional
            Print results at each evaluation. The default is False.
        print_final : bool, optional
            Print the final results. The default is False.

        Returns
        -------
        None.

        '''
        q = list(q)
        columns = ["Burn-in (M)", "Total (N)", "Lower Bound (Nmin)",
                   "Dependence Factor (I)", "Thinning (k)", "Quantile",
                   "Parameter"]
        all_samples = pd.DataFrame(columns=columns)
        for i in range(len(q)):
            needed_size = samp_size_calc_raftery(q[i], r, s)[1]
            if needed_size > len(self.chains[chainName]):
                print("not enough samples in the chain for quantile " +
                      str(q[i]) + ". Could not evaluate.")
            else:
                res = _raftery(self.chains[chainName], q=q[i], r=r, s=s,
                               converge_eps=converge_eps, thin=thin,
                               pr=print_each)
                res["Quantile"] = q[i]
                res['Parameter'] = res.index
                all_samples = all_samples.append(res, ignore_index=True)

        maxcols = ["Max Burn-in (M)", "Max Total (N)",
                   "Max Lower Bound (Nmin)", "Max Dependence Factor (I)",
                   "Max Thinning (k)"]
        maxvals = np.max(all_samples.iloc[:, :-2])
        finalres = pd.DataFrame(np.array(maxvals).reshape(1, 5),
                                columns=maxcols)
        inputs = {"r": r, "s": s, "q": q}

        if print_final:
            print("\r\nInput Parameters: ", "\r\n",
                  "\tQuantile (q): ", inputs["q"], "\r\n",
                  "\tAccuracy (r): +/-", inputs["r"], "\r\n",
                  "\tProbability (s): ", inputs["s"], "\r\n",
                  "\r\n", finalres)
        self.diagnostic_results[chainName + "_Raftery"] = finalres

    def TJ_Convergence_test(self, chainName, eps=0.025, quantiles=[0.05, 0.95],
                            window_size=None, num_windows=5, slide=50,
                            window_space=0, bin_limit=0.6, print_final=False):
        '''
        A homemade test to evaluate convergence. This test evaluates a moving
        window, or a list of moving windows and compares the values of the
        distribution tails in those windows. If the distribution tails of all
        the moving windows is in line with the distribution tails of the final
        n samples of the chain, then the chain is considered to have converged.

        Parameters
        ----------
        chainName : str
            The name of the chain from the collection.
        eps : float, optional
            The threshold for comparing similarity in the chain. If the
            similarity between the moving window distributions and the
            distribution at the end of the chain is below the threshold, then
            the chain is considered to have converged. The default is 0.025.
        quantiles : list, or list-like, optional
            The sections of the distributions to consider for similarity.
            The default is [0.05, 0.95].
        window_size : int, optional
            How many chain samples to consider in each window. If None,
            it is automatically determined based on the size of the chain.
            The default is None.
        num_windows : int, optional
            The number of moving windows to use in the evaluation.
            The default is 5.
        slide : int, optional
            The number of samples to slide after each iteration, or in other 
            words, how fast the moving windows move (a larger value for slide
            means less total evaluations). The default is 50.
        window_space : int, optional
            The distance between each window. If positive, there is a gap. If
            negative, there is overlap. The default is 0.
        bin_limit : float, optional
            The bin_limit is the percent (between 0 and 1) of samples
            that are used in the moving windows, or in other words, the 
            evaluation stops once the right edge (i.e. most recently sampled 
            observation in the chain) matches the upper limit determined by
            the bin_limit percent. Also, 1 - bin_limit is the amount of samples
            from the end that are used as a baseline to see if the chain has
            converged. The default is 0.6.
        print_final : bool, optional
            Print the final results. The default is False.

        Raises
        ------
        ValueError
            Raises ValueError if argumnets passed by the user are outside
            function constraints.

        Returns
        -------
        None.

        '''
        chain = np.array(self.chains[chainName])
        # eps must be greater than 0
        # window size needs to be a number between 1 and
        # bin_limit of the length of the chain
        # bin_limit must be between 0 and 1
        if float(bin_limit) <= 0 or float(bin_limit) >= 1:
            raise ValueError("bin_limit must be between 0 and 1")
        if float(eps) <= 0:
            raise ValueError("""eps must be greater than 0 (its good to pick a
                                decimal close to zero, but not equal to it)""")
        if np.round(float(num_windows)) <= 0:
            raise ValueError("num_windows must be greater than 0")

        nrows = chain.shape[0]
        chain = chain.reshape(nrows, -1)
        ncols = chain.shape[1]

        if window_size is None:
            window_size = np.ceil(0.05*nrows)
        elif float(window_size) < 1 or float(slide) < 1:
            raise ValueError("window_size or slide must be at least 1")

        max_row = np.ceil(nrows * bin_limit)
        init_right_edge = (window_size * num_windows) + \
                          (window_space * (num_windows - 1))
        bins = np.floor((max_row - init_right_edge) / slide)

        col_names = ["Burn-in", "Ending Bin", "Total Bins", "Status",
                     "Notes", "Msg Details"]
        res = pd.DataFrame(columns=col_names)

        for col in range(ncols):
            cur_chain = chain[:, col]
            end_cur_chain = cur_chain[int(max_row):]
            quantiles.sort()
            end_per = np.percentile(end_cur_chain, 100*np.array(quantiles))
            bin_i = 1
            per_ratios = np.ones((num_windows, len(quantiles)))*(2+eps)
            bin_mat = np.arange(num_windows)

            if (num_windows == 1):
                index_vals = np.array([0.0])
            else:
                index_vals = (bin_mat * window_size) + (bin_mat * window_space)

            while (np.any(np.abs(per_ratios - 1) >= eps)
                    and bin_i <= bins):
                for i in range(len(index_vals)):
                    window_chain = cur_chain[
                            int(index_vals[i]):int(index_vals[i]+window_size)]
                    per_ratios[i, :] = np.percentile(
                            window_chain, 100*np.array(quantiles)) / end_per

                index_vals += slide
                bin_i += 1

            if bin_i > bins:
                msg = "Unsuccessful"
                sub_msg = """Fully iterated before meeting criteria, may not
                have stabilized on a distribution. Try adjusting the settings
                and try again, or take a look at the plot"""
            elif np.any(np.abs(per_ratios - 1) >= eps):
                msg = "Unsuccessful"
                sub_msg = """Did not appear to stabilize on a distribution.
                          Try adjusting the settings and try again, or take a
                          look at the plot"""
            else:
                msg = "Successful"
                sub_msg = ""
            msg_display = "See Msg Details in DataFrame"

            if num_windows % 2 == 1:
                burnin = index_vals[
                        int(np.median(bin_mat))] + (np.ceil(window_size / 2))
            else:
                burnin = index_vals[
                        int(np.floor(np.median(bin_mat)))] + window_size

            res = res.append(pd.DataFrame([[burnin, bin_i, bins, msg,
                                            msg_display, sub_msg]],
                                          columns=col_names),
                             ignore_index=True)

        inputs = {"eps": eps, "window_size": window_size,
                  "num_windows": num_windows, "slide": slide,
                  "window_space": window_space, "bin_limit": bin_limit}

        if print_final:
            print("\r\nInput Parameters: ", "\r\n",
                  "\tRatio Epsilon: ", inputs["eps"], "\r\n",
                  "\tWindow Size: ", inputs["window_size"], "\r\n",
                  "\tNumber of Windows: ", inputs["num_windows"], "\r\n",
                  "\tSlide Amount: ", inputs["slide"], "\r\n",
                  "\tSpace between Windows: ", inputs["window_space"], "\r\n",
                  "\tBin Limit Percentage: ", inputs["bin_limit"], "\r\n",
                  "\r\n", res.loc[:, res.columns != "Msg Details"])

        self.diagnostic_results[chainName + "_Convtest"] = res

    def runMCMC(self, start, initSampleSize, lpost, algo, algoOpts=None,
                raftOpts=None, chainName=None, max_tries=100, sd=0.02,
                plot_t=True, plot_d=True, write_plots=True,
                plot_dir='./Plots/', plot_d_smoothing=0.05, plot_d_vlines=None,
                do_raftery=True, iters_to_go=None, max_iters=750000,
                burnin=0, lpost_args={}):
        '''
        TODO: Update to include ESS option, instead of just raftery.
        
        Generate MCMC samples and evaluate samples size (using Raftery)
        and convergence.
        
        Parameters
        ----------
        start : numpy array
            Starting values for the MCMC.
        initSampleSize : int
            The number of MCMC samples to draw on the first run. It's good to 
            start relatively small, because the Raftery evaluation will tell
            you how many more samples need to be drawn.
        lpost : function
            Log posterior function.
        algo : function
            The MCMC algorithm to use (could be anything, but needs to have the
        same arguments as inputs for the algorithms already defined, namely:
            start : numpy array
                Starting values for the MCMC.
            niter : int
                Number of iterations.
            lpost : function
                Log posterior function.
            postArgs : dict
                Extra arguments for the log posterior function. The default is
                an empty dictionary
            options : dict, optional
                Extra arguments for the specific MCMC algorithm
        algoOpts : dict, optional
            Extra arguments for the specific MCMC algorithm.
            The default is None.
        raftOpts : dict, optional
            A dictionary containing the options for the Raftery evaluation, 
            namely:
                q : float, optional
                    Quantiles of interest (in terms of percentiles, i.e.
                                           between 0 and 1).
                    The default is [0.025, 0.5, 0.975].
                r : float, optional
                    Accuracy. The default is 0.005.
                s : float, optional
                    Probability. The default is 0.95.
                converge_eps : float, optional
                    Convergence threshold (epsilon). The default is 0.001.
                thin : int, optional
                    Thining amount. The default is 1.
                print_each : bool, optional
                    Print results at each evaluation. The default is False.
                print_final : bool, optional
                    Print the final results. The default is False.
            The default is None.
        chainName : str, optional
            The name of the chain that will be created. If None,
            'Chain_' + int (for number of chains in the collection) 
            will be used. The default is None.
        max_tries : int, optional
            The max number of times to try and jitter before admitting defeat.
            If the jitter fails, the reason or the covaraince matrix not being
            positive definite may not be due to randomness, and may require
            a re-evaluation of the problem space. The default is 100.
        sd : float, optional
            The standard deviation of the normal distribution used to draw the
            jitter amount from. In other words, the jittered covariance is the 
            covaraince matrix plus a random draw X, where X~N(0, sd). 
            The default is 0.02.
        plot_t : bool, optional
            Plot the trace of the MCMC samples. The default is True.
        plot_d : bool, optional
            Plot the posterior density of the MCMC samples.
            The default is True.
        write_plots : bool, optional
            Save any plots to a directory. The default is True.
        plot_dir : str, optional
            The directory to save any plot to. The default is './Plots/'.
        plot_d_smoothing : float, optional
            The amount of smoothing to use on the kde plot.
            See seaborn.kde_plot for details.
            The default is 0.05.
        plot_d_vlines : numpy array, optional
            The x-axis locations of any predetermined vertical lines on the
            density plots.
            
            Generally, this is not helpful when creating the
            first MCMC chain. However, after generating an initial run,
            it can be useful to call the plot_density method
            on it's own and include the location of the mean, mode, etc.
            The default is None.
        do_raftery : bool, optional
            Whether to perform the raftery evaluation, or stop after the
            first chain generation. Default is True.
        max_iters : int, optional
            The max number of new samples to draw. For example, if the Raftery
            evaluation recommends 1 million new samples, and max_iters is 
            750000, then the new samples will be restricted at 750000.
            The default is 750000.
        burnin : int, optional
            The number of initial MCMC samples to drop from the chain.
            If burnin is negative or zero, the burnin amount will be determined
            automatically. If positive, it will drop that amount. If 
            None, it will do nothing. The default is 0.
        lpost_args : dict, optional
            Any extra arguments to pass to the log posterior function.
            The default is an empty dictionary.

        Returns
        -------
        None.

        '''
        if algoOpts is None:
            algoOpts = {}
        first_run_results = applyMCMC(st=start, ni=initSampleSize, lp=lpost,
                                      algo=algo, algoOpts=algoOpts,
                                      postArgs=lpost_args,
                                      sd=sd, max_tries=max_tries)
        previous_values = first_run_results[0][1]
        new_start = first_run_results[0][0][-1, :]
        self.addChain(first_run_results[0][0], chainName, concat=False)
        self.previous_values[chainName + "_latestrun"] = previous_values

        if do_raftery:
            qq = [0.025, 0.5, 0.975]
            rr = 0.005
            ss = 0.90
            ce = 0.001
            th = 1
            pe = False
            pf = False
            if raftOpts is not None:
                keys = list(raftOpts.keys())
                if 'q' in keys:
                    qq = raftOpts['q']
                if 'r' in keys:
                    rr = raftOpts['r']
                if 's' in keys:
                    ss = raftOpts['s']
                if 'converge_eps' in keys:
                    ce = raftOpts['converge_eps']
                if 'thin' in keys:
                    th = raftOpts['thin']
                if 'print_each' in keys:
                    pe = raftOpts['print_each']
                if 'print_final' in keys:
                    pf = raftOpts['print_final']
            
            self.bestRaftery(chainName, q=qq, r=rr, s=ss, converge_eps=ce,
                             thin=th, print_each=pe, print_final=pf)
            raftmin = int(self.diagnostic_results[chainName + "_Raftery"]["Max Total (N)"].values)
            initial_len = self.chains[chainName].shape[0]
            iters_to_go = raftmin - initial_len
            
            if iters_to_go >= 1:
                if iters_to_go >= max_iters:
                    iters_to_go = max_iters
                pv = self.previous_values[chainName + "_latestrun"]
                algoOpts.update({'prev_vals': pv})
                final_run_results = applyMCMC(st=new_start,
                                              ni=int(iters_to_go),
                                              lp=lpost,
                                              algo=algo,
                                              algoOpts=algoOpts,
                                              postArgs=lpost_args,
                                              sd=sd,
                                              max_tries=max_tries)
                previous_values = final_run_results[0][1]
                self.addChain(final_run_results[0][0], chainName, concat=True)
                self.previous_values[chainName + "_latestrun"] = previous_values

        burnin_param = None
        burnVal = 0
        if burnin is not None:
            if burnin <= 0:
                self.TJ_Convergence_test(chainName)
                conv_diag = self.diagnostic_results[chainName + "_Convtest"]
                # burnin_param gets used in plotting later. It's possible for each
                # parameter to have a seperate optimal burn-in point, however,
                # the parameters themselves should not be considered 
                # independently, and thus, need a common burn-in value
                burnin_param = np.array(conv_diag["Burn-in"])
                burnVal = int(conv_diag["Burn-in"].max())
            else:
                # burnin_param is for plotting purposes only
                burnin_param = np.repeat(burnin, len(start))
                burnVal = burnin

        if plot_t:
            self.plotTrace(chainName, CTres=burnin_param, write=write_plots,
                           pdir=plot_dir)
        if plot_d:
            self.plotDensity(chainName, smoothing=plot_d_smoothing,
                             write=write_plots, pdir=plot_dir,
                             vlines=plot_d_vlines)

        self.burnin(chainName, burnVal)

    def plotTrace(self, chainName, CTres=None, write=True, pdir="./Plots/",
                  fileType="jpg", figsize=(15, 12)):
        '''
        Plot the trace of the MCMC chain.

        Parameters
        ----------
        chainName : str
            The name of the MCMC chain.
        CTres : numpy array, optional
            The results of the TJ_Convergence_Test. The default is None.
        write : bool, optional
            Write plot to a directory. The default is True.
        pdir : str, optional
            The directory to write the plots to. The default is "./Plots/".
        fileType : str, optional
            The filetype of the image. The default is "jpg".
        figsize : tuple, optional
            The figure size (see matplotlib documentation for more details).
            The default is (15, 12).

        Returns
        -------
        None.

        '''
        trace = self.chains[chainName]
        trace = trace.reshape(trace.shape[0], -1)
        nparam = trace.shape[1]
        fig, ax = plt.subplots(nrows=nparam, ncols=1, figsize=figsize,
                               squeeze=False)
        for i in range(nparam):
            ax[i, 0].plot(trace[:, i])
            ax[i, 0].set_ylabel(''.join(["Trace Dis: ", self.name,
                                         ", Run Param: ", str(i+1), "/",
                                         str(nparam)]))
            ax[i, 0].set_xlabel("Iteration")
            if CTres is not None:
                ax[i, 0].axvline(x=int(CTres[i]), color="red",
                                 linewidth=2.0)
        if write:
            pathlib.Path(pdir).mkdir(exist_ok=True)
            fig.savefig(''.join([pdir, self.name, '_', chainName,
                                 "_trace.", fileType]))
        else:
            plt.show(fig)
        plt.close(fig)

    def plotDensity(self, chainName, smoothing=0.05, write=True,
                    pdir="./Plots/", vlines=None, fileType="jpg",
                    figsize=(15, 12)):
        '''
        Plot the density of the MCMC chain.

        Parameters
        ----------
        chainName : str
            The name of the MCMC chain.
        smoothing : float, optional
            The amount of smoothing to use on the kde plot.
            See seaborn.kde_plot for details.
            The default is 0.05.
        write : bool, optional
            Write plot to a directory. The default is True.
        pdir : str, optional
            The directory to write the plots to. The default is "./Plots/".
        vlines : TYPE, optional
            The x-axis locations of any predetermined vertical lines on the
            density plots, such as mean, median, or mode.
            The default is None.
        fileType : str, optional
            The filetype of the image. The default is "jpg".
        figsize : tuple, optional
            The figure size (see matplotlib documentation for more details).
            The default is (15, 12).

        Returns
        -------
        None.

        '''
        trace = self.chains[chainName]
        trace = trace.reshape(trace.shape[0], -1)
        nparam = trace.shape[1]
        fig, ax = plt.subplots(nrows=1, ncols=nparam, figsize=figsize,
                               sharey=True, squeeze=False)
        for i in range(nparam):
            sns.kdeplot(trace[:, i], ax=ax[0, i], shade=True)
            ax[0, i].set_ylabel("Density")
            ax[0, i].set_xlabel(''.join(["Trace Results: ", self.name,
                                         ", Run Param: ",
                                         str(i+1), "/", str(nparam)]))
            if vlines is not None:
                ax[0, i].axvline(vlines[i])

        if write:
            pathlib.Path(pdir).mkdir(exist_ok=True)
            fig.savefig(''.join([pdir, self.name, '_', chainName, "_density.",
                                 fileType]))
        else:
            plt.show(fig)
        plt.close(fig)

    def plotACF(self, chainName, bounds=True, ci=0.95, acfType="acf",
                write=False, pdir="./Plots/", fileType="jpg", lw=None,
                figsize=(15, 12)):
        '''
        Plot the Autocorrelation function of the chain.

        Parameters
        ----------
        chainName : str
            The name of the MCMC chain.
        bounds : bool, optional
            Draw the bounds of the autocorrelation. The default is True.
        ci : float, optional
            The size of the bounds (confidence interval), if applicable.
            The default is 0.95.
        acfType : str, optional
            The type of acf plot to draw. Can be either 'acf' or 'pacf'.
            The default is "acf".
        write : bool, optional
            Write plot to a directory. The default is True.
        pdir : str, optional
            The directory to write the plots to. The default is "./Plots/".
        fileType : str, optional
            The filetype of the image. The default is "jpg".
        lw : float, optional
            The line width to use on the plot. If None, it will be calculated
            automatically. The default is None.
        figsize : tuple, optional
            The figure size (see matplotlib documentation for more details).
            The default is (15, 12).

        Returns
        -------
        None.

        '''
        try:
            self.diagnostic_results[chainName + "_" + acfType]
        except KeyError:
            print("""No ACF found. Please calculate ACF, pACF,
                  or a variant using available methods""")
        allacf = self.diagnostic_results[chainName + "_" + acfType]
        allacf = allacf.reshape(allacf.shape[0], -1)
        (samples, nparam) = allacf.shape
        fig, ax = plt.subplots(nrows=nparam, ncols=1, figsize=figsize,
                               squeeze=False)
        # Picked this as the line width value because it seems to pick a good
        # width with respect to the lag number. the plot is designed to mimic
        # the R acf() function
        if lw is None:
            lw = 1-np.exp(-0.00346103*(samples-1))
        for i in range(nparam):
            ax[i, 0].bar(range(len(allacf[:, i])), allacf[:, i], width=lw)
            ax[i, 0].set_ylabel(''.join([acfType, " Dis: ", self.name,
                                         ", Run Param: ", str(i+1), "/",
                                         str(nparam)]))
            ax[i, 0].set_xlabel("Lag")
            ax[i, 0].axhline(y=0, linewidth=0.5)
            if bounds:
                bnd = qnorm_aprox((1+ci)/2)/np.sqrt(samples)
                ax[i, 0].axhline(y=bnd, color="red",
                                 linestyle="dashed", linewidth=0.5)
                ax[i, 0].axhline(y=-bnd, color="red",
                                 linestyle="dashed", linewidth=0.5)
        if write:
            pathlib.Path(pdir).mkdir(exist_ok=True)
            fig.savefig(''.join([pdir, self.name, '_', chainName, acfType, ".", fileType]))
        else:
            plt.show(fig)
        plt.close(fig)

	
    def acf(self, chainName, lag=50, partial=False, demean=True):
        '''
        ACF definition for a wide-sense stationary process, partial acf uses
        Yule-Walker MLE method.

        Parameters
        ----------
        chainName : str
            The name of the MCMC chain.
        lag : int, optional
            The lag in the autocorrelation. The default is 50.
        partial : bool, optional
            Calculate pACF instead of ACF. The default is False.
        demean : bool, optional
            Center the chain before calculating autocorrelation.
            The default is True.

        Returns
        -------
        None.

        '''
        trace = self.chains[chainName]
        trace = trace.reshape(trace.shape[0], -1)
        params = trace.shape[1]
        samples = trace.shape[0]
        allacf = np.zeros((lag + 1, params))
        for param in range(params):
            x = trace[:, param]
            if demean:
                center = x - x.mean()
            else:
                center = x
            acf = np.zeros(lag + 1)
            acf[0] = 1
            z = np.sum(center**2)/samples ## variance of x
            for l in range(1, lag+1):
                if partial:
                    r = np.zeros(l+1)
                    r[0] = z
                    for k in range(1, l+1):
                        headlag = center[k:]
                        taillag = center[:-k]
                        r[k] = np.sum(headlag*taillag)/(samples-k)
                    r[l] = np.sum(headlag*taillag)/(samples-l)
                    R = toeplitz(r[:-1])
                    rho = solve(R, r[1:])
                    #sigma2 = r[0] - (r[1:]*rho).sum()
                    acf[l] = rho[-1]
                else:
                    headlag = center[l:]
                    taillag = center[:-l]
                    acf[l] = np.sum(headlag*taillag)/samples/z
            allacf[:, param] = acf
        if partial:
            self.diagnostic_results[chainName + "_pacf"] = allacf
        else:
            self.diagnostic_results[chainName + "_acf"] = allacf
                    