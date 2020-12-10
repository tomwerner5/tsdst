from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
import numpy as np
import random
import sys

from numba import jit
from numpy import linalg as la
from scipy import linalg as sla
from timeit import default_timer as dt

from tsdst.utils import pretty_print_time
    

def updateProgBarMCMC(curIter, totalIter, t0, ar, barLength=20):
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
        

def applyMCMC(st, ni, lp, algo, algoOpts=None, postArgs=None,
              sd=0.02, max_tries=100):
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
            print("Number of Cholesky tries: " + str(try_num))
        except np.linalg.LinAlgError:
            try_num += 1
        
        if try_num >= max_tries:
            raise ValueError("Cholesky Decomposition was not successful after " + str(max_tries) + " tries. Try new starting values")
    return res              


# For upper triangle rank one update
@jit
def cholupdate(L, x, update=True):
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


def adaptive_mcmc(start, niter, lpost, postArgs, options=None):
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


def rwm_with_lap(start, niter, lpost, postArgs, options=None):
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


def rwm(start, niter, lpost, postArgs, options=None):
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
        
        
                    