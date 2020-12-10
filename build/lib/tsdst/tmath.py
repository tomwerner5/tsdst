from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)
from itertools import chain, combinations
from numpy.linalg import cholesky, inv, eig, LinAlgError
from scipy.stats import rankdata, kendalltau, gaussian_kde


def custom_abs(x):
    m = x % (x**2 - x + 2)
    p = m % (x**2 + 2)
    return 2*p - x


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
  

def percentIncrease(old, new):
    return (new - old)/old


# This will do a 'first' ranking of the data and return the ranks.
# For R-like rank functionality, import scipy's rankdata from 
# this module
def rank(x, small_rank_is_high_num=True, rank_from_1=True):
    rk = np.argsort(np.argsort(x))
    if small_rank_is_high_num:
        rk = (len(x) - 1) - rk
    if rank_from_1:
        rk += 1
    return rk


def mann_whitney(x, y, y_is='groups'):
    if isinstance(y, (list, tuple)):
        y = np.array(y)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    if y_is == 'groups':
        X = x
        Y = y
        overall_rank = rankdata(X, 'average')
    else:
        if len(x.shape) >= 2:
            x = x.ravel()
        if len(y.shape) >= 2:
            y = y.ravel()
        X = np.concatenate((x, y))
        Y = np.concatenate((np.repeat(0, x.size), np.repeat(1, y.size)))
        overall_rank = rankdata(X, 'average')
    uni_groups = np.unique(Y)
    rank_group1 = overall_rank[Y == uni_groups[0]]
    rank_group2 = overall_rank[Y == uni_groups[1]]
    sum_ranks_g1 = rank_group1.sum()
    sum_ranks_g2 = rank_group2.sum()
    
    n1 = rank_group1.shape[0]
    n2 = rank_group2.shape[0]
    
    U1 = sum_ranks_g1 - (n1*(n1 + 1))/2
    U2 = sum_ranks_g2 - (n2*(n2 + 1))/2
    
    res = {'U1': U1,
           'U2': U2,
           'n1': n1,
           'n2': n2,
           'rank_group1': rank_group1,
           'rank_group2': rank_group2,
           'sum_ranks_g1': sum_ranks_g1,
           'sum_ranks_g2': sum_ranks_g2}
    
    res['max_U'] = 'U1' if U1 >= U2 else 'U2'
    return res


def biserial_rank_corr(x, groups, y_is='groups'):
    mw = mann_whitney(x, groups, y_is=y_is)
    max_u = mw[mw['max_U']]
    r = (2*max_u)/(mw['n1']*mw['n2']) - 1
    return r


# mimics R cor function. Columns are correlation variables.
# X and Y must at least have the same length
def corr(x, y=None, method='pearson'):
    cor_mat = None
    X = np.array(x, ndmin=2)
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if X.shape[0] == 1:
        X = X.T
    if y is not None:
        Y = np.array(y, ndmin=2)
        if Y.shape[0] == 1:
            Y = Y.T
        X = np.concatenate((X, Y), axis=1)
    if method == 'pearson':
        cor_mat = np.corrcoef(X, rowvar=False)
    elif method == 'spearman':
        M = np.apply_along_axis(rankdata, 0, X)
        D = np.eye(X.shape[1])*M.std(ddof=1, axis=0)
        Dinv = inv(D)
        cor_mat = Dinv.dot(np.cov(M, rowvar=False)).dot(Dinv)
    elif method == 'kendall':
        cor_mat = np.zeros((X.shape[1], X.shape[1]))
        corrs = []
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                c, _ = kendalltau(X[:, i], X[:, j])
                corrs.append(c)
        corrs = np.array(corrs)
        row, col = np.triu_indices(X.shape[1])
        cor_mat[row, col] = corrs
        cor_mat[col, row] = corrs
    else:
        raise ValueError('Not a valid correlation method. (%s) ' % method)
    
    if isinstance(x, pd.DataFrame):
        cor_mat = pd.DataFrame(cor_mat, index=x.columns, columns=x.columns)
    return cor_mat


def histogram_mode(data, delta=75, dataMax_thres=1e9, median_thres=None):
    bin_min = np.round(np.floor(np.min(data/delta))*delta, 3)
    bin_max = np.round(np.ceil(np.max(data) / delta + 1)*delta, 3)
    data_max = np.max(data)
    if median_thres is None:
        if data_max > dataMax_thres:
            bin_max = dataMax_thres
    else:
        data_median = np.median(data)
        if data_max < dataMax_thres:
            pass
        elif data_max >= dataMax_thres and data_median < median_thres:
            bin_max = np.round(np.ceil((2*data_median) / delta + 1) * delta, 3)
    bin_array = np.arange(bin_min, bin_max, delta)
    hist = np.histogram(data, bin_array)
    return hist[1][np.where(hist[0] == np.max(hist[0]))][0], hist


def mode_kde(data):
    kernel = gaussian_kde(data)
    height = kernel.pdf(data)
    mode = data[np.argmax(height)]
    return mode


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

        
def mahalanobis(data, produce=None):
    '''
    assumes that rows are observations and columns are features
    '''
    arr = np.array(data).reshape(data.shape[0], -1)
    cent = arr - arr.mean(axis=0)
    covmat = np.cov(cent, rowvar=False)
    invcov = None
    if arr.shape[1] == 1:
        invcov = 1/covmat
    else:
        try:
            invcov = np.linalg.inv(covmat)
        except np.linalg.LinAlgError:
            invcov = np.linalg.pinv(covmat)
    md2 = np.sum(cent.dot(invcov) * cent, axis=1)
    if produce == "squared":
        return md2
    elif produce == "leverage":
        n = data.shape[0]
        return ((md2/(n - 1)) + (1/n))
    else:
        return np.sqrt(md2)
        
        
def js_div(px, py):
    '''
    Jensen-Shannon Divergence
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    '''
    midpoint = (px + py)*0.5
    js = rel_entr(px, midpoint)*0.5 + rel_entr(py, midpoint)*0.5
    return np.sum(js)
