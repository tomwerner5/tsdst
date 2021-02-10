import numpy as np
import pandas as pd

from scipy.special import rel_entr
from sklearn.metrics import confusion_matrix, roc_auc_score
from statsmodels.tools import add_constant

from .distributions import (glm_likelihood_bernoulli, glm_likelihood_poisson,
                            glm_likelihood_gaussian)
from .utils import one_hot_decode, decision_boundary_1D


def top_20p(y_true, y_score):
    '''
    Measures the percent of true positives that are captured in the top 20%
    of the population (sorted by predicted probabilities).

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_score : numpy array
        Predicted probabilities.

    Returns
    -------
    top_20 : float
        Percent of true positives that are captured in the top 20% 
        of the population.

    '''
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    cutoff = int(0.2*y_true.shape[0])
    prob_sort = np.argsort(y_score)[::-1]
    top_20 = y_true[prob_sort][:cutoff].sum()/y_true.sum()
    return top_20


# For use with custom scorers in sklearn, otherwise, use top_20p
def top_20p_score(estimator, X, y, yprob=None):
    '''
    Measures the percent of true positives that are captured in the top 20%
    of the population (sorted by predicted probabilities).
    
    For use with custom scorers in sklearn, otherwise, use top_20p.

    Parameters
    ----------
    estimator : sklearn model, or similar
        The model. Needs to have a predict_proba method.
    X : numpy array or pandas dataframe
        The design or feature matrix.
    y : numpy array or pandas series
        The target or response variable.
    yprob : None
        For compatibility with sklearn custom scorers. Default is None.

    Returns
    -------
    top_20 : float
        Percent of true positives that are captured in the top 20% 
        of the population.

    '''
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_true = y.values
    else:
        y_true = y
    y_score = estimator.predict_proba(X)[:, 1]
    cutoff = int(0.2*y_true.shape[0])
    prob_sort = np.argsort(y_score)[::-1]
    top_20 = y_true[prob_sort][:cutoff].sum()/y_true.sum()
    return top_20


def auc_score(estimator, X, y):
    '''
    Area under the ROC curve. Wrapper for compatibility with sklearn
    custom scorers.

    Parameters
    ----------
    estimator : sklearn model, or similar
        The model. Needs to have a predict_proba method.
    X : numpy array or pandas dataframe
        The design or feature matrix.
    y : numpy array or pandas series
        The target or response variable.

    Returns
    -------
    auc : float
        The area under the roc curve.

    '''
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_true = y.values
    else:
        y_true = y
    y_score = estimator.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true, y_score, average='weighted')
    return auc


def cox_snell_r2(ll_est, ll_null, n_obs):
    '''
    Cox-Snell Pseudo R-squared.

    Parameters
    ----------
    ll_est : numpy array
        Estimated Log likelihood.
    ll_null : numpy array
        Null-model log-likelihood.
    n_obs : int
        Number of observations.

    Returns
    -------
    cs_r2 : float
        Cox-Snell Pseudo R-squared.

    '''
    ratio = 2/n_obs
    cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    return cs_r2


def nagelkerke_r2(ll_null, n_obs, cs_r2=None, ll_est=None):
    '''
    Nagelkerke Pseudo R-squared. Provides a correction to Cox-Snell Pseudo
    R-squared to bound between 0, 1.

    Parameters
    ----------
    ll_null : numpy array
        Null-model log-likelihood.
    n_obs : int
        Number of observations.
    cs_r2 : float
        Cox-Snell Pseudo R-squared. Default is None.
    ll_est : numpy array
        Only needed if cs_r2 not provided. Estimated Log likelihood.
        Default is None.

    Returns
    -------
    cs_r2 : float
        Nagelkerke Pseudo R-squared.

    '''
    ratio = 2/n_obs
    r2_max = 1 - np.exp(ratio*ll_null)
    if cs_r2 is None:
        cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    # Correction to CS R2 to bound between 0, 1
    n_r2 = cs_r2/r2_max
    return n_r2


def tjur_r2(y_true, y_score):
    '''
    Tjur Pseudo R-squared

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_score : numpy array
        Predicted probabilities.

    Returns
    -------
    t_r2 : float
        Tjur Pseudo R-squared.

    '''
    y_mu1 = y_score[y_true == 1].mean()
    y_mu0 = y_score[y_true == 0].mean()
    t_r2 = np.abs(y_mu1 - y_mu0)
    return t_r2


def mcfadden_r2(ll_est, ll_null):
    '''
    McFadden's Pseudo R-squared when the saturated model is not available.

    Parameters
    ----------
    ll_est : numpy array
        Estimated Log likelihood.
    ll_null : numpy array
        Null-model log-likelihood.

    Returns
    -------
    m_r2 : float
        McFadden Pseudo R-squared.

    '''
    m_r2 = 1 - (ll_est/ll_null)
    return m_r2


def number_of_nonzero_coef(X, model):
    '''
    Calculates the number of non-zero coefficients in a model. Useful for
    evaluating models with an L1 penalty.

    Parameters
    ----------
    X : numpy array or pandas dataframe
        The design or feature matrix.
    model : sklearn object, or similar
        The model, which needs to have a coef_ attribute.

    Returns
    -------
    num_coef : int
        Number of non-zero coefficients.

    '''
    num_coef = None
    try:
        num_coef = sum((True if c != 0 else False for c in model.coef_.reshape(-1,)))
    except:
        num_coef = X.shape[1]
    return num_coef


def conf_mat_metrics(y_true, y_pred, conf_metric='all'):
    '''
    Confusion Matrix metrics for 0/1 class classification. Assumes 1 is the
    positive class (i.e. 1 == true positive, 0 == true negative).

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted 0/1 values.
    conf_metric : str, optional
        The metrics to return, can be `Sens/Recall`, `Specificity`, `ppv`,
        `npv`, `conf_mat`, or `all`. The default is 'all'.

    Returns
    -------
    dict, or float
        Returns dictionary for `all`, otherwise, float for the metric of
        interest.

    '''
    confMat = confusion_matrix(y_true, y_pred)
    metrics = {'Sens/Recall': 0,
               'Specificity': 0,
               'ppv': 0,
               'npv': 0,
               'Conf_Mat': 0}
    tn = confMat[0, 0]
    tp = confMat[1, 1]
    fn = confMat[0, 1]
    fp = confMat[1, 0]
    tnfp = tn + fp
    tpfn = tp + fn
    tpfp = tp + fp
    tnfn = tn + fn
    if tnfp != 0:
        metrics['Specificity'] = tn/tnfp
    if tpfn != 0:
        metrics['Sens/Recall'] = tp/tpfn
    if tpfp != 0:
        metrics['ppv'] = tp/tpfp
    if tnfn != 0:
        metrics['npv'] = tn/tnfn
    if tpfp*tpfn*tnfp*tnfn != 0:
        metrics['matthews_coef'] = ((tp*tn) - (fp*fn))/np.sqrt(tpfp*tpfn*tnfp*tnfn)
    metrics['Conf_Mat'] = confMat
    
    if conf_metric == 'all':
        return metrics
    else:
        return metrics[conf_metric]     

    
def bias(y_true, y_pred):
    '''
    Calculate model bias (systemic errors in estimation) in Linear Regression.
    
    Defined as The average distance (and direction) the predictions are from
    the true values (mean(y_pred - y_true)). If bias < 0, predictions are too
    low, and if bias > 0, then predictions are too high. Bias is measured in 
    the same units as the response.

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted values.

    Returns
    -------
    bias : float
        Model bias.

    '''
    ydiff = y_pred - y_true
    bias = np.mean(ydiff)
    return bias


def rpmse(y_true, y_pred, root=True):
    '''
    Calculate model PMSE/RPMSE (root predictive mean squared error) in
    Linear Regression.
    
    PMSE or MSE is defined as the average squared distance the predictions
    are from the true values (mean((y_pred - y_true)^2)), and measure how far 
    off the predictions are on average (i.e. how variable the predictions are).
    The root MSE is convenient because it is measured in the same units as the
    response.

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted values.

    Returns
    -------
    bias : float
        Model bias.

    '''
    ydiff = y_pred - y_true
    if root:
        return np.sqrt(np.mean(ydiff**2))
    else:
        return np.mean(ydiff**2)


def r2(y_true, y_pred):
    '''
    R-squared for a linear regression model. The percent of the variance in 
    the response that can be explained by the predictors.
    
    SSE : Sum of Squares Errors (Residuals)
    SSR : Sum of Squares Regression
    SST : Sum of Squares Total
    
    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted values.

    Returns
    -------
    rsquared : float
        R-squared.

    '''
    y_bar = np.mean(y_true)
    SST = np.sum((y_true - y_bar)**2)
    SSE = np.sum((y_true - y_pred)**2)
    #SSR = np.sum((y_pred - y_bar)**2)
    rsquared = 1 - SSE/SST
    # Note: the following deifinitions are only equal to the above when the
    # model is linear, and will only be equivalent for the training data (i.e.
    # the test data is not guarenteed to have equivalent definitions of R2)
    # The equivalence of these definitions will hold for some non-linear
    # models, but not for others. Therefore,
    # Kvalseth TO. Cautionary Note About R2. Am. Statistic. 1985;39:279–285.
    # recommends using 1 - SSE/SSR for the general cases because by it's 
    # definition, it is the most robust to a general suite of problems
    
    # alternative R2: np.corrcoef((ypred, ytrue))**2
    # rsquared = SSR/SST
    return rsquared


def lr_se_fromModel(estimator, X, y):
    '''
    Linear Regression Model Standard deviation (error) estimate, or,
    the estimate of the true standard deviation of the underlying distribution
    for y_true. This is the maximum likelihood estimate for sigma, where
    y_true ~ N(y_pred, sigma^2).

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted values.
    num_params : int
        Number of Model Parameters

    Returns
    -------
    sigma : float
        Model error estimate.

    '''
    p = X.shape[1]
    y_pred = estimator.predict(X)
    sigma = np.sqrt(np.sum((y - y_pred)**2)/(y.shape[0] - p - 1))
    return sigma


def lr_se_fromPrediction(y_true, y_pred, num_params=2):
    '''
    Linear Regression Model Standard deviation (error) estimate, or,
    the estimate of the true standard deviation of the underlying distribution
    for y_true. This is the maximum likelihood estimate for sigma, where
    y_true ~ N(y_pred, sigma^2).

    Parameters
    ----------
    estimator : sklearn model, or similar
        The model. Needs to have a predict_proba method.
    X : numpy array or pandas dataframe
        The design or feature matrix.
    y : numpy array or pandas series
        The target or response variable.

    Returns
    -------
    sigma : float
        Model error estimate.

    '''
    sigma = np.sqrt(np.sum((y_true - y_pred)**2)/(y_true.shape[0] - num_params - 1))
    return sigma


def adj_r2(y_true, y_pred, X, rsquared=None):
    '''
    Adjusted R-squared for a linear regression model. The percent of the
    variance in the response that can be explained by the predictors. The 
    adjustment is for the ratio or predictors to observations.

    Parameters
    ----------
    y_true : numpy array
        True values (observed response).
    y_pred : numpy array
        Predicted values.
    X : numpy array
        The design or feature matrix.
    rsquared : float
        The Unadjusted R-squared for the model, if available. Otherwise,
        it's calculated from `r2`

    Returns
    -------
    rsquared : float
        R-squared.

    '''
    if rsquared is None:
        rsquared = r2(y_true, y_pred)
    adj_r2 = 1 - (1 - rsquared)*((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))
    return adj_r2
    

def aicCalc(loglike, num_model_params, sample_size, c=2, metric="aicc"):
    '''
    Generic Akaike information criterion (AIC) calculation for models.
    Includes metrics for AIC: `aic`, Corrected AIC: `aicc`, Bayesian
    Information Criterion (BIC): `bic`, or extended BIC: `ebic`.

    Parameters
    ----------
    loglike : float
        The loglikelihood of the model.
    num_model_params : int
        Number of parameters in the model.
    sample_size : int
        Number of observations.
    c : int or float, optional
        correction parameter for ebic. The default is 2.
    metric : str, optional
        The metric to evaluate (see main description). The default is "aicc".

    Raises
    ------
    ValueError
        Raised if no valid metric is selected.

    Returns
    -------
    ic : float
        The IC value.

    '''
    ic = None
    if metric == "aic" or metric == "aicc":
        ic = (2*num_model_params) - 2*(loglike)
        if metric == "aicc":
            ic = (ic + ((2*(num_model_params**2) +
                    2*num_model_params) /
                   (sample_size - num_model_params - 1))
                if (sample_size - num_model_params - 1) != 0 else ic)
    elif metric == "bic":
        ic = (np.log(sample_size)*num_model_params) - 2*loglike
    elif metric == "ebic":
        ic = (c*np.log(sample_size)*num_model_params) - 2*loglike
    if ic is not None:
        return ic
    else:
        raise ValueError("Not a valid method. No criteria was calculated")
        

def glm_regularized_AIC(X, Y, reg_mod, unreg_mod,
             tol=1e-6, method="kawano", family="binomial"):
    '''
    Calculate AIC for a Generalized Linear Model with regularization.
    
    See 'AIC for the Lasso in GLMs', Y. Ninomiya and S. Kawano (2016)

    Parameters
    ----------
    X : numpy array or pandas dataframe
        Feature or design matrix.
    Y : numpy array or pandas series
        Target or response variable.
    reg_mod : sklearn, or similar
        The regularized model.
    unreg_mod : sklearn, or similar
        The unregularized model.
    tol : float, optional
        Tolerance cutoff for counting non-zero coefficients. The default is
        1e-6.
    method : str, optional
        The method for calculating the AIC. Either `kawano` or `Hastie`.
        The default is "kawano".
    family : str, optional
        The type of generalised linear model. The default is "binomial".

    Raises
    ------
    ValueError
        Raised if an invalid family is picked.

    Returns
    -------
    aic : float
        The calculated AIC.

    '''
    # requires predict_proba method for logreg, and predict method for others, for poisson, predict output should be lambda, i.e. it should already be exponentiated
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame or pd.Series):
        Y = Y.values
    aic = None
    if family == "binomial":
        nllf = glm_likelihood_bernoulli
        unreg_prob = unreg_mod.predict_proba(X)[:, 1]
        reg_prob = reg_mod.predict_proba(X)[:, 1]
        y_mat_unreg = np.diag(unreg_prob * (1 - unreg_prob))
        y_mat_reg = np.diag(reg_prob * (1 - reg_prob))
    elif family == "poisson":
        nllf = glm_likelihood_poisson
        unreg_pred = unreg_mod.predict(X)
        reg_pred = reg_mod.predict(X)
        y_mat_unreg = np.diag(unreg_prob)
        y_mat_reg = np.diag(reg_prob)
    elif family == "gaussian":
        nllf = glm_likelihood_gaussian
        unreg_pred = unreg_mod.predict(X)
        reg_pred = reg_mod.predict(X)
        y_mat_unreg = np.diag(unreg_pred)
        y_mat_reg = np.diag(reg_pred)
    else:
        raise ValueError("Not a valid family")
    
    reg_mod_coef = np.concatenate((reg_mod.intercept_, np.squeeze(reg_mod.coef_)))
    nonzero_idx = np.where([True if np.abs(coef) > tol else False for coef in reg_mod_coef])[0]
    if method == "kawano":
        X_nz = add_constant(X[:, nonzero_idx], prepend=True)
        
        j22 = np.linalg.multi_dot([X_nz.T, y_mat_reg, X_nz])
        j22_2 = np.linalg.multi_dot([X_nz.T, y_mat_unreg, X_nz])
        
        negloglike = nllf(reg_mod_coef, X, Y, lamb=0, l_norm=0)
        aic = 2*negloglike + np.sum(np.diag(np.linalg.inv(j22).dot(j22_2)))
    else:
        # Tibshirani, Hastie, Zou 2007, On the degrees of freedom on the lasso
        negloglike = nllf(reg_mod_coef, X, Y, lamb=0, l_norm=0)
        # Not 100% sure on this calculation. should it be 2*len(nonzero_idx),
        #the count of nonzero columns, or 2*mean(non_zero_idx)?
        aic = 2*negloglike + 2*len(nonzero_idx)
    return aic


def js_div(px, py):
    '''
    Jensen-Shannon Divergence, which is a smoothed version of KL divergence.
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    '''
    midpoint = (px + py)*0.5
    js = rel_entr(px, midpoint)*0.5 + rel_entr(py, midpoint)*0.5
    return np.sum(js)


def kl_div(px, py):
    '''
    KL divergence.
    
    Note: scipy has a KL divergence function of the same name, but it adds
    extra terms. 
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    '''
    kl = np.sum(px*np.log(px/py))
    return kl


def HellingerDistanceMVN(mu1, mu2, cov1, cov2, squared=False):
    '''
    Quantifies the similarity between two multivariate normal distributions.

    Parameters
    ----------
    mu1 : numpy array
        Mean of the first distribution.
    mu2 : numpy array
        Mean of the second distribution.
    cov1 : numpy array
        Covariance of the first distribution.
    cov2 : numpy array
        Covariance of the second distribution.
    squared : bool, optional
        Return the squared distance. The default is False.

    Returns
    -------
    float
        Hellinger Distance.

    '''
    mu_diff = mu1 - mu2
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    cov_sum = (cov1 + cov2)/2
    det_cov_sum = np.linalg.det(cov_sum)
    inv_cov_sum = np.linalg.inv(cov_sum)
    
    dets = (((det_cov1**0.25)*(det_cov2**0.25))/det_cov_sum**0.5)
    expon = -0.125*mu_diff.dot(inv_cov_sum).dot(mu_diff)

    H2 = 1 - dets*np.exp(expon)
    if squared:
        return H2
    else:
        return np.sqrt(H2)

def HellingerDistanceUN(mu1, mu2, sd1, sd2, squared=False):
    '''
    Quantifies the similarity between two Univariate normal distributions.

    Parameters
    ----------
    mu1 : float
        Mean of the first distribution.
    mu2 : float
        Mean of the second distribution.
    sd1 : float
        Standard Deviation of the first distribution.
    sd2 : float
        Standard Deviation of the second distribution.
    squared : bool, optional
        Return the squared distance. The default is False.

    Returns
    -------
    float
        Hellinger Distance.

    '''
    mu_diff2 = (mu1 - mu2)**2
    sd_sum = (sd1**2 + sd2**2)
    H2 = 1 - np.sqrt((2*sd1*sd2)/sd_sum)*np.exp(-0.25*(mu_diff2/sd_sum))
    
    if squared:
        return H2
    else:
        return np.sqrt(H2)
    

def HotellingsTwoSampleMVTtest(mu1, mu2, cov1, cov2, n1, n2, pval=True):
    '''
    Used for multivariate hypothesis testing. Can test the similarity of two
    Multivariate samples assumed to follow a normal distribution.

    Parameters
    ----------
    mu1 : numpy array
        Mean of the first distribution.
    mu2 : numpy array
        Mean of the second distribution.
    cov1 : numpy array
        Covariance of the first distribution.
    cov2 : numpy array
        Covariance of the second distribution.
    n1 : int
        Number of observations in the first sample.
    n2 : int
        Number of observations in the first sample.
    pval : bool, optional
        Calculate the p-value of the statistic. The default is True.

    Returns
    -------
    float
        Either p-value or t-statistic.

    '''
    mu_diff = mu1 - mu2
    pooled_cov = (((n1 - 1)*cov1) + ((n2 - 1)*cov2))/(n1 + n2 - 2)
    inv_pooled_cov = np.linalg.inv(pooled_cov)
    
    t2 = ((n1*n2)/(n1 + n2))*mu_diff.dot(inv_pooled_cov).dot(mu_diff)
    if pval:
        # the T2 dist is proportional to an F-dist. For simplicity...
        p = cov1.shape[0]
        m = (n1 + n2 - p - 1)
        
        f = (m/((n1 + n2 - 2)*p))*t2
        pv = f.sf(f, dfn=p, dfd=m) 
        return pv
    else:
        return t2
    

def accuracy(y_true, y_pred):
    if y_true.shape[1] != 1:
        return np.mean(one_hot_decode(y_true) == one_hot_decode(y_pred))
    else:
        return np.mean(y_true == decision_boundary_1D(y_pred))
