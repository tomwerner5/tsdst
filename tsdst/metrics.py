import numpy as np
import pandas as pd

from scipy.special import rel_entr
from scipy.stats import f as f_distribution
from sklearn.metrics import confusion_matrix, roc_auc_score

from .distributions import (glm_likelihood_bernoulli, glm_likelihood_poisson,
                            glm_likelihood_gaussian)
from .utils import one_hot_decode


def top_20p(y_true, y_score):
    """
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

    """
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    cutoff = int(0.2*y_true.shape[0])
    prob_sort = np.argsort(y_score)[::-1]
    top_20 = y_true[prob_sort][:cutoff].sum()/y_true.sum()
    return top_20


# For use with custom scorers in sklearn, otherwise, use top_20p
def top_20p_score(estimator, X, y, yprob=None):
    """
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

    """
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
    """
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

    """
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_true = y.values
    else:
        y_true = y
    y_score = estimator.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true, y_score, average='weighted')
    return auc


def cox_snell_r2(ll_est, ll_null, n_obs):
    """
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

    """
    ratio = 2/n_obs
    cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    return cs_r2


def nagelkerke_r2(ll_null, n_obs, cs_r2=None, ll_est=None):
    """
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

    """
    ratio = 2/n_obs
    r2_max = 1 - np.exp(ratio*ll_null)
    if cs_r2 is None:
        cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    # Correction to CS R2 to bound between 0, 1
    n_r2 = cs_r2/r2_max
    return n_r2


def tjur_r2(y_true, y_score):
    """
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

    """
    y_mu1 = y_score[y_true == 1].mean()
    y_mu0 = y_score[y_true == 0].mean()
    t_r2 = np.abs(y_mu1 - y_mu0)
    return t_r2


def mcfadden_r2(ll_est, ll_null):
    """
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

    """
    m_r2 = 1 - (ll_est/ll_null)
    return m_r2


def number_of_nonzero_coef(X, model):
    """
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

    """
    num_coef = None
    try:
        num_coef = sum((True if c != 0 else False for c in model.coef_.reshape(-1,)))
    except:
        num_coef = X.shape[1]
    return num_coef


def conf_mat_metrics(y_true, y_pred, conf_metric='all'):
    """
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

    """
    confMat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {'Sens/Recall': 0,
               'Specificity': 0,
               'ppv': 0,
               'npv': 0,
               'Conf_Mat': 0}
    tn = confMat[0, 0]
    tp = confMat[1, 1]
    fp = confMat[0, 1]
    fn = confMat[1, 0]
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
        # Note matthews correlation is actually equivalent to a pearson
        # correlation between y_true and y_pred
        metrics['matthews_coef'] = ((tp*tn) - (fp*fn))/np.sqrt(tpfp*tpfn*tnfp*tnfn)
    metrics['Conf_Mat'] = confMat
    
    if conf_metric == 'all':
        return metrics
    else:
        return metrics[{'PPV': 'ppv', 'NPV': 'npv', 'Confusion Matrix': 'Conf_Mat'}.get(conf_metric, conf_metric)]

    
def bias(y_true, y_pred):
    """
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

    """
    ydiff = y_pred - y_true
    bias = np.mean(ydiff)
    return bias


def rpmse(y_true, y_pred, root=True):
    """
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

    """
    ydiff = y_pred - y_true
    if root:
        return np.sqrt(np.mean(ydiff**2))
    else:
        return np.mean(ydiff**2)


def r2(y_true, y_pred):
    """
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

    """
    y_bar = np.mean(y_true)
    SST = np.sum((y_true - y_bar)**2) # sum of squares total
    SSE = np.sum((y_true - y_pred)**2) # sum of squares error
    #SSR = np.sum((y_pred - y_bar)**2) # sum of squares regression
    rsquared = 1 - SSE/SST
    # Note: the following deifinitions are only equal to the above when the
    # model is linear, and will only be equivalent for the training data (i.e.
    # the test data is not guarenteed to have equivalent definitions of R2)
    # The equivalence of these definitions will hold for some non-linear
    # models, but not for others. Therefore,
    # Kvalseth TO. Cautionary Note About R2. Am. Statistic. 1985;39:279–285.
    # recommends using 1 - SSE/SST for the general cases because by it's 
    # definition, it is the most robust to a general suite of problems
    
    # alternative R2: np.corrcoef((ypred, ytrue))**2
    # rsquared = SSR/SST
    return rsquared


def lr_sd_fromModel(estimator, X, y):
    """
    Linear Regression Model Standard deviation (error) estimate, or,
    the estimate of the true standard deviation of the underlying distribution
    for y_true. This is the maximum likelihood estimate for sigma, where
    y_true ~ N(y_pred, sigma^2), or equivalently, error ~ N(0, sigma^2).

    Parameters
    ----------
    estimator : sklearn model, or similar
        The model needs to have a predict method.
    X : numpy array or pandas dataframe
        The design or feature matrix.
    y : numpy array or pandas series
        The target or response variable.

    Returns
    -------
    sigma : float
        Model error estimate.

    """
    p = X.shape[1]
    y_pred = estimator.predict(X)
    sigma = np.sqrt(np.sum((y - y_pred)**2)/(y.shape[0] - p - 1))
    return sigma


def lr_sd_fromPrediction(y_true, y_pred, num_params=2):
    """
    Linear Regression Model Standard deviation (error) estimate, or,
    the estimate of the true standard deviation of the underlying distribution
    for y_true. This is the maximum likelihood estimate for sigma, where
    y_true ~ N(y_pred, sigma^2), or equivalently, error ~ N(0, sigma^2).

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

    """
    sigma = np.sqrt(np.sum((y_true - y_pred)**2)/(y_true.shape[0] - num_params - 1))
    return sigma


def adj_r2(y_true, y_pred, X, rsquared=None):
    """
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

    """
    if rsquared is None:
        rsquared = r2(y_true, y_pred)
    adj_r2 = 1 - (1 - rsquared)*((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))
    return adj_r2
    

def aicCalc(loglike, num_model_params, sample_size, c=2, metric="aicc"):
    """
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

    """
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
        

def glm_regularized_AIC(X, Y, reg_mod, unreg_mod, tol=1e-6,
                        method='kawano', family='binomial'):
    """Calculate AIC with active coefficients or a Fisher information correction.

    The intercept participates in the likelihood and the parameter count.
    The Kawano correction compares the two models on the active columns.
    """
    original_X = X
    X, Y = np.asarray(X, dtype=float), np.asarray(Y).ravel()
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.r_[np.asarray(reg_mod.intercept_).ravel(), np.asarray(reg_mod.coef_).ravel()]
    active = np.abs(beta) > tol
    active[0] = True
    if family == 'binomial':
        nll = glm_likelihood_bernoulli(beta, design, Y, lamb=0)
        p, p0 = reg_mod.predict_proba(original_X)[:, 1], unreg_mod.predict_proba(original_X)[:, 1]
        weights, weights0 = p * (1 - p), p0 * (1 - p0)
    elif family == 'poisson':
        nll = glm_likelihood_poisson(beta, design, Y, lamb=0)
        weights, weights0 = reg_mod.predict(original_X), unreg_mod.predict(original_X)
    elif family == 'gaussian':
        nll = glm_likelihood_gaussian(beta, design, Y, lamb=0)
        weights = weights0 = np.ones(len(X))
    else:
        raise ValueError('Family must be binomial, poisson, or gaussian')
    if method == 'kawano':
        selected = design[:, active]
        fisher = selected.T @ (weights[:, None] * selected)
        fisher0 = selected.T @ (weights0[:, None] * selected)
        degrees = np.trace(np.linalg.pinv(fisher) @ fisher0)
    elif method == 'hastie':
        degrees = active.sum()
    else:
        raise ValueError('Method must be kawano or hastie')
    return 2 * (nll + degrees)


def js_div(px, py):
    """
    Jensen-Shannon Divergence, which is a smoothed version of KL divergence.
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    """
    midpoint = (px + py)*0.5
    js = rel_entr(px, midpoint)*0.5 + rel_entr(py, midpoint)*0.5
    return np.sum(js)


def kl_div(px, py):
    """
    KL divergence.
    
    Note: scipy has a KL divergence function of the same name, but it adds
    extra terms. 
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    """
    kl = np.sum(px*np.log(px/py))
    return kl


def HellingerDistanceMVN(mu1, mu2, cov1, cov2, squared=False):
    """
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

    """
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
    """
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

    """
    mu_diff2 = (mu1 - mu2)**2
    sd_sum = (sd1**2 + sd2**2)
    H2 = 1 - np.sqrt((2*sd1*sd2)/sd_sum)*np.exp(-0.25*(mu_diff2/sd_sum))
    
    if squared:
        return H2
    else:
        return np.sqrt(H2)
    

def HotellingsTwoSampleMVTtest(mu1, mu2, cov1, cov2, n1, n2, pval=True):
    """
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

    """
    mu_diff = mu1 - mu2
    pooled_cov = (((n1 - 1)*cov1) + ((n2 - 1)*cov2))/(n1 + n2 - 2)
    inv_pooled_cov = np.linalg.inv(pooled_cov)
    
    t2 = ((n1*n2)/(n1 + n2))*mu_diff.dot(inv_pooled_cov).dot(mu_diff)
    if pval:
        # the T2 dist is proportional to an F-dist. For simplicity...
        p = cov1.shape[0]
        m = (n1 + n2 - p - 1)
        
        f = (m/((n1 + n2 - 2)*p))*t2
        pv = f_distribution.sf(f, dfn=p, dfd=m)
        return pv
    else:
        return t2
    

def accuracy(y_true, y_pred):
    """Return accuracy for labels, binary scores, or one-hot arrays."""
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim == 2 and truth.shape[1] > 1:
        return np.mean(one_hot_decode(truth) == one_hot_decode(prediction))
    return np.mean(truth.ravel() == (prediction.ravel() >= 0.5))
