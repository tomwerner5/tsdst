import numpy as np
import pandas as pd

from scipy.special import rel_entr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from statsmodels.tools import add_constant


def top_20p(y_true, y_score):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    cutoff = int(0.2*y_true.shape[0])
    prob_sort = np.argsort(y_score)[::-1]
    top_20 = y_true[prob_sort][:cutoff].sum()/y_true.sum()
    return top_20


# For use with custom scorers in sklearn, otherwise, use top_20p
def top_20p_score(estimator, X, y, yprob=None):
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
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_true = y.values
    else:
        y_true = y
    y_score = estimator.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true, y_score, average='weighted')
    return auc


def cox_snell_r2(ll_est, ll_null, n_obs):
    # Cox Snell R2
    ratio = 2/n_obs
    cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    return cs_r2


def nagelkerke_r2(ll_null, n_obs, cs_r2=None, ll_est=None):
    ratio = 2/n_obs
    r2_max = 1 - np.exp(ratio*ll_null)
    if cs_r2 is None:
        cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
    # Correction to CS R2 to bound between 0, 1
    n_r2 = cs_r2/r2_max
    return n_r2


def tjur_r2(y_true, y_score):
    #Tjur R2
    y_mu1 = y_score[y_true == 1].mean()
    y_mu0 = y_score[y_true == 0].mean()
    t_r2 = np.abs(y_mu1 - y_mu0)
    return t_r2


def mcfadden_r2(ll_est, ll_null):
    # When the saturated model is not available (common), then
    # Mcfaddens R2 is equivalent to the likelihood ratio r2
    m_r2 = 1 - (ll_est/ll_null)
    return m_r2


def number_of_nonzero_coef(X, model):
    num_coef = None
    try:
        num_coef = sum((True if c != 0 else False for c in model.coef_.reshape(-1,)))
    except:
        num_coef = X.shape[1]
    return num_coef


def conf_mat_metrics(y_true, y_pred, conf_metric='all'):
    # Assumes 1 as the positive class
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
    metrics['Conf_Mat'] = confMat
    
    if conf_metric == 'all':
        return metrics
    else:
        return metrics[conf_metric]     

    
def bias(y_true, y_pred):
    ydiff = y_pred - y_true
    bias = np.mean(ydiff)
    return bias


def rpmse(y_true, y_pred):
    ydiff = y_pred - y_true
    rpmse = np.sqrt(np.mean(ydiff**2))
    return rpmse


def r2(y_true, y_pred):
    y_bar = np.mean(y_true)
    SST = np.sum((y_true - y_bar)**2)
    #SSE = np.sum((y_true - y_pred)**2)
    SSR = np.sum((y_pred - y_bar)**2)
    # also equal to 1 - SSE/SST
    # alternative R2: np.corrcoef((ypred, ytrue))**2
    r2 = SSR/SST
    return r2


def adj_r2(y_true, y_pred, X, r2=None):
    if r2 is None:
        y_bar = np.mean(y_true)
        SST = np.sum((y_true - y_bar)**2)
        #SSE = np.sum((y_true - y_pred)**2)
        SSR = np.sum((y_pred - y_bar)**2)
        # also equal to 1 - SSE/SST
        # alternative R2: np.corrcoef((ypred, ytrue))**2
        r2 = SSR/SST
    adj_r2 = 1 - (1 - r2)*((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))
    return adj_r2
    

def aicCalc(loglike, num_model_params, sample_size, c=2, metric="aicc"):
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
        

# TODO: Check the kawano method for accuracy... something doesn't look right, particularily with the nonzero_idx
def lassoAIC(Xtrain, Ytrain, reg_mod, unreg_mod, tol=1e-8, method="kawano", pos_class=1):
    aic = None
    if method == "kawano":
        """See 'AIC for the Lasso in GLMs', Y. Ninomiya and S. Kawano (2016)"""
        pred_prob_unreg = unreg_mod.predict_proba(Xtrain)[:, pos_class]
        pred_prob_reg = reg_mod.predict_proba(Xtrain)[:, pos_class]

        reg_mod_coef = np.concatenate((reg_mod.intercept_, np.squeeze(reg_mod.coef_)))

        #nonzero_idx = np.where([True if np.abs(coef) > tol else False for coef in reg_mod_coef])
        Xtrain_const = add_constant(Xtrain, prepend=True)
        Xtrain_nonzero = add_constant(Xtrain_const)
        unreg_prob_cov = np.diag(pred_prob_unreg*(1 - pred_prob_unreg))
        reg_prob_cov = np.diag(pred_prob_reg*(1 - pred_prob_reg))
        
        j22 = np.linalg.multi_dot([Xtrain_nonzero.T, reg_prob_cov, Xtrain_nonzero])
        j22_2 = np.linalg.multi_dot([Xtrain_nonzero.T, unreg_prob_cov, Xtrain_nonzero])
        aic = (-2*np.sum((Ytrain*np.log(pred_prob_reg)) +
                         ((1 - Ytrain)*np.log(1 - pred_prob_reg))) +
               2*np.sum(np.diag(np.linalg.pinv(j22).dot(j22_2))))
    else:
        pred_prob_reg = unreg_mod.predict_proba(Xtrain)[:, pos_class]
        
        reg_mod_coef = np.concatenate((reg_mod.intercept_, np.squeeze(reg_mod.coef_)))

        nonzero_idx_sum = sum((1 if np.abs(coef) > tol else 0 for coef in reg_mod_coef))
        aic = (-2*np.sum((Ytrain*np.log(pred_prob_reg)) +
                         ((1 - Ytrain)*np.log(1 - pred_prob_reg))) +
               2*nonzero_idx_sum)
    return aic
    

def calcAICsLasso(penalty, Xtrain, Ytrain, sample_size, unreg_full_mod=None,
                  rs=123):
    cur_mod_reg = LogisticRegression(C=penalty, max_iter=10000, penalty="l1",
                                     solver='liblinear', random_state=rs,
                                     n_jobs=1).fit(Xtrain, Ytrain)
    Yprob = cur_mod_reg.predict_proba(Xtrain)[:, 1]
    Ypred = cur_mod_reg.predict(Xtrain)
    mod_coef = np.concatenate((cur_mod_reg.intercept_, np.squeeze(cur_mod_reg.coef_)))
    nzc = sum((1 if coef != 0 else 0 for coef in mod_coef))
    loglike = np.sum(Ytrain*np.log(Yprob) + (1 - Ytrain)*np.log(1 - Yprob))
    
    aics = []
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="aic"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="aicc"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="bic"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="ebic"))
    aics.append(np.mean(Ypred != Ytrain))
    aics.append(f1_score(Ytrain, Ypred))
    if unreg_full_mod is not None:
        aics.append(lassoAIC(Xtrain, Ytrain, cur_mod_reg, unreg_full_mod,
                             tol=1e-8, method="kawano"))
        aics.append(lassoAIC(Xtrain, Ytrain, cur_mod_reg, unreg_full_mod,
                             tol=1e-8, method="hastie"))
    
    return nzc, aics


def js_div(px, py):
    '''
    Jensen-Shannon Divergence
    
    px: Probability of x (float or array of floats)
    py: Probability of y (float or array of floats)
    '''
    midpoint = (px + py)*0.5
    js = rel_entr(px, midpoint)*0.5 + rel_entr(py, midpoint)*0.5
    return np.sum(js)


def HellingerDistanceMVN(mu1, mu2, cov1, cov2, squared=False):
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
    mu_diff2 = (mu1 - mu2)**2
    sd_sum = (sd1**2 + sd2**2)
    H2 = 1 - np.sqrt((2*sd1*sd2)/sd_sum)*np.exp(-0.25*(mu_diff2/sd_sum))
    
    if squared:
        return H2
    else:
        return np.sqrt(H2)
    

def HotellingsTwoSampleMVTtest(mu1, mu2, cov1, cov2, n1, n2, pval=True):
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