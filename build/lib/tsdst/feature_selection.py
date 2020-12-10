import numpy as np
import pandas as pd

from .utils import updateProgBar


def naiveVarDrop(X, searchCols, tol=0.0001, standardize=False, asList=False, print_=False):
    cols_to_drop = []
    for i in searchCols:
        var = 1
        if standardize:
            std_data = (X.loc[:, i] - X.loc[:, i].mean())/X.loc[:, i].std(ddof=1)
            var = np.var(std_data, ddof=1)
        else:
            var = X.loc[:, i].var(ddof=1)
        if var < tol:
            cols_to_drop.append(i)
    if print_:
        print("Dropped " + str(len(cols_to_drop)) + " low-var Columns")
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)


def naiveScoreDrop(X, scores, tol=0.001, asList=False):
    cols_to_drop = []
    for i in scores.index:
        score = scores.loc[i].values
        if score < tol:
            cols_to_drop.append(i)
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)
    
    
def getHighCorrs(corr_mat, corr_thres, split_key="..&.."):
    top_corr = {}
    columns = corr_mat.columns
    for i in corr_mat.columns:
        for j in columns:
            if i == j:
                continue
            if np.isnan(corr_mat.loc[i, j]):
                continue
            cur_corr = corr_mat.loc[i, j]
            if abs(cur_corr) >= corr_thres:
                top_corr[split_key.join([i, j])] = cur_corr
        columns = columns.drop(i)
    return top_corr


# Note: in it's current form, this should only be used on pre-scaled data
def dropHighCorrs(X, top_corr, split_key="..&..", asList=False, print_=False):
    cols_to_drop = []
    for i in top_corr.keys():
        cols = i.split(split_key)
        var1 = np.nanvar(X.loc[:, cols[0]])
        var2 = np.nanvar(X.loc[:, cols[1]])
        if var1 < var2:
            cols_to_drop.append(cols[0])
        else:
            cols_to_drop.append(cols[1])
    if print_:
        print("Dropped " + str(len(cols_to_drop)) + " high-corr Columns")
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)


# expects a dataframe or ndarray where columns are attributes being correlated
def dropCorrProcedure(X, corr_thres, split_key="..&..", lowVarDrop=True, asList=False, print_=False):
    if lowVarDrop:
        X = naiveVarDrop(X, X.columns, tol=0.0001, standardize=False, asList=False, print_=print_)
    corr_mat = corr(X, rowvar=False)
    top_corr = getHighCorrs(corr_mat, corr_thres, split_key=split_key)
    dropped_cor = dropHighCorrs(X, top_corr, split_key=split_key, asList=asList, print_=print_)
    return dropped_cor

# assumes that your metric function takes inputs in this order: ytrue, yperd.
# If it doesn't, write a simple wrapper that will. Also assumes a pretrained model
# with a predict method
def permutation_importance(fit_model, Xtest, Ytest,
                           metric_func, seed=None, sort="dsc"):
    Ypred = fit_model.predict(Xtest)
    baseline = metric_func(Ytest, Ypred)
    imps = []
    if seed is not None:
        np.random.seed(seed)
    for col in Xtest.columns:
        col_copy = Xtest.loc[:, col].copy()
        Xtest.loc[:, col] = np.random.permutation(Xtest.loc[:, col])
        Ypred = fit_model.predict(Xtest)
        score = metric_func(Ytest, Ypred)
        Xtest.loc[:, col] = col_copy
        imps.append(baseline - score)
    res = pd.DataFrame(imps, index=Xtest.columns, columns=["Importance"])
    if sort == "asc":
        res = res.sort_values('Importance', ascending=True)
    elif sort == "dsc":
        res = res.sort_values('Importance', ascending=False)
    return res


def crossval_perm_imp(model, X, Y, metric_func, num_splits=8, seed=None, sort="dsc"):
    splits = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    pi_total = np.zeros((X.shape[1], 1))
    i = 0
    for train_idx, test_idx in splits.split(X, Y):
        Xtrain, Xtest = X.iloc[train_idx], X.iloc[test_idx]
        Ytrain, Ytest = Y.iloc[train_idx], Y.iloc[test_idx]
        fi_model = model.fit(Xtrain, Ytrain)
        pi = permutation_importance(fi_model, Xtest, Ytest,
                                metric_func, seed=seed, sort="None")
        pi_total += pi.values
        i += 1
        print("Completed " + str(i) + " of " + str(num_splits))
    pi_total /= num_splits
    pi_tot_df = pd.DataFrame(pi_total, index=X.columns, columns=["Importance"])
    if sort == "asc":
        pi_tot_df = pi_tot_df.sort_values('Importance', ascending=True)
    elif sort == "dsc":
        pi_tot_df = pi_tot_df.sort_values('Importance', ascending=False)
    return pi_tot_df


def calcForwardAics(X, Y, model, metric):

    mod = model.fit(X, Y)

    ss = X.shape[0]
    ncoefs = X.shape[1]

    Yprob = mod.predict_proba(X)[:, 1]
    #mod_coef = np.concatenate((model.intercept_, np.squeeze(model.coef_)))
    #ncoefs = sum([True if coef != 0 else False for coef in mod_coef])

    loglike = np.sum(Y*np.log(Yprob) + (1 - Y)*np.log(1 - Yprob))
    score = aicCalc(loglike, ncoefs, sample_size=ss, c=2, metric=metric)
    return score


def doParallelForward(save_path, save_name, metric, model,
                      target_var, new_predictor, use_probabilites,
                      mod_cols, Yprob=None):
    # Note: Should probably update this to feather (as of 2019)
    # since it is faster and better on memory in general.
    # Since this is not long term storage, feather would be a 
    # good choice. However, for long-term storage, parquet is 
    # probably the best option

    data = pd.read_msgpack(save_path + save_name + '.msg')
    chunks = [int(x[6:]) for x in data.keys()]
    chunks = sorted(chunks)
    mergedDF = pd.DataFrame()
    for chunk in chunks:
        mergedDF = mergedDF.append(data['chunk_'+str(chunk)])

    Y = mergedDF.loc[:, target_var]
    
    if use_probabilities:
        X = pd.DataFrame({'Yprob': Yprob, new_predictor: mergedDF.loc[:, new_predictor].values},
                         index=mergedDF.index)
    else:
        X = mergedDF.loc[:, mod_cols + new_predictor]
    
    res = calcForwardAics(X, Y, model, metric)

        # For debugging purposes, to attempt to force garbage collection in parallel
    data = []
    chunks = []
    mergedFD = []
    X = []
    Y = []
        
    return res


def forwardSelection(XY, target_var, model, metric='bic', verbose=True,
                     parallel=False, n_jobs=5, early_stop=False, perc_min=0.05,
                     stop_at_p=1000, stop_when=5, use_probabilites=False,
                     return_type='all'):
    
    def prepare_for_parallel(XY):
        num_chunks = int(((XY.memory_usage(index=True).sum()/(1024**3))/2) + 1)
            
        save_path = 'forward_tmp'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_name = '/forward_run_' + str(datetime.datetime.now())
        
        pd.to_msgpack(save_path + save_name + '.msg', {
            'chunk_{0}'.format(i):chunk for i, chunk in enumerate(np.array_split(XY, num_chunks))
            })
        
        return save_name, save_path, num_chunks
        
    
    t0 = dt()
    
    total_possible_complexity = int(((XY.shape[1]-1)**2) - ((XY.shape[1]-1)*((XY.shape[1]-1) - 1)/2))
    
    if early_stop:
        if stop_at_p >= (XY.shape[1] - 1):
            complexity = total_possible_complexity
            stop_at_p = XY.shape[1] - 1
        else:
            complexity = np.sum((XY.shape[1] - 1) - np.arange(0, stop_at_p))
    else:
        complexity = total_possible_complexity
    
    
    if verbose:
        print_time("Problem Complexity: " + str(complexity) + " Iterations Needed, " +
                   str(total_possible_complexity) + " Possible...",
                   t0, te=dt(), backsn=True)
    
    # Use this for an exhaustive search
    #all_combos = list(powerset(X.columns))
    final_mod_cols = []
    leftover_cols = [x for x in XY.columns if x != target_var]
    prev_minScore = np.inf
    early_stop_counter = 0
    loop_counter = 0
    
    if parallel:
        if verbose:
            print_time("\nPreparing Parallel Operation...", t0, te=dt())
            
        save_name, save_path, num_chunks = prepare_for_parallel(XY)
        arg = {'save_path': save_path,
               'save_name': save_name,
               'metric': metric,
               'model': model,
               'target_var': target_var,
               'use_probabilites': use_probabilites,
               'mod_cols': final_mod_cols,
               'Yprob': None
              }
    
    for i, step in enumerate(range(XY.shape[1] - 1)):
        scores = []
        if use_probabilites and i > 1:
            # create current model and probability array
            initial_fit = model.fit(XY.loc[:, final_mod_cols], XY.loc[:, target_var])
            Yprob = initial_fit.predict_proba(XY.loc[:, final_mod_cols])[:, 1]
            #print('\nprob gen step: ', Yprob.shape, "\n")
            if parallel:
                arg['Yprob'] = Yprob
        
        if parallel:
            if verbose:
                print_time("\nPerforming " + str(i + 1) + " of " + str(stop_at_p) + " Steps...",
                           t0, te=dt(), backsn=True)
            loop_arg = [{'new_predictor': [col]} for col in leftover_cols]
            scores = p_prog_simp(arg, loop_arg, doParallelForward, n_jobs)
            
        else:
            for k, col in enumerate(leftover_cols):
                if use_probabilites and i > 1:
                    #print('loop step: ', XY.loc[:, col].values.shape, "\n")
                    XY_prob = pd.DataFrame({'Yprob': Yprob, col: XY.loc[:, col].values}, index=XY.index)
                    score = calcForwardAics(XY_prob, XY.loc[:, target_var],
                                            model, metric)
                else:
                    score = calcForwardAics(XY.loc[:, final_mod_cols + [col]], XY.loc[:, target_var],
                                            model, metric)
                scores.append(score)
            
                if verbose:
                    loop_counter += 1
                    updateProgBar(loop_counter, complexity, t0)
        
        minScore = np.min(scores)
        if minScore < prev_minScore:
            prev_minScore = minScore
        
        minScoreLoc = np.where(minScore == np.array(scores))[0][0]
        final_mod_cols.append(leftover_cols[minScoreLoc])
        del leftover_cols[minScoreLoc]
        
        if early_stop:
            if np.abs(percentIncrease(minScore, prev_minScore)) >= perc_min:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            
            if early_stop_counter >= stop_when:
                print('Stopped Early')
                break
            if i >= stop_at_p:
                print('Stopped at ', str(i), 'Generations')
                break

    if parallel:
        file_ = save_path + save_name + '.msg'
        try:
            os.remove(file_)
        except FileNotFoundError:
            print('Could Not find file ', file_, 'Continuing...')
            
    if return_type == 'list':
        return final_mod_cols
    elif return_type == 'model':
        return model.fit(X.loc[:, final_mod_cols], Y)
    elif return_type == 'data':
        return X.loc[:, final_mod_cols]
    else:
        return final_mod_cols, model.fit(XY.loc[:, final_mod_cols],
                                         XY.loc[:, target_var]), XY.loc[:, final_mod_cols]

## TODO: import vif from modeling and incorporate a vif drop  
def vifDrop():
    return 0

