import itertools
import sys
import numpy as np
import pandas as pd
import patsy
import warnings

from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import _check_sample_weight
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(r'C:\Users\tomwe\PycharmProjects\tsdst')
from tsdst.modeling import crossVal, RegressionSE
from tsdst.metrics import r2, adj_r2
from tsdst.sampling import latinHypercube1D
from scipy.stats import t, f
#https://online.stat.psu.edu/stat503/lesson/11/11.2/11.2.1


class lm(LinearRegression):
    def fit(self, X, y, sample_weight=None, fit_data_labels=None):
        super(lm, self).fit(X, y, sample_weight=None)
        
        if fit_data_labels is None:
            if isinstance(X, pd.DataFrame):
                self.fit_data_labels = X.columns.astype(str)
            else:
                self.fit_data_labels = ['x'+str(i) for i in range(1, X.shape[1]+1)]
        else:
            self.fit_data_labels = fit_data_labels
        
        # sckit-learn preprocessing steps for LinearRegression.fit.
        # Borrowing this here to do validations on X and y
        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )
        # End of borrowed section
        
        y_pred = self.predict(X)
        n_obs = y.shape[0]
        n_coef = self.coef_.size + 1
    
        (self.se, self.cov, self.XtX_inv,
         self.sigma2, self.betas) = RegressionSE(X,
                                                y,
                                                self,
                                                logit=False,
                                                low_memory=False,
                                                lamda=0)
        
        na_coefs = np.isnan(self.se)
        self.total_df = n_obs - 1
        self.model_df = n_coef - 1 - na_coefs.sum()
        # could also be self.residual = n_obs - n_coef
        self.residual_df = self.total_df - self.model_df

        self.t_values = self.betas.reshape(-1, ) / self.se
        self.coef_p_values = 2 * (1 - t.cdf(np.abs(self.t_values),
                                                  self.residual_df))
        
        self.r2 = r2(y, y_pred)
        self.adj_r2 = adj_r2(y, y_pred, X[:, ~na_coefs[1:]], rsquared=self.r2)
        
        # global test if *any* outcomes associated with y
        df_percent_diff = (self.total_df
                           - self.model_df)/self.model_df
        self.f_stat = (self.r2/(1 - self.r2))*df_percent_diff
        self.global_p_value = (1 - f.cdf(np.abs(self.f_stat),
                                               self.model_df,
                                               self.residual_df))
        return self
    
    def _get_stars(self, p_value):
        if p_value >= 0.1:
            return r''
        elif p_value >= 0.05 and p_value < 0.1:
            return r'.'
        elif p_value >= 0.01 and p_value < 0.05:
            return r'*'
        elif p_value >= 0.001 and p_value < 0.01:
            return r'**'
        elif p_value >= 0 and p_value < 0.001:
            return r'***'
        else:
            warnings.warn('lm model did not return sensible p-values. '
                          'Check procedure and outputs for accuracy.')
            return 'Unknown'
    
    def _p_value_format(self, default_format, value,
                        sci_nota_thres=1e-5, min_value_thres=2.16e-16):
        if value >= sci_nota_thres or np.isnan(value):
            return default_format.format(value)
        elif value < sci_nota_thres and value >= min_value_thres:
            return '{:.4e} '.format(value)
        elif value < min_value_thres:
            return '{:<10s} '.format('< 2.16e-16')
        else:
            raise ValueError('p-value format could not be returned from '
                             'value ' + str(value))
    
    def summary(self, print_=True):
        
        summary = ''
        intercept_label = '(Intercept)'
        
        # Coefficient Table
        table_headers = [r'', r'Estimate',  r'Std. Error',  r't value', 
                         r'Pr(>|t|)', '']
        
        max_feature_label_length = max([len(label) for label in self.fit_data_labels])
        if max_feature_label_length < len(intercept_label):
            max_feature_label_length = len(intercept_label)
        
        table_formats = ['{:<' + str(max_feature_label_length) + '.' +
                         str(max_feature_label_length) + 's} ',
                         '{:12.8f} ', '{:12.8f} ', '{:12.3f} ', '{:0.8f} ',
                         '{:7.7s}']
        
        table_header_formats = ['{:<' + str(max_feature_label_length) + '.' +
                         str(max_feature_label_length) + 's} ',
                         '{:>12s} ', '{:>12s} ', '{:>12s} ', '{:>10s} ',
                         '{:>7s}']
        
        for i in range(len(table_headers)):
            summary += table_header_formats[i].format(table_headers[i])
        summary += '\n'
        
        num_coefs = len(self.betas)
        for i in range(num_coefs):
            if i == 0:
                summary += table_formats[0].format('(Intercept)')
            else:
                summary += table_formats[0].format(self.fit_data_labels[i-1])
            
            summary += table_formats[1].format(self.betas[i][0])
            summary += table_formats[2].format(self.se[i])
            summary += table_formats[3].format(self.t_values[i])
            summary += self._p_value_format(table_formats[4], self.coef_p_values[i])
            summary += table_formats[5].format(self._get_stars(self.coef_p_values[i]))
            summary += '\n'
            
            if i == (num_coefs-1):
                summary += '---\n'
                summary += r'Signif. codes:  0 "***" 0.001 "**" 0.01 "*" 0.05 "." 0.1 " " 1'
        
        summary += ('\n\nResidual standard error: '
                    '{:0.5f}'.format(np.sqrt(self.sigma2)) +
                    ' on ' + '{:.0f}'.format(self.residual_df) +
                    ' degrees of freedom'
                    '\nMultiple R-squared:  '
                    '{:0.5f}'.format(self.r2) +
                    ',    Adjusted R-squared:  '
                    '{:0.5f}'.format(self.adj_r2) + '\n'
                    'F-statistic: ' + '{:.3f}'.format(self.f_stat) +
                    ' on ' + '{:.0f}'.format(self.model_df) +
                    ' and ' + '{:.0f}'.format(self.residual_df) +
                    ' DF, p-value: ' + self._p_value_format('{:.10f}', self.global_p_value))
        
        if print_:
            print(summary)
            
        return summary
    


class RSMSearchCV(object):
    """Response Surface Methodolgy Hyperparameter tuning.

    This performs a grid search that searches through the hyperparameter
    space using a response surface methodology.

    """
    def __init__(self, model, params, cv_iterations, surface_metric,
                 calculate='Out of Sample', cv_args=None, max_iterations=1000,
                 interpolate_range=4,
                 evaluate_every=5, early_stop_tol=1e-8, eps=0.05,
                 eigen_thres=0.1, n_jobs=1):
        self.params = params
        self.num_hyperparams = len(params.keys())
        self.surface_metric = surface_metric
        self.calculate = calculate
        self.max_iterations = max_iterations
        self.interpolate_range = interpolate_range
        self.evaluate_every = evaluate_every
        self.early_stop_tol = early_stop_tol
        self.eps = eps
        self.n_jobs = n_jobs
        self.eigen_thres = eigen_thres
        
        keys_to_remove = ['model', 'cv_iterations', 'calculate']
        
        if cv_args is None:
            self.cv_args = {'metrics': [surface_metric]}
        else:
            for key in cv_args.keys():
                if key in keys_to_remove:
                    del cv_args[key]
                if key == 'metrics':
                    if isinstance(cv_args[key], str):
                        cv_args[key] = [cv_args[key]]
                    if surface_metric not in cv_args[key]:
                        cv_args[key].append(surface_metric)
            self.cv_args = cv_args
        
        self.model = model
        self.cv_iterations = cv_iterations
        
    def _unpack_params(self):
        starting_params = {}
        for param in self.params.keys():
            if self.params[param][0] == 'exact':
                starting_params[param] = self.params[param][2]
            else:
                minx = np.min(self.params[param][2])
                maxx = np.max(self.params[param][2])
                values = np.linspace(minx, maxx, self.interpolate_range)
                if self.params[param][1] == 'integer':
                    values = np.round(values).astype(int)
                    if np.unique(values).shape[0] != values.shape[0]:
                        raise ValueError('Design points are not unique.'
                                         ' Either decrease interpolate_range'
                                         ' or adjust the min/max range of '
                                         + str(param))
                    
                starting_params[param] = values
        self.starting_params = list(ParameterGrid(starting_params))
        
    def _fit_initial_hyperparameters(self, X, y):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(crossVal)(X=X, Y=y, cv_iterations=self.cv_iterations,
                              model=self.model.set_params(**params),
                              **self.cv_args)
            for params in tqdm(self.starting_params)
        )
        
        cv_rows_to_duplicate = len(results[0][self.calculate][self.surface_metric])
        self.surface_mean = []
        self.hyperparams = []
        for i, res in enumerate(results):
            self.hyperparams = self.hyperparams + [self.starting_params[i]]*cv_rows_to_duplicate 
            self.surface_mean = self.surface_mean + res[self.calculate][self.surface_metric]
            
        self.hyperparams = pd.DataFrame(self.hyperparams)
        self.surface_mean = np.array(self.surface_mean)
    
    def _transform_hyperparameters(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.hyperparams)
        self.transformed_hyperparams = self.scaler.transform(self.hyperparams)

    def _calculate_regression(self, eigen_thres=0.1):
        # eigenthres sets the threshold for detecting a stationary ridge.
        # set to 0 to ignore canonical analysis
        self.model = lm().fit(self.fit_data,
                              self.surface_mean,
                              fit_data_labels=self.fit_data_labels)
        self.B = np.zeros((self.num_hyperparams, self.num_hyperparams))
        
        #y_pred = self.model.predict(self.fit_data)
        #n_obs = self.surface_mean.shape[0]
        #n_coef = self.model.coef_.size + 1
        
        self.B[np.triu_indices(self.num_hyperparams)] = self.model.coef_[
                                                        self.num_hyperparams:]
        self.B += np.triu(self.B, 1).T
        I = np.eye(self.num_hyperparams)
        self.B *= ((1 - I) / 2 + I)
        self.Original_eigen_values_B, self.eigen_vectors_B = np.linalg.eig(self.B)
        max_eigen = np.max(np.abs(self.Original_eigen_values_B))
        active = max_eigen*eigen_thres <= np.abs(self.Original_eigen_values_B)
        self.eigen_values_B = self.Original_eigen_values_B
        
        if sum(active) == 0:
            warnings.warn('Max eigenvalue is less than the threshold. '
                          '(' + max_eigen*eigen_thres + '). Adjust threshold.')
            self.stationary_points_encoded = (-0.5) * np.linalg.inv(self.B).dot(
                self.model.coef_[:self.num_hyperparams])
        else:
            nzero = self.Original_eigen_values_B.shape[0] - sum(active)
            if nzero > 0:
                warnings.warn('Near-stationary-ridge situation detected -- '
                              'stationary point altered. Change eigen_thres ' 
                              'if this is not what you intend')
                U = self.eigen_vectors_B[:, active]
                laminv = 1.0/self.eigen_values_B[active]
                self.stationary_points_encoded = -0.5 * U.dot(np.diag(laminv)).dot(U.T).dot(self.model.coef_[:self.num_hyperparams])

                if sum(active) < U.shape[0]:
                    self.eigen_values_B[~active] = 0
            else:
                self.stationary_points_encoded = (-0.5) * np.linalg.inv(self.B).dot(
                    self.model.coef_[:self.num_hyperparams])
        
        self.stationary_points = self.scaler.inverse_transform(
            self.stationary_points_encoded.reshape(1, -1))
               
    def fit(self, X, y):
        self._unpack_params()
        self._fit_initial_hyperparameters(X, y)
        self._transform_hyperparameters()
        
        # I supect the sklearn version of this is faster, but in case I need it:
        #formula = ('surface_mean ~ (' + 
        #          '+'.join(self.transformed_hyperparams.columns.difference(['surface_mean'])) +
        #          ')**2 + ' +
        #          '+'.join(['I('+p+'**2)' for p in self.transformed_hyperparams.columns.difference(['surface_mean'])]))
        #
        #self.fit_data = patsy.dmatrices(formula, data=self.transformed_hyperparams,
        #                           return_type='dataframe')[1]
        
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.poly.fit(self.transformed_hyperparams)
        self.fit_data = self.poly.transform(self.transformed_hyperparams)
        self.fit_data_labels = self.poly.get_feature_names(self.hyperparams.columns)

        self._calculate_regression(self.eigen_thres)
        
        # should return self, otherwise for debuging
        return self.model
    
    def _get_stars(self, p_value):
        if p_value >= 0.1:
            return r''
        elif p_value >= 0.05 and p_value < 0.1:
            return r'.'
        elif p_value >= 0.01 and p_value < 0.05:
            return r'*'
        elif p_value >= 0.001 and p_value < 0.01:
            return r'**'
        elif p_value >= 0 and p_value < 0.001:
            return r'***'
        else:
            warnings.warn('lm model did not return sensible p-values. '
                          'Check procedure and outputs for accuracy.')
            return 'Unknown'
    
    def _p_value_format(self, default_format, value,
                        sci_nota_thres=1e-5, min_value_thres=2.16e-16):
        if value >= sci_nota_thres:
            return default_format.format(value)
        elif value < sci_nota_thres and value >= min_value_thres:
            return '{:.4e} '.format(value)
        elif value < min_value_thres:
            return '{:<10s} '.format('< 2.16e-16')
        else:
            raise ValueError('p-value format could not be returned from '
                             'value ' + str(value))
    
    def summary(self, print_=True):
        
        lm_summary = self.model.summary(print_=False)
        
        summary = (lm_summary +
                    '\n\nStationary point of response surface:\n')
        
        stat_point_df = pd.DataFrame(self.stationary_points.reshape(1, -1),
                                     columns=self.fit_data_labels[:self.num_hyperparams]).to_string(index=False)
        stat_point_df_encoded = pd.DataFrame(self.stationary_points_encoded.reshape(1, -1),
                                             columns=['x' + str(i) for i in range(self.num_hyperparams)]).to_string(index=False)
        eigen_vectors_df = pd.DataFrame(self.eigen_vectors_B,
                                        index=['x' + str(i) for i in range(self.num_hyperparams)],
                                        columns=['V' + str(i) for i in range(self.num_hyperparams)]).to_string()
        
        summary += (stat_point_df_encoded +
                    '\n\nStationary point in original units:\n' +
                    stat_point_df +
                    '\n\nEigen Analysis:\nnumpy.linalg.eig() decomposition\nvalues\n' +
                    str(self.eigen_values_B) +
                    '\n\nvectors\n' +
                    eigen_vectors_df)
        
        if print_:
            print(summary)
        return summary
        
np.random.seed(42)

for j in [7, 8]:
    for i in range(1, j+1):
        data = pd.DataFrame(np.round(np.random.normal(size=(j, 3))))
        samp = latinHypercube1D(data, i, random_state=42, shuffle_after=True,
                             sort_=True, sort_method="quicksort", sort_cols=None,
                             stratified=True, bin_placement="spaced", verbose=False)
        print(samp.shape[0] == i)

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier as gbm
from tsdst.nn.model import NeuralNetwork

X, y =  make_classification(1000, n_features=20, n_informative=12,
                            class_sep=0.75, flip_y=0.2, random_state=42)

# X, y = make_regression(n_samples=100, n_features=20, n_informative=12,
#                        n_targets=1, bias=3, effective_rank=3,
#                        tail_strength=0.5, noise=1, shuffle=True,
#                        coef=False, random_state=42)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X.to_csv('X.csv')
y.to_csv('y.csv')

params = {
        'max_depth': ('range', 'integer', [1, 100]),
        'subsample': ('range', 'float', [0.25, 1]),
        'min_samples_split': ('range', 'integer', [2, 100]),
        'min_samples_leaf': ('range', 'integer', [1, 100])
    }

model = gbm(n_estimators=100, random_state=42)
        

#mod = lm().fit(X, y)
#mod.summary()

cur_rsm = RSMSearchCV(model, params, cv_iterations=5, surface_metric='AUC',
                  cv_args={'print_': False, 'metrics':['AUC']}, max_iterations=1000,
                  interpolate_range=3, evaluate_every=5, early_stop_tol=1e-8,
                  eps=0.05, eigen_thres=0.01, n_jobs=16)
results = cur_rsm.fit(X, y)
cur_rsm.hyperparams['surface_mean'] = cur_rsm.surface_mean
cur_rsm.hyperparams.to_csv('hyperparams_test_exp.csv')

cur_rsm.summary()