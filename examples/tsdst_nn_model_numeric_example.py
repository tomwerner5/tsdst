from sklearn.datasets import make_regression

import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as dt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet
from sklearn.metrics import confusion_matrix

from tsdst.metrics import r2
from tsdst.nn.model import NeuralNetwork


def cust_r2(y_true, y_pred):
    return np.mean([r2(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])


X_og, Y_og = make_regression(n_samples=1000, n_features=50, 
                             n_informative=20, n_targets=1,
                             bias=0.0, effective_rank=None, tail_strength=0.5,
                             noise=0.0, shuffle=True, coef=False,
                             random_state=42)
 
t0 = dt()

X_train, X_test, Y_train, Y_test = train_test_split(X_og, Y_og, test_size=0.3,
                                                    random_state=42)

Y_train = Y_train.reshape(X_train.shape[0], -1)
Y_test = Y_test.reshape(X_test.shape[0], -1)

model = {
         'hidden0': {'depth': 10,
                     'activation': 'relu',
                     'derivative': 'relu_der',
                     'activation_args': {},
                     'initializer': 'he_uniform',
                     'dropout_keep_prob': 1,
                     'lambda': 0.01,
                     'lp_norm': 2
                     },
         'hidden1': {'depth': 7,
                     'activation': 'relu',
                     'derivative': 'relu_der',
                     'activation_args': {},
                     'initializer': 'he_uniform',
                     'dropout_keep_prob': 1,
                     'lambda': 0.01,
                     'lp_norm': 2
                     },
         'output': {'activation': 'linear',
                    'activation_args': {},
                    'cost': 'mse',
                    'cost_args': {},
                    'derivative': 'mse_linear_der',
                    'derivative_args': {},
                    'initializer': 'xavier_normal',
                    'evaluation_metric': cust_r2
                    }
         }
         
n_iters = 100000
nn = NeuralNetwork(model=model,
                   eval_size=1000,
                   num_iterations=n_iters,
                   optimizer='adam',
                   optimizer_args={'learning_rate': 0.001,
                                   'beta1': 0.9,
                                   'beta2': 0.999,
                                   'eps': 1e-8},
                   m=None,
                   m_scale=1,
                   print_cost=True,
                   random_state=42)
nn = nn.fit(X_train, Y_train)
t1 = dt()
print('\nRuntime (s):', t1 - t0, '\n') 
long_seq = np.arange(0, n_iters, 1)
short_seq = np.arange(0, n_iters, 1000)
plt.figure()
plt.plot(long_seq, nn.costs)
plt.plot(short_seq, nn.costs[::1000])
plt.figure()
plt.plot(long_seq, nn.metric)
plt.plot(short_seq, nn.metric[::1000])

h_test = nn.predict(X_test)
print("NN Test RMSE: ", np.sqrt(np.mean((h_test - Y_test)**2)))
print("NN Test my R2: ", cust_r2(Y_test, h_test))
#print("NN Test Accuracy: ", np.mean(one_hot_decode(h_test) == one_hot_decode(Y_test_oh)))

#mod = LinearRegression().fit(X_train, Y_train.reshape(-1, ))
mod = MultiTaskElasticNet(l1_ratio=0.00001).fit(X_train, Y_train)

#print(wb['bias0'], wb['Weight0'].reshape(-1, ))    
#print(glm.intercept_, glm.coef_.reshape(-1, ))
print("GLM Test my R2: ", cust_r2(Y_test, mod.predict(X_test)))
print("GLM Test RMSE: ", np.sqrt(np.mean((mod.predict(X_test) - Y_test)**2)))


samples = nn.draw_predictive_samples(X_test, n_samples=10000, n_outputs=1)
plt.figure()
plt.hist(samples[0, :], bins=30)