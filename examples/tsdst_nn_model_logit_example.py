import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as dt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from tsdst.metrics import accuracy
from tsdst.utils import (one_hot_encode, one_hot_decode)
from tsdst.nn.model import NeuralNetwork

t0 = dt()

X_og, Y_og = make_classification(n_samples=1000, n_features=8, n_informative=2,
                            n_classes=2, flip_y=0.1, class_sep=1,
                            random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X_og, Y_og, test_size=0.3,
                                                    random_state=42)

Y_train_oh = one_hot_encode(Y_train)
Y_test_oh = one_hot_encode(Y_test)

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

model = {
         'hidden0': {'depth': 10,
                     'activation': 'relu',
                     'derivative': 'relu_der',
                     'activation_args': {},
                     'initializer': 'he_uniform',
                     'dropout_keep_prob': 1,
                     'lambda': {'Weight': 0,
                               'activity': 0,
                               'bias': 0
                              },
                    'lp_norm': {'Weight': 2,
                               'activity': 2,
                               'bias': 2
                              },
                    'use_batch_norm': False
                     },
         'hidden1': {'depth': 7,
                     'activation': 'relu',
                     'derivative': 'relu_der',
                     'activation_args': {},
                     'initializer': 'he_uniform',
                     'dropout_keep_prob': 1,
                     'lambda': {'Weight': 0,
                               'activity': 0,
                               'bias': 0
                              },
                    'lp_norm': {'Weight': 2,
                               'activity': 2,
                               'bias': 2
                              },
                    'use_batch_norm': False
                     },
         'output': {'activation': 'softmax',
                    #'activation': 'sigmoid',
                    'activation_args': {},
                    'cost': 'cross_entropy',
                    #'cost': 'cross_entropy_binary',
                    'cost_args': {},
                    'derivative': 'softmax_cross_entropy_der',
                    #'derivative': 'sigmoid_cross_entropy_binary_der',
                    'derivative_args': {},
                    'initializer': 'xavier_normal',
                    'evaluation_metric': 'accuracy',
                    'lambda': {'Weight': 0,
                               'activity': 0,
                               'bias': 0
                              },
                    'lp_norm': {'Weight': 2,
                               'activity': 2,
                               'bias': 2
                              },
                    'use_batch_norm': False
                    }
         }
         
n_iters = 50
batch_size = 1
eval_size = int(X_train.shape[0]/batch_size) + sum([X_train.shape[0] % batch_size != 0])
nn = NeuralNetwork(model=model,
                   eval_size=eval_size,
                   batch_size=batch_size,
                   num_iterations=n_iters,
                   optimizer='adam',
                   optimizer_args={'learning_rate': 0.001,
                                   'beta1': 0.9,
                                   'beta2': 0.999,
                                   'eps': 1e-8},
                   m_scale=1,
                   bn_tol=1e-6,
                   bn_momentum=0,
                   shuffle=False,
                   print_cost=True,
                   random_state=42)
nn = nn.fit(X_train, Y_train_oh)
t1 = dt()
print('\nRuntime (s):', t1 - t0, '\n') 
long_seq = np.arange(0, nn.costs.shape[0], 1)
short_seq = np.arange(0, nn.costs.shape[0], eval_size)
costs_mean = [nn.costs[short_seq[i]:short_seq[i+1]].mean() for i in range(short_seq.shape[0]-1)]
metric_mean = [nn.metric[short_seq[i]:short_seq[i+1]].mean() for i in range(short_seq.shape[0]-1)]

plt.figure()
#plt.plot(long_seq, nn.costs)
plt.plot(np.array(costs_mean))
plt.figure()
#plt.plot(long_seq, nn.metric)
plt.plot(metric_mean)

h_test = nn.predict(X_test)
print("NN Test Accuracy: ", accuracy(Y_test_oh, h_test))

glm = LogisticRegression(C=1, max_iter=100000, multi_class='auto', tol=1e-8).fit(X_train, Y_train.reshape(-1, ))
#print(wb['bias0'], wb['Weight0'].reshape(-1, ))    
#print(glm.intercept_, glm.coef_.reshape(-1, ))
print("GLM Test Accuracy: ", glm.score(X_test, Y_test.reshape(-1, 1)))

cm = confusion_matrix(one_hot_decode(Y_test_oh), one_hot_decode(h_test))