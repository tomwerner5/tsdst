import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as dt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('../')
from tsdst.metrics import accuracy
from tsdst.utils import one_hot_encode
from tsdst.nn.model import NeuralNetwork

import warnings
warnings.filterwarnings('ignore')

nf = 8
nc = 2
nobs = 1000
X_og, Y_og = make_classification(n_samples=nobs, n_features=nf,
                                 n_informative=2, n_classes=nc,
                                 flip_y=0.1, class_sep=1,
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
                    'activation_args': {},
                    'cost': 'cross_entropy',
                    'cost_args': {},
                    'derivative': 'softmax_cross_entropy_der',
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
         
n_iters = 5000
batch_size = X_train.shape[0]
eval_size = 1000
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

t0 = dt()                
nn = nn.fit(X_train, Y_train_oh)
t1 = dt()

print('tsdst Runtime (s):', t1 - t0)

#################################################################
################### Comparison With Keras #######################
#################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.keras.optimizers import Adam


def baseline_model():
	  # create model
    model = Sequential()
    model.add(Dense(10, input_dim=nf,
                    #kernel_regularizer=l2(0.01),
                    kernel_initializer=RandomUniform(minval=-np.sqrt(6/nf),
                                                     maxval=np.sqrt(6/nf),
                                                     seed=42)))
    model.add(Activation('relu'))
    model.add(Dense(7,
                    #kernel_regularizer=l2(0.01),
                    kernel_initializer=RandomUniform(minval=-np.sqrt(6/10),
                                                     maxval=np.sqrt(6/10),
                                                     seed=42)))
    model.add(Activation('relu'))
    model.add(Dense(nc,
                    #kernel_regularizer=l2(0.01),
                    kernel_initializer=TruncatedNormal(mean=0.0,
                                                       stddev=np.sqrt(2/(nc + 7)),
                                                       seed=42)))
    model.add(Activation('softmax'))
    
	  # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(epsilon=1e-8),
                  metrics=['accuracy']
                 )
    return model


k_model = baseline_model()
t2 = dt()
history = k_model.fit(X_train, Y_train_oh, epochs=5000, batch_size=X_train.shape[0],
                    verbose=0, shuffle=False)
t3 = dt()
print('Keras Runtime (s):', t3 - t2)

h_test = nn.predict(X_test)
print("NN Test Accuracy: ", accuracy(Y_test_oh, h_test))
print("Keras Test Accuracy: ", k_model.evaluate(X_test, Y_test_oh,
                                                verbose=0)[1])

glm = LogisticRegression(C=1, max_iter=100000, multi_class='auto', tol=1e-8).fit(X_train, Y_train.reshape(-1, ))
print("GLM Test Accuracy: ", glm.score(X_test, Y_test.reshape(-1, 1)))

plt.figure()
plt.title("Training Cost per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.plot(nn.costs)
plt.plot(history.history['loss'])
plt.legend(["tsdst", "Keras"])
plt.show()

plt.figure()
plt.title("Training Accuracy per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.plot(nn.metric)
plt.plot(history.history['accuracy'])
plt.legend(["tsdst", "Keras"])
plt.show()