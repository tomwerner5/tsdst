import numpy as np
from scipy.special import xlogy


def relu(x, alpha=0):
    z = x.copy()
    # If alpha is between 0 and 1, the function performs leaky relu
    # alpha values are commonly between 0.1 and 0.3 for leaky relu
    z[z < 0] = z[z < 0]*alpha
    return z


def relu_der(x, alpha=0):
    # Note: relu derivative is technically undefined at x=0, but
    # tensorflow.nn.relu uses 0 for this special case, so that's what
    # I do here
    
    # If alpha != 0, then derivative is leaky relu
    dZ = x.copy()
    dZ[x <= 0] = alpha
    dZ[x > 0] = 1
    return dZ


def elu(x, alpha=0.1):
    '''
    Exponential Linear Unit.
    
    Main pro is it avoids dead relu. main con is computation time.
    
    Parameters
    ----------
    x : numpy array
        Values to be activated.
    alpha : float, optional
        The scale factor for the linear unit.
        Typical values are between 0.1 and 0.3.
        The default is 0.1.

    Returns
    -------
    z : numpy array
        The activated values.

    '''
    z = x.copy()
    
    z[z < 0] = alpha*(np.exp(z[z < 0]) - 1)
    return z


def elu_der(x, alpha=0.1):
    dZ = x.copy()
    dZ[x <= 0] = alpha*(np.exp(dZ[x <= 0]) - 1) + alpha
    dZ[x > 0] = 1
    return dZ


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    sigm = sigmoid(x)
    der = sigm * (1 - sigm)
    return der


def softmax(x, center=True):
    # for numerical stability only
    x_temp = x.copy()
    if center:
        # TODO: see which is faster, axis=1, or global max
        x_temp -= np.max(x, axis=1, keepdims=True)
        #x_temp = (x_temp - np.mean(x))/np.std(x_temp)
    expx = np.exp(x_temp)
    return expx/(np.sum(expx, axis=1, keepdims=True))


def softmax_oneout(x, center=True):
    # for numerical stability only
    if center:
        x_temp = x - np.max(x)
    else:
        x_temp = x.copy()
    x_temp = x.copy()
    x_temp[:, x_temp.shape[1]-1] = 0
    expx = np.exp(x_temp)
    return expx/(np.sum(expx, axis=1, keepdims=True))


def create_jac_mask(shape, n_classes):
    mask = np.zeros(shape=shape)
    
    for i in range(int(shape[0]/n_classes)):
        mask[(i*n_classes):((i+1)*n_classes),
             (i*n_classes):((i+1)*n_classes)] = 1
    return mask


def softmax_der(x, center=True):
    # full jacobian, but diagonal is the only needed result
    soft = softmax(x, center=True).reshape(-1, 1)
    kronecker = np.dot(soft, soft.T)
    jac_mask = create_jac_mask(shape=kronecker.shape, n_classes=x.shape[1])
    
    out = np.diagflat(soft) - kronecker*jac_mask
    return out


def selu(x, alpha=1.6732632423543772848170429916717,
         lamda=1.0507009873554804934193349852946):
    # scaled exponential relu
    # Note: If lambda=1, then selu reduces to elu
    # Note: vanishing/exploding gradients are impossible with selu
    dZ = x.copy()
    dZ[x > 0] = lamda
    dZ[x <= 0] = lamda*(alpha*np.exp(x[x <= 0]))
    return dZ


def selu_der(x, alpha=1.6732632423543772848170429916717,
             lamda=1.0507009873554804934193349852946):
    # scaled exponential relu
    # Note: If lambda=1, then selu reduces to elu
    z = x.copy()
    z[z > 0] = lamda*z[z > 0]
    z[z <= 0] = lamda*(alpha*(np.exp(z[z < 0]) - 1))
    return z


def cross_entropy_binary(y_true, y_pred, delta=1e-9):
    '''
    Binary Cross Entropy.
    
    While the definition varies a little bit across ML or information theory
    domains, Cross-entropy in general is a method of measuring the difference
    of information between two probability distributions (i.e. a way of
    measuring how similar they are). In our context, cross entropy is the
    difference between our observed values (which can be viewed as a
    probability distribution where every probability is either 0 or 1, since
    every value is known and thus has an extreme probability) and the
    predicted values (which are actual probabilities since the true values
    are unknown). Since we are interested in predicting values that follow a
    bernoulli distribution, the cross entropy takes the form of the negative
    log-likelihood of the bernoulli distribution.
    
    With this cost function defined, Neural Networks are just performing
    a more BA version of maximum likelihood estimation. A negative cost is
    defined because maximizing the likelihood is the same as minimizing the
    negative likelihood. 
    
    Parameters
    ----------
    y_true : numpy array
        True, observed values. The outcome of an event (where 1 == success).
    y_pred : numpy array
        The predicted success probabilities.
    delta : float, optional
        A small, positive constant to offset predicted probabilities that are
        zero, which avoids log(0). Is ignored if delta = 0.
        The default is 1e-9.

    Returns
    -------
    cost : float
        The binary cross-entropy.

    '''

    # Compute the cross-entropy cost
    # To avoid log(0) errors (not necessary in most cases)
    ypred = y_pred.copy()
    if delta != 0:
        ypred[ypred <= delta] = delta
        ypred[ypred >= 1-delta] = 1-delta
    
    # m is the number of observations, and m_scale is a scaling factor to make
    # the computation easier in case the gradients are really big. 
    cost = -np.sum(xlogy(y_true, ypred) + xlogy(1 - y_true, 1 - ypred))
    return cost


def cross_entropy(y_true, y_pred, delta=1e-9):
    '''
    Binary Cross Entropy.
    
    While the definition varies a little bit across ML or information theory
    domains, Cross-entropy in general is a method of measuring the difference
    of information between two probability distributions (i.e. a way of
    measuring how similar they are). In our context, cross entropy is the
    difference between our observed values (which can be viewed as a
    probability distribution where every probability is either 0 or 1, since
    every value is known and thus has an extreme probability) and the
    predicted values (which are actual probabilities since the true values
    are unknown). Since we are interested in predicting values that follow a
    bernoulli distribution, the cross entropy takes the form of the negative
    log-likelihood of the bernoulli distribution.
    
    With this cost function defined, Neural Networks are just performing
    a more BA version of maximum likelihood estimation. A negative cost is
    defined because maximizing the likelihood is the same as minimizing the
    negative likelihood. 
    
    Parameters
    ----------
    y_true : numpy array
        True, observed values. The outcome of an event (where 1 == success).
    y_pred : numpy array
        The predicted success probabilities.
    delta : float, optional
        A small, positive constant to offset predicted probabilities that are
        zero, which avoids log(0). Is ignored if delta = 0.
        The default is 1e-9.

    Returns
    -------
    cost : float
        The binary cross-entropy.

    '''

    # Compute the cross-entropy cost
    # To avoid log(0) errors (not necessary in most cases)
    ypred = y_pred.copy()
    if delta != 0:
        ypred[ypred <= delta] = delta
        ypred[ypred >= 1-delta] = 1-delta
    
    # m is the number of observations, and m_scale is a scaling factor to make
    # the computation easier in case the gradients are really big. 
    
    cost = -np.sum(xlogy(y_true, ypred))
    return cost


def cross_entropy_binary_der(y_true, y_pred, delta=1e-9):
    '''
    The derivative of binary cross-entropy.
    
    The derivative of the cross-entropy function (i.e. derivative of the
    error output) with respect to the predicted value (y_pred).
    The value inside the brackets is the value of the derivative of the
    log-likelihood for a bernoulli distribution. The negative is added
    because of our definition of cross-entropy.

    Parameters
    ----------
    y_true : numpy array
        True, observed values. The outcome of an event (where 1 == success).
    y_pred : numpy array
        The predicted success probabilities.
    delta : float, optional
        A small, positive constant to offset predicted probabilities that are
        zero, which avoids log(0). Is ignored if delta = 0.
        The default is 1e-9.

    Returns
    -------
    float
        The derivative of binary cross-entropy.

    '''
    # Compute the cross-entropy cost
    # To avoid log(0) errors (not necessary in most cases)
    ypred = y_pred.copy()
    if delta != 0:
        ypred[ypred <= delta] = delta
        ypred[ypred >= 1-delta] = 1-delta
    
    return -((y_true/ypred) - (1-y_true)/(1-ypred))


def cross_entropy_der(y_true, y_pred, delta=1e-9):
    '''
    The derivative of binary cross-entropy.
    
    The derivative of the cross-entropy function (i.e. derivative of the
    error output) with respect to the predicted value (y_pred).
    The value inside the brackets is the value of the derivative of the
    log-likelihood for a bernoulli distribution. The negative is added
    because of our definition of cross-entropy.

    Parameters
    ----------
    y_true : numpy array
        True, observed values. The outcome of an event (where 1 == success).
    y_pred : numpy array
        The predicted success probabilities.
    delta : float, optional
        A small, positive constant to offset predicted probabilities that are
        zero, which avoids log(0). Is ignored if delta = 0.
        The default is 1e-9.

    Returns
    -------
    float
        The derivative of binary cross-entropy.

    '''
    # Compute the cross-entropy cost
    # To avoid log(0) errors (not necessary in most cases)
    ypred = y_pred.copy()
    if delta != 0:
        ypred[ypred <= delta] = delta
        ypred[ypred >= 1-delta] = 1-delta
    
    return -(y_true/ypred)


def softmax_cross_entropy_der(y_true, y_pred, z=None):
    return y_pred - y_true


def softmax_cross_entropy_der_fullmath(y_true, y_pred, z,
                                       delta=1e-9, center=True):
    dce = cross_entropy_der(y_true=y_true, y_pred=y_pred,
                            delta=delta)
    dout = softmax_der(x=z, center=center)
    dZ = dce.reshape(1, -1).dot(dout).reshape(y_true.shape[0], -1)
    return dZ


def sigmoid_cross_entropy_binary_der(y_true, y_pred, z,
                                       delta=1e-9, center=True):
    dce = cross_entropy_binary_der(y_true=y_true, y_pred=y_pred,
                                   delta=delta)
    dout = sigmoid_der(x=z)
    dZ = dce*dout
    return dZ


def mse(y_true, y_score):
    mse = np.sum((y_true - y_score)**2)
    return mse


def mse_der(y_true, y_score):
    der = -2*(y_true - y_score)
    return der


def linear(x):
    return x


def linear_der(x):
    return 1


def mse_linear_der(y_true, y_score, z):
    mseder = mse_der(y_true, y_score)
    # There was no activation function for the linear layer,
    # there is no derivative to compute here
    # linear_der = 1
    # mseder = mseder*linear_der
    return mseder
