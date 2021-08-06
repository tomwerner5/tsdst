import numpy as np
from scipy.stats import truncnorm


def he_uniform(incoming, outgoing):
    """
    He uniform initialization. Used for Neural Network weight initialization. Generally used with a relu activation
    function.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(6/incoming),
                          high=np.sqrt(6/incoming))
    return W


def he_normal(incoming, outgoing):
    """
    He normal initialization. Used for Neural Network weight initialization. Generally used with a relu activation
    function.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(2/incoming))
    return W


def xavier_uniform(incoming, outgoing):
    """
    Xavier uniform initialization. Used for Neural Network weight initialization. Generally used with sigmoid, softmax,
    or tanh activation functions.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(6/(incoming + outgoing)),
                          high=np.sqrt(6/(incoming + outgoing)))
    return W


def xavier_normal(incoming, outgoing):
    """
    Xavier normal initialization. Used for Neural Network weight initialization. Generally used with sigmoid, softmax,
    or tanh activation functions.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(2/(incoming + outgoing)))
    return W


def lecun_uniform(incoming, outgoing):
    """
    Lecun uniform initialization. Used for Neural Network weight initialization.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(3/(incoming)),
                          high=np.sqrt(3/(incoming)))
    return W


def lecun_normal(incoming, outgoing):
    """
    Lecun normal initialization. Used for Neural Network weight initialization. Generally used with a selu activation
    function.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(1/incoming))
    return W
    

def random(incoming, outgoing, scale=0.01):
    """
    Random initialization. Uses a normal distribution for Neural Network weight initialization.

    Parameters
    ----------
    incoming : int
        Shape from the incoming layer (i.e. number of inputs from the previous layer)
    outgoing : int
        Shape outgoing from the current layer (i.e. number of outputs from the current layer)
    scale : float
        The standard deviation, or scale parameter, of a normal distribution

    Returns
    -------
    W : numpy array
        The initialized weights.
    """
    W = np.random.normal(size=(incoming, outgoing), scale=scale)
    return W
