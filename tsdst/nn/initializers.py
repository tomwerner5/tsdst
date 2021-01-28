import numpy as np
from scipy.stats import truncnorm


# He (uniform or normal) for relu variations
def he_uniform(incoming, outgoing):
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(6/(incoming)),
                          high=np.sqrt(6/(incoming)))
    return W


def he_normal(incoming, outgoing):
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(2/(incoming)))
    return W


# Xavier (uniform or normal) for sigmoid/softmax/tanh
def xavier_uniform(incoming, outgoing):
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(6/(incoming + outgoing)),
                          high=np.sqrt(6/(incoming + outgoing)))
    return W


def xavier_normal(incoming, outgoing):
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(2/(incoming + outgoing)))
    return W


def lecun_uniform(incoming, outgoing):
    W = np.random.uniform(size=(incoming, outgoing),
                          low=-np.sqrt(3/(incoming)),
                          high=np.sqrt(3/(incoming)))
    return W


# For selu
def lecun_normal(incoming, outgoing):
    W = truncnorm.rvs(size=(incoming, outgoing), a=-2, b=2,
                      scale=np.sqrt(1/(incoming)))
    return W
    

def random(incoming, outgoing):
    W = np.random.normal(size=(incoming, outgoing), scale=0.01)
    return W
