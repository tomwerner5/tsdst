import numpy as np


def gradient_descent(wb, dwdb, optimizer_args):
    '''
    The simple gradient descent update.

    Parameters
    ----------
    wb : dict
        A dictionary of the weights/biases for each layer.
    dwdb : dict
        A dictionary of the gradients with respect to the weights and
        biases.
    optimizer_args : dict
        Optional optimizer configurations.

    Returns
    -------
    wb : dict
        A dictionary of the updated weights and biases.

    '''
    for key in wb.keys():
        wb[key] = wb[key] - optimizer_args['learning_rate']*dwdb[key]
    return wb


def adam(wb, dwdb, optimizer_args):
    '''
    The Adam optimizer. (Adaptive Moment Estimation)

    Parameters
    ----------
    wb : dict
        A dictionary of the weights/biases for each layer.
    dwdb : dict
        A dictionary of the gradients with respect to the weights and
        biases.
    optimizer_args : dict
        Optional optimizer configurations.

    Returns
    -------
    wb : dict
        A dictionary of the updated weights and biases.

    '''
    for key in wb.keys():
        optimizer_args['mt'][key] = (optimizer_args['beta1']*optimizer_args['mt'][key] +
                                     (1-optimizer_args['beta1'])*dwdb[key])
        optimizer_args['vt'][key] = (optimizer_args['beta2']*optimizer_args['vt'][key] +
                                     (1-optimizer_args['beta2'])*(dwdb[key]**2))
        mhat = optimizer_args['mt'][key]/(1-(optimizer_args['beta1']**(optimizer_args['i']+1)))
        vhat = optimizer_args['vt'][key]/(1-(optimizer_args['beta2']**(optimizer_args['i']+1)))
        wb[key] = wb[key] - optimizer_args['learning_rate']*(mhat/(np.sqrt(vhat)+optimizer_args['eps']))
    return wb