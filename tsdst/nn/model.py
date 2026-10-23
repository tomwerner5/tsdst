import numpy as np

from ..metrics import rpmse, bias, accuracy
from ..tmath import norm, norm_der
from .activations import (relu, elu, gelu, gelu_der, gelu_approx,
                     gelu_speedy, gelu_speedy_der,
                     sigmoid, sigmoid_der,
                     softmax, softmax_der, relu_der, cross_entropy,
                     cross_entropy_der, selu, selu_der, elu_der,
                     softmax_cross_entropy_der,
                     softmax_cross_entropy_der_fullmath,
                     sigmoid_cross_entropy_binary_der,
                     cross_entropy_binary, cross_entropy_binary_der,
                     linear, mse, mse_linear_der)
from .initializers import (he_uniform, he_normal, xavier_uniform,
                             xavier_normal, lecun_uniform, lecun_normal,
                             random_normal
                             )
from .optimizers import gradient_descent, adam


class NeuralNetwork(object):
    """The base class for custom neural networks in numpy
    """
    def __init__(self,
                 model,
                 eval_size=10,
                 batch_size=64,
                 num_iterations=500,
                 optimizer='adam',
                 optimizer_args=None,
                 m_scale=1,
                 bn_tol=1e-9,
                 bn_momentum=0,
                 scorer='accuracy',
                 shuffle=False,
                 print_cost=True,
                 random_state=42):
        r"""The constructor for the NeuralNetwork class.

        Parameters
        ----------
        model : dict
            A dictionary containing the model components, layers, and other
            specifications. The dictionary should have the following general
            structure:

            .. highlight:: python
            .. code-block:: python

                {
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

            Each layer should have the components defined above, however, not
            every component needs to be used (for example, setting
            dropout_keep_prob = 1 disables dropout). There can be as many
            hidden layers as desired (including none). Simply copy the
            'hidden0' sub-dictionary before the output layer to add a new
            hidden layer. However, the network must have an output layer
            defined. The key names for the layers can be anything, but the
            output layer must be positioned last.

            A description of each layer key is defined below:

                - activation (str or function): The activation function to be
                                              used. If custom function, it
                                              will pass the affine
                                              transformation of the current 
                                              layer as the first input to the
                                              function.
                - activation_args (dict) : An optional dictionary for passing
                                         additional arguments to the activation
                                         or derivative function. If there are
                                         none to pass, use an empty dictionary.
                                         For hidden layers, the derivative and
                                         activation arguments should be the
                                         same, so they share this dictionary.
                - cost (str or function): The cost function to be
                                        used. If custom function, it
                                        will pass the true Y values and
                                        the predicted Y values as the first
                                        two inputs to the function.
                - cost_args (dict) : An optional dictionary for passing
                                   additional arguments to the 
                                   cost function. If there are
                                   none to pass, use an empty dictionary.
                                   Only applies ot the output layer.
                - depth (int): The number of hidden nodes in the layer
                - derivative (str or function): The derivative of the combined
                                              cost and output layer activation
                                              function to be
                                              used. If custom function, it
                                              will pass the true Y values,
                                              the predicted Y values, and the
                                              non-activated output layer values
                                              as the first inputs to the 
                                              function.
                - derivative_args (dict) : An optional dictionary for passing
                                         additional arguments to the derivative
                                         function. If there are none to pass,
                                         use an empty dictionary. This only
                                         applies to the output layer.
                - dropout_keep_prob (float) : The proportion of nodes to keep at
                                            the respective layer. Between 0
                                            and 1. If dropping 10% of the
                                            nodes, the keep prob is 0.9
                - evaluation_metric (str or function) : An additional evaluation
                                                      metric to be used in 
                                                      training. This is only
                                                      used for printing an
                                                      additional output along
                                                      with cost at each epoch
                                                      or specified iteration to
                                                      track the training
                                                      progress
                - initializer (str or function) : The function to be used in 
                                                initializing the layer weights
                                                and biases. If custom, it must
                                                accept two arguments,  
                                                'incoming' and 'outgoing',
                                                which represent how many inputs
                                                are recieved from the previous
                                                layer, and how many outputs
                                                will be calculated at the 
                                                current layer.
                - lambda (dict) : A dictionary containing the regularization 
                                penalties for each type of regularization.

                                The options are:

                                    - Weight (float) : The kernel or weight
                                                     regularizer
                                                     (recommended for use)
                                    - activity (float) : A regularizer placed on
                                                       the activation function
                                                       output (experimental in
                                                       this code, not
                                                       recommended for use)
                                    - bias (float) : A regularizer for the bias
                                                   (not recommended for use for
                                                   theoretical reasons, but
                                                   should be correct to use)

                                A value of zero disables that regularization type for the layer.
                - lp_norm (dict) : A dictionary containing the regularization                                  norm funcitons for each type                                  regularization.

                                 The options are:

                                    - Weight (int) : The lp-norm for the weight
                                                   or kernel regularizer
                                    - activity (int) : The lp-norm for the
                                                     activity regularizer
                                    - bias (int) : The lp-norm for the bias
                                                 regularizer
                - use_batch_norm (bool) : If true, perform batch normalization
                                        on the current layer. For this
                                        implementation, the batch norm layer
                                        is placed before the activation
                                        function and before dropout (if used
                                        together)

        eval_size : int, optional
            The number of model evaluations to perform before printing
            an output. It is recommended that this number be
            `int(n/batch_size) + sum([n % batch_size != 0])` where n is the
            number of observations
        batch_size : int, optional
            The number of observations used to update the model at each step.
            The default is 64.
        num_iterations : int, optional
            The total number of full passes through the data to perform
            (i.e. the number of epochs). The default is 500.
        optimizer : str, optional
            The type of optimizer to use for gradient descent.
            The default is 'adam'.
        optimizer_args : dict, optional
            Arguments passed to the selected optimizer. If None, use
            ``learning_rate=0.001``, ``beta1=0.9``, ``beta2=0.999``, and ``eps=1e-8``.
        m_scale : float, optional
            An optional scaling parameter to scale up or down the cost and
            gradient values. For example, m_scale=2 will multiply the cost
            function by 0.5. The default is 1.
        bn_tol : float, optional
            The tolerance used in the batch norm equations.
            The default is 1e-9.
        bn_momentum : float, optional
            The momentum used in the exponential moving average for the mean
            and variance of the batch norm process. The default is 0.
        scorer : str or function, optional
            The function used in the score method. If custom, it must accept
            the true Y values and the predicted y values as the first two
            arguments of the function.
            The default is 'accuracy'.
        shuffle : bool, optional
            Shuffle the training set before training. The default is False.
        print_cost : bool, optional
            Print the cost (and possibly another metric) at each eval_step.
            The default is True.
        random_state : int, optional
            The random state of the process (for reproducibility).
            The default is 42.

        Returns
        -------
        None.
        """
        from copy import deepcopy
        self.model = deepcopy(model)
        self.num_layers = len(model.keys())
        self.eval_size = eval_size
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.optimizer_args = dict(learning_rate=.001, beta1=.9, beta2=.999, eps=1e-8) if optimizer_args is None else dict(optimizer_args)
        self.m_scale = m_scale
        self.bn_tol = bn_tol
        self.bn_momentum = bn_momentum
        self.print_cost = print_cost
        self.random_state = random_state
        self.shuffle = shuffle
        self.scorer = scorer
        self.activations = {'linear': linear,
                            'mse': mse,
                            'mse_linear_der': mse_linear_der,
                            'relu': relu,
                            'relu_der': relu_der,
                            'elu': elu,
                            'elu_der': elu_der,
                            'selu': selu,
                            'selu_der': selu_der,
                            'gelu': gelu,
                            'gelu_approx': gelu_approx,
                            'gelu_speedy': gelu_speedy,
                            'gelu_der': gelu_der,
                            'gelu_speedy_der': gelu_speedy_der,
                            'sigmoid': sigmoid,
                            'sigmoid_der': sigmoid_der,
                            'softmax': softmax,
                            'softmax_der': softmax_der,
                            'cross_entropy': cross_entropy,
                            'cross_entropy_der': cross_entropy_der,
                            'cross_entropy_binary': cross_entropy_binary,
                            'cross_entropy_binary_der': cross_entropy_binary_der,
                            'softmax_cross_entropy_der': softmax_cross_entropy_der,
                            'softmax_cross_entropy_der_fullmath': softmax_cross_entropy_der_fullmath,
                            'sigmoid_cross_entropy_binary_der': sigmoid_cross_entropy_binary_der
                            }
        self.initializers = {'he_uniform': he_uniform,
                             'he_normal': he_normal,
                             'xavier_uniform': xavier_uniform,
                             'xavier_normal': xavier_normal,
                             'lecun_uniform': lecun_uniform,
                             'lecun_normal': lecun_normal,
                             'random': random_normal
                             }
        self.optimizers = {'adam': adam,
                           'gradient_descent': gradient_descent}
        self.scorer_list = {'accuracy': accuracy,
                       'rpmse': rpmse,
                       'bias': bias
                       }
        self.optimizer = optimizer
        self.update_parameters = self.optimizers[optimizer]
    

    def initialize_wb(self, X):
        """
        Initialize the network.

        Parameters
        ----------
        X : numpy array
            The input data.

        Returns
        -------
        wb_list : dict
            A dictionary of the weights/biases for each layer.

        """
        wb_list = {}
        xrows = X.shape[0]
        X = X.reshape(xrows, -1)
        prev_col = X.shape[1]
    
        for k, key in enumerate(self.model.keys()):
            cur_row = prev_col
            cur_col = self.model[key]['depth']
            
            initializer = self.model[key]['initializer']
            init_func = self.initializers[initializer] if isinstance(initializer, str) else initializer
            
            # Weights cannot be initialized to zero, or the model can't train
            W = init_func(cur_row, cur_col)
            
            # Zero is a common bias initialization. Some argue that a small
            # positive value like 0.01 should be used instead. Others argue
            # that makes it worse. LSTMs typically initialize bias at 1
            b = np.zeros(shape=(1, cur_col))
            wb_list["Weight" + str(k)] = W
            wb_list["bias" + str(k)] = b
            if self.model[key]['use_batch_norm']:
                wb_list['gamma' + str(k)] = np.ones(shape=(1, cur_col))
                wb_list['beta' + str(k)] = np.zeros(shape=(1, cur_col))
            prev_col = cur_col
    
        return wb_list

    def forward_prop(self, X, wb, batch_norm, train=True, sample=False):
        """
        The forward propagation step

        Parameters
        ----------
        X : numpy array
            The input data.
        wb : dict
            A dictionary of the weights/biases for each layer.
        batch_norm : dict
            A dictionary containing the initial (or previous) batch norm
            results.
        train : bool, optional
            Whether the forward propagation method is being used to train or
            calculate the network in it's current state.
            The default is True.
        sample : bool, optional
            Whether or not the forward propagation is being used to generate
            random smaples of the output from the distribution created using
            dropout. The default is False.

        Returns
        -------
        hz : numpy array
            The final predicted output.
        zs : dict
            A dictionary of the linearly (affine transform) activated values
            for each layer.
        batch_norm : dict
            A dictionary containing the initial (or previous) batch norm
            results.
        hzs : dict
            A dictionary of the fully activated values for each layer
        dropout : dict
            A dictionary of the dropout status for each layer.
        regularizers : dict
            A dictionary of the regularization status for each layer.

        """
        hz = X
        zs = {}
        hzs = {}
        dropout = {}
        regularizers = {}
        
        for i, key in enumerate(self.model.keys()):
            indx = str(i)
            z = hz.dot(wb["Weight" + indx]) + wb["bias" + indx]
            
            # It is a highly debated topic whether or not batch norm or dropout
            # first, and on top of that, whether batch norm should occur before
            # or after activation. It likely depends on specific situations on
            # types of networks. Other sources say they shouldn't be used
            # together anyway. Since this is a demonstration, I chose to put
            # batch norm first, before acivation. Note: moving them around 
            # will affect the derivative, so if doing by hand and not using
            # autograd etc., watch out for this.
            
            if self.model[key]['use_batch_norm']:
                if train:
                    batch_norm['mu' + indx] = np.mean(z, axis=0)
                    batch_norm['var' + indx] = np.var(z, axis=0)
                    batch_norm['z_mu' + indx] = z - batch_norm['mu' + indx]
                    batch_norm['std' + indx] = np.sqrt(batch_norm['var' + indx] + self.bn_tol)
                    batch_norm['xhat' + indx] = batch_norm['z_mu' + indx]/batch_norm['std' + indx] 
                    batch_norm['norm_z' + indx] = wb['gamma' + indx]*batch_norm['xhat' + indx] + wb['beta' + indx]
                    
                    # Exponential running mean for mu/var, if desired. Use
                    # momentum = 0 for regular batch norm process
                    batch_norm['mu_r' + indx] = self.bn_momentum*batch_norm['mu_r' + indx] + (1 - self.bn_momentum)*batch_norm['mu' + indx]
                    batch_norm['var_r' + indx] = self.bn_momentum*batch_norm['var_r' + indx] + (1 - self.bn_momentum)*batch_norm['var' + indx]
                else:
                    batch_norm['xhat' + indx] = (z - batch_norm['mu_r' + indx])/np.sqrt(batch_norm['var_r' + indx] + self.bn_tol)
                    batch_norm['norm_z' + indx] = wb['gamma' + indx]*batch_norm['xhat' + indx] + wb['beta' + indx]
            else:
                batch_norm['norm_z' + indx] = z
            
            activation = self.model[key]['activation']
            actf = self.activations[activation] if isinstance(activation, str) else activation
            hz = actf(batch_norm['norm_z' + indx], **self.model[key]['activation_args'])
            
            if i != self.num_layers - 1:
                if train or sample:
                    dropout_keep_prob = self.model[key]['dropout_keep_prob']
                else:
                    dropout_keep_prob = 1
                drop_mask = np.random.uniform(size=hz.shape) if dropout_keep_prob < 1 else np.zeros(hz.shape)
                dropout["Weight" + indx] = drop_mask <= dropout_keep_prob
                hz = (hz*dropout["Weight" + indx])/dropout_keep_prob
                
            lamda_w = self.model[key]['lambda']['Weight']
            p_w = self.model[key]['lp_norm']['Weight']
            lamda_a = self.model[key]['lambda']['activity']
            p_a = self.model[key]['lp_norm']['activity']
            lamda_b = self.model[key]['lambda']['bias']
            p_b = self.model[key]['lp_norm']['bias']
            
            # kernel/weight regularizer
            # Note: information here is only recorded, it does not affect
            # the forward propagation calculations at all
            if train:
                regularizers["Weight" + indx] = (1/p_w)*lamda_w*norm(wb["Weight" + indx], p_w)
                regularizers["activity" + indx] = (1/p_a)*lamda_a*norm(hz, p_a)
                regularizers["bias" + indx] = (1/p_b)*lamda_b*norm(wb["bias" + indx], p_b)
                
            zs['z' + str(i)] = z
            hzs['hz' + str(i)] = hz
        return hz, zs, batch_norm, hzs, dropout, regularizers
        
    def back_prop(self, X, Y, wb, zs, batch_norm, hzs, dropout):
        r"""Differentiate the recorded network loss with respect to all fitted parameters.

        Parameters
        ----------
        X : numpy.ndarray
            Feature batch; rows are observations.
        Y : numpy.ndarray
            True output values for the batch.
        wb : dict
            Layer weights, biases, and any batch-normalization gamma/beta parameters.
        zs : dict
            Raw affine values returned by forward_prop. Retained in this interface;
            activation derivatives use batch_norm's norm_z values because batch
            normalization precedes activation.
        batch_norm : dict
            Recorded normalization means, variances, standardized values, standard
            deviations, and activation inputs for this forward pass.
        hzs : dict
            Layer activations, including hidden-layer dropout scaling.
        dropout : dict
            Hidden-layer masks from the same forward pass.

        Returns
        -------
        dwdb : dict
            Gradients with the same parameter names and shapes as wb, including
            normalization gamma/beta when present.

        Notes
        -----
        The differentiated objective is (summed data loss plus the configured
        weight, activity, and bias penalties)/(batch_size*m_scale). The output
        loss derivative introduces that normalization once. Hidden layers propagate
        it through their activation, dropout, and normalization steps; they do not
        divide the data gradient by batch size again.

        Activity penalties are differentiated at the stored activation, before
        propagating through dropout and activation. Weight and bias penalties use
        the configured lambda/p scaling. Batch-normalization derivatives use the
        recorded standardized inputs and statistics from the same training pass.
        """
        gradients = {}
        normalizer = X.shape[0]*self.m_scale
        keys = list(self.model)
        upstream = None
        for i in range(self.num_layers-1, -1, -1):
            layer = self.model[keys[i]]
            index = str(i)
            activation_input = batch_norm['norm_z'+index]
            activated = hzs['hz'+index]
            activity = layer['lambda']['activity'] / layer['lp_norm']['activity']
            activity_gradient = activity*norm_der(activated, layer['lp_norm']['activity'])/normalizer
            if i == self.num_layers-1:
                derivative = layer['derivative']
                function = self.activations[derivative] if isinstance(derivative, str) else derivative
                dZ = function(Y, activated, activation_input, **layer['derivative_args'])/normalizer
                if activity:
                    activation = layer['activation']
                    if activation == 'softmax':
                        dZ += activated*(activity_gradient-np.sum(activity_gradient*activated, axis=1, keepdims=True))
                    elif activation == 'linear':
                        dZ += activity_gradient
                    else:
                        derivative_name = activation + '_der' if isinstance(activation, str) else None
                        if derivative_name not in self.activations:
                            raise ValueError('Output activity regularization needs a known activation derivative')
                        dZ += activity_gradient*self.activations[derivative_name](activation_input, **layer['activation_args'])
            else:
                dA = upstream + activity_gradient
                dA = dA*dropout['Weight'+index]/layer['dropout_keep_prob']
                derivative = layer['derivative']
                function = self.activations[derivative] if isinstance(derivative, str) else derivative
                if layer['activation'] == 'softmax':
                    probabilities = self.activations['softmax'](activation_input, **layer['activation_args'])
                    dZ = probabilities*(dA-np.sum(dA*probabilities, axis=1, keepdims=True))
                else:
                    dZ = dA*function(activation_input, **layer['activation_args'])
            if layer['use_batch_norm']:
                xhat = batch_norm['xhat'+index]
                gradients['gamma'+index] = np.sum(dZ*xhat, axis=0, keepdims=True)
                gradients['beta'+index] = np.sum(dZ, axis=0, keepdims=True)
                scaled = dZ*wb['gamma'+index]
                # Propagate through the recorded batch-normalization transform.
                dZ = (scaled-scaled.mean(axis=0)-xhat*np.mean(scaled*xhat, axis=0))/batch_norm['std'+index]
            # The data gradient already includes the batch mean from the loss derivative.
            # Apply that normalization once; weight and bias penalties carry their own scale.
            previous = X if i == 0 else hzs['hz'+str(i-1)]
            for parameter, data_gradient in (('Weight', previous.T @ dZ), ('bias', np.sum(dZ, axis=0, keepdims=True))):
                power = layer['lp_norm'][parameter]
                penalty_gradient = layer['lambda'][parameter]*norm_der(wb[parameter+index], power)/(power*normalizer)
                gradients[parameter+index] = data_gradient+penalty_gradient
            upstream = dZ @ wb['Weight'+index].T
        return gradients


    def fit(self, X, Y):
        """
        Fits the Neural Network.

        Parameters
        ----------
        X : numpy array
            The input data.
        Y : numpy array
            The true Y values.

        Returns
        -------
        self
            The fitted model.

        """
        # This is only here for debugging or reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Store total training observations
        self.m = X.shape[0]
        
        # Reshape Y so that it is a matrix, even in 1D
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y).reshape(self.m, -1)
        self.model['output']['depth'] = Y.shape[1]
        
        # Check for an additional metric to evaluate other than cost
        output_layer_components = list(self.model['output'].keys())
        has_metric = False
        if 'evaluation_metric' in output_layer_components:
            if isinstance(self.model['output']['evaluation_metric'], str):
                met_func = self.scorer_list[self.model['output']['evaluation_metric']]
            else:
                met_func = self.model['output']['evaluation_metric']
            has_metric = True
        
        # Initialize the graph
        wb = self.initialize_wb(X)
        
        # Total batches needed for minibatch
        if self.batch_size > self.m:
            total_batches = 1
        else:
            total_batches = int(self.m/self.batch_size) + sum([self.m % self.batch_size != 0])
        
        # Initialize result arrays and weights
        costs = np.zeros(self.num_iterations*total_batches)
        metric = np.zeros(self.num_iterations*total_batches)
        
        batch_norm = {'mu_r'+str(i): 0 for i in range(self.num_layers)}
        batch_norm.update({'var_r'+str(i): 1 for i in range(self.num_layers)})
        
        batches = [(b*self.batch_size, (b+1)*self.batch_size) for b in range(total_batches)]
        
        # TODO: Figure out if this (and below) is the proper way to do minibatch
        if self.shuffle:
            rand_indx = np.random.permutation(X.shape[0])
            X = X[rand_indx, :]
            Y = Y[rand_indx, :] 

        # marker for counting total iterations in minibatch        
        count = 0
        for i in range(self.num_iterations):
            # TODO: figure out if each batch is meant to be random in minibatch
            #batch_order = np.random.choice(range(total_batches), total_batches)
            batch_order = range(total_batches)
            for j in batch_order:
                
                X_batch = X[batches[j][0]:batches[j][1], :]
                Y_batch = Y[batches[j][0]:batches[j][1], :]
                
                batch_m = X_batch.shape[0]
                
                # Forward propagation. 
                (hz, zs, batch_norm, hzs, dropout, regularizers) = self.forward_prop(X_batch,
                                                                                     wb,
                                                                                     batch_norm,
                                                                                     train=True,
                                                                                     sample=False)
                
                # Calculate cost without regularization
                cost_function = self.model['output']['cost']
                costf = self.activations[cost_function] if isinstance(cost_function, str) else cost_function
                cost = costf(Y_batch, hz,
                             **self.model['output']['cost_args'])
                
                # Get regularization information for the cost function,
                # the activity, weight, and bias regularizers
                norms_w = np.sum([regularizers["Weight"+str(l)] for l in range(self.num_layers)])
                norms_a = np.sum([regularizers["activity"+str(l)] for l in range(self.num_layers)]) 
                norms_b = np.sum([regularizers["bias"+str(l)] for l in range(self.num_layers)]) 
                
                # Update Cost with regularization
                cost = (cost + norms_w + norms_a + norms_b)/(batch_m*self.m_scale)
                
                # Backpropagation.
                dwdb = self.back_prop(X_batch, Y_batch, wb, zs, 
                                      batch_norm, hzs, dropout)
                
                # Update parameters with optimizing function
                if self.optimizer == 'adam':
                    if count == 0:
                        self.optimizer_args['mt'] = {}
                        self.optimizer_args['vt'] = {}
                        for key in wb.keys():
                            self.optimizer_args['mt'][key] = np.zeros(wb[key].shape)
                            self.optimizer_args['vt'][key] = np.zeros(wb[key].shape)
                    self.optimizer_args['i'] = count
                
                # Update weights, biases, and other parameters for batch norm
                wb = self.update_parameters(wb, dwdb, self.optimizer_args)
                
                # Store cost and other metric, if applicable
                costs[count] = cost
                if has_metric:
                    metric[count] = met_func(Y_batch, hz)
    
                # Print the cost/metric every eval_size iterations
                if self.print_cost and (count + 1) % self.eval_size == 0:
                    if has_metric:
                        print("Evaluation %i: Cost: %f, Metric: %f" % (int((count + 1)/self.eval_size),
                                                                      costs[((count + 1)-self.eval_size):(count + 1)].mean(),
                                                                      metric[((count + 1)-self.eval_size):(count + 1)].mean()))
                    else:
                        print("Evaluation %i: Cost: %f" % (int((count + 1)/self.eval_size),
                                                          costs[((count + 1)-self.eval_size):(count + 1)].mean()))
                count += 1
                
        self.wb = wb
        self.dwdb = dwdb
        self.costs = costs
        self.batch_norm = batch_norm
        if has_metric:
            self.metric = metric
        return self
    
    def predict(self, X):
        r"""Predict output values for the supplied feature rows.

        Parameters
        ----------
        X : array-like
            Feature matrix in the fitted column order.

        Returns
        -------
        hz : numpy.ndarray
            Output activations. Classification outputs are probabilities; linear
            outputs retain the fitted response scale.

        Notes
        -----
        Prediction uses the stored running batch-normalization statistics and
        turns off dropout. Temporary forward-pass values do not modify the stored
        normalization dictionary, and deterministic inference does not draw dropout
        masks from the RNG.
        """
        from copy import deepcopy
        output, _, _, _, _, _ = self.forward_prop(np.asarray(X), self.wb, deepcopy(self.batch_norm),
                                                 train=False, sample=False)
        return output

    def draw_predictive_samples(self, X, n_samples=1000, n_outputs=None):
        r"""Draw repeated predictions with dropout enabled during inference.

        Parameters
        ----------
        X : array-like
            Feature rows in fitted column order.
        n_samples : int, optional
            Positive number of dropout draws. The default is 1000.
        n_outputs : int, optional
            Expected output width. If supplied, it must match the fitted network.
            None infers the width from the fitted output layer. The default is None.

        Returns
        -------
        draws : numpy.ndarray
            Shape (n_observations, n_samples*n_outputs). Consecutive output columns
            belong to one draw. Each draw keeps the original observation order.

        Notes
        -----
        Hidden-layer dropout masks vary between draws. The stored training
        normalization statistics stay fixed, as in predict. This method samples
        the dropout approximation to predictive model uncertainty; it does not add
        observation noise or sample arbitrary response distributions. With dropout
        keep probability one, repeated predictions are deterministic.
        """
        from copy import deepcopy
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError('n_samples must be a positive integer')
        width = self.model['output']['depth']
        if n_outputs is not None and n_outputs != width:
            raise ValueError('n_outputs must match the fitted network output count')
        draws = np.empty((len(X), n_samples*width))
        for i in range(n_samples):
            output, _, _, _, _, _ = self.forward_prop(np.asarray(X), self.wb, deepcopy(self.batch_norm),
                                                     train=False, sample=True)
            draws[:, i*width:(i+1)*width] = output
        return draws
    
    def score(self, X, y):
        """
        A method to score/test the model based on the inputs X, y. This can
        be a custom function or called with a string from something built-in.

        Parameters
        ----------
        X : numpy array
            The input data.
        y : numpy array
            The true Y values.

        Returns
        -------
        float or numpy array
            The score result for X, y.

        """
        if isinstance(self.scorer, str):
            score_func = self.scorer_list[self.scorer]
        else:
            score_func = self.scorer
        y_score = self.predict(X)
        return score_func(y, y_score)
