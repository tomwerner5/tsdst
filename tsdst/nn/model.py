import numpy as np

from ..metrics import rpmse, bias, accuracy
from ..tmath import norm, norm_der
from .activations import ( relu, elu,
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
                             random
                             )
from .optimizers import gradient_descent, adam


class NeuralNetwork(object):
    def __init__(self,
                 model,
                 eval_size=10,
                 batch_size=64,
                 num_iterations=500,
                 optimizer='adam',
                 optimizer_args={'learning_rate': 0.001,
                                 'beta1': 0.9,
                                 'beta2': 0.999,
                                 'eps': 1e-8},
                 m_scale=1,
                 bn_tol=1e-9,
                 bn_momentum=0,
                 scorer='accuracy',
                 shuffle=False,
                 print_cost=True,
                 random_state=42):
        '''
        The constructor for the NeuralNetwork class.

        Parameters
        ----------
        model : dict
            A dictionary containing the model components, layers, and other
            specifications. The dictionary should have the following general
            structure:
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
            'hidden1' sub-dictionary before the output layer to add a new
            hidden layer. However, the network must have an output layer
            defined. The key names for the layers can be anything, but the
            output layer must be positioned last.
            
            A description of each layer key is defined below:

                activation (str or function): The activation function to be
                                              used. If custom function, it
                                              will pass the affine
                                              transformation of the current 
                                              layer as the first input to the
                                              function.
                activation_args (dict) : An optional dictionary for passing
                                         additional arguments to the activation
                                         or derivative function. If there are
                                         none to pass, use an empty dictionary.
                                         For hidden layers, the derivative and
                                         activation arguments should be the
                                         same, so they share this dictionary.
                cost (str or function): The cost function to be
                                        used. If custom function, it
                                        will pass the true Y values and
                                        the predicted Y values as the first
                                        two inputs to the function.
                cost_args (dict) : An optional dictionary for passing
                                   additional arguments to the 
                                   cost function. If there are
                                   none to pass, use an empty dictionary.
                                   Only applies ot the output layer.
                depth (int): The number of hidden nodes in the layer
                derivative (str or function): The derivative of the combined
                                              cost and output layer activation
                                              function to be
                                              used. If custom function, it
                                              will pass the true Y values,
                                              the predicted Y values, and the
                                              non-activated output layer values
                                              as the first inputs to the 
                                              function.
                derivative_args (dict) : An optional dictionary for passing
                                         additional arguments to the derivative
                                         function. If there are none to pass,
                                         use an empty dictionary. This only
                                         applies to the output layer.
                dropout_keep_prob (float) : The proportion of nodes to keep at
                                            the respective layer. Between 0
                                            and 1. If dropping 10% of the
                                            nodes, the keep prob is 0.9
                evaluation_metric (str or function) : An additional evaluation
                                                      metric to be used in 
                                                      training. This is only
                                                      used for printing an
                                                      additional output along
                                                      with cost at each epoch
                                                      or specified iteration to
                                                      track the training
                                                      progress
                initializer (str or function) : The function to be used in 
                                                initializing the layer weights
                                                and biases. If custom, it must
                                                accept two arguments,  
                                                'incoming' and 'outgoing',
                                                which represent how many inputs
                                                are recieved from the previous
                                                layer, and how many outputs
                                                will be calculated at the 
                                                current layer.
                lambda (dict) : A dictionary containing the regularization 
                                penalties for each type of regularization.
                                The options are:
                                    Weight (float) : The kernel or weight
                                                     regularizer
                                                     (recommended for use)
                                    activity (float) : A regularizer placed on
                                                       the activation function
                                                       output (experimental in
                                                       this code, not
                                                       recommended for use)
                                    bias (float) : A regularizer for the bias
                                                   (not recommended for use for
                                                   theoretical reasons, but
                                                   should be correct to use)
                                
                                A value of zero for any of the lambdas will
                                that regularization type for that layers.
                lp_norm (dict) : A dictionary containing the regularization 
                                 norm funcitons for each type of
                                 regularization.
                                 
                                 The options are:
                                    Weight (int) : The lp-norm for the weight
                                                   or kernel regularizer
                                    activity (int) : The lp-norm for the
                                                     activity regularizer
                                    bias (int) : The lp-norm for the bias
                                                 regularizer
                use_batch_norm (bool) : If true, perform batch normalization
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
            Optional arguments to send to the optimizer (learning rate, etc.). 
            The default is {'learning_rate': 0.001,
                            'beta1': 0.9,
                            'beta2': 0.999,
                            'eps': 1e-8}.
        m_scale : float, optional
            An optional scaling paramter to scale up or down the cost and
            gradient values. The default is 1.
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

        '''
        self.model = model
        self.num_layers = len(model.keys())
        self.eval_size = eval_size
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.optimizer_args = optimizer_args
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
                            'selu_dir': selu_der,
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
                             'random': random
                             }
        self.optimizers = {'adam': adam,
                           'gradient_descent': gradient_descent}
        self.scorer = {'accuracy': accuracy,
                       'rpmse': rpmse,
                       'bias': bias
                       }
        self.optimizer = 'adam'
        self.update_parameters = self.optimizers[optimizer]
    

    def initialize_wb(self, X):
        '''
        Initialize the network.

        Parameters
        ----------
        X : numpy array
            The input data.

        Returns
        -------
        wb_list : dict
            A dictionary of the weights/biases for each layer.

        '''
        wb_list = {}
        xrows = X.shape[0]
        X = X.reshape(xrows, -1)
        prev_col = X.shape[1]
    
        for k, key in enumerate(self.model.keys()):
            cur_row = prev_col
            cur_col = self.model[key]['depth']
            
            init_func = self.initializers[self.model[key]['initializer']]
            
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
        '''
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

        '''
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
            
            actf = self.activations[self.model[key]['activation']]
            hz = actf(batch_norm['norm_z' + indx], **self.model[key]['activation_args'])
            
            if i != self.num_layers - 1:
                if train or sample:
                    dropout_keep_prob = self.model[key]['dropout_keep_prob']
                else:
                    dropout_keep_prob = 1
                drop_mask = np.random.uniform(size=hz.shape)
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
        '''
        The backward propagation step.

        Parameters
        ----------
        X : numpy array
            The input data.
        Y : numpy array
            The true Y values.
        wb : dict
            A dictionary of the weights/biases for each layer.
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

        Returns
        -------
        dwdb : dict
            A dictionary of the gradients with respect to the weights and
            biases.

        '''
        dwdb = {}
        batch_m = X.shape[0]
        keys = list(self.model.keys())
        for i in range(self.num_layers - 1, -1, -1):
            lamda_w = self.model[keys[i]]['lambda']['Weight']
            p_w = self.model[keys[i]]['lp_norm']['Weight']
            lamda_a = self.model[keys[i]]['lambda']['activity']
            p_a = self.model[keys[i]]['lp_norm']['activity']
            lamda_b = self.model[keys[i]]['lambda']['bias']
            p_b = self.model[keys[i]]['lp_norm']['bias']
            
            if i == self.num_layers - 1:
                dcostoutf = self.activations[self.model[keys[i]]['derivative']]
                dZ = dcostoutf(Y, hzs["hz" + str(i)], zs["z" + str(i)],
                               **self.model[keys[i]]['derivative_args'])/(batch_m*self.m_scale)
                dZ += lamda_a*norm_der(hzs["hz" + str(i)], p_a)/(batch_m*self.m_scale)
                
                # Batchnorm step, if applicable
                if self.model[keys[i]]['use_batch_norm']:
                    dxhat = dZ * wb["gamma" + str(i)]
                    dvar = -0.5*np.sum(dxhat*batch_norm["z_mu"+str(i)], axis=0)*(1/batch_norm["std"+str(i)]**3)
                    dxdstd = dxhat/batch_norm["std"+str(i)]
                    dmu = -np.sum(dxdstd, axis=0) 
                    dmu -= 2*dvar*np.mean(batch_norm["z_mu"+str(i)], axis=0)
                    
                    dgamma = np.sum(dZ*batch_norm["xhat"+str(i)], axis=0)
                    dbeta = np.sum(dZ, axis=0)
                    
                    dwdb["gamma" + str(i)] = dgamma
                    dwdb["beta" + str(i)] = dbeta
                    
                    dZ = dxdstd + 2*dvar*batch_norm["z_mu"+str(i)]/zs["z"+str(i)].shape[0]
                    dZ += dmu/zs["z"+str(i)].shape[0]

            else:
                dZ = dZ.dot(wb["Weight" + str(i + 1)].T)
                
                dropout_keep_prob = self.model[keys[i]]['dropout_keep_prob']
                dZ = dZ * dropout["Weight" + str(i)]/dropout_keep_prob
                
                dactf = self.activations[self.model[keys[i]]['derivative']]

                dZ = dZ * dactf(zs["z" + str(i)],
                                **self.model[keys[i]]['activation_args'])
                dZ = dZ + lamda_a*norm_der(hzs["hz" + str(i)], p_a)/(batch_m*self.m_scale)
                
                # Batchnorm step, if applicable
                if self.model[keys[i]]['use_batch_norm']:
                    dxhat = dZ * wb["gamma" + str(i)]
                    dvar = -0.5*np.sum(dxhat*batch_norm["z_mu"+str(i)], axis=0)*(1/batch_norm["std"+str(i)]**3)
                    dxdstd = dxhat/batch_norm["std"+str(i)]
                    dmu = -np.sum(dxdstd, axis=0) 
                    dmu -= 2*dvar*np.mean(batch_norm["z_mu"+str(i)], axis=0)
                    
                    dgamma = np.sum(dZ*batch_norm["xhat"+str(i)], axis=0)
                    dbeta = np.sum(dZ, axis=0)
                    
                    dwdb["gamma" + str(i)] = dgamma
                    dwdb["beta" + str(i)] = dbeta
                    
                    dZ = dxdstd + 2*dvar*batch_norm["z_mu"+str(i)]/zs["z"+str(i)].shape[0]
                    dZ += dmu/zs["z"+str(i)].shape[0]
                
            if i == 0:
                A = X
            else:
                A = hzs["hz" + str(i - 1)]
                
            # m doesn't come from the derivative. We are dividing by m because we
            # technically calculated the gradient for each observation. Thus, we
            # average them to get the overall impact. We could also just sum and 
            # not use the mean, but having a lower value is helpful to not run
            # into errors with large numbers
            
            dW = (batch_m*self.m_scale)*(np.dot(A.T, dZ) + lamda_w*norm_der(wb["Weight" + str(i)], p_w)/p_w)
            dB = (batch_m*self.m_scale)*(np.sum(dZ, axis=0, keepdims=True) + lamda_b*norm_der(wb["bias" + str(i)], p_b)/p_b)
            dwdb["Weight" + str(i)] = dW
            dwdb["bias" + str(i)] = dB
        return dwdb


    def fit(self, X, Y):
        '''
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

        '''
        # This is only here for debugging or reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Store total training observations
        self.m = X.shape[0]
        
        # Reshape Y so that it is a matrix, even in 1D
        Y = Y.reshape(self.m, -1)
        self.model['output']['depth'] = Y.shape[1]
        
        # Check for an additional metric to evaluate other than cost
        output_layer_components = list(self.model['output'].keys())
        has_metric = False
        if 'evaluation_metric' in  output_layer_components:
            if isinstance(self.model['output']['evaluation_metric'], str):
                met_func = self.scorer[self.model['output']['evaluation_metric']]
            else:
                met_func = self.model['output']['evaluation_metric']
            has_metric = True
        
        # Initialize the graph
        wb = self.initialize_wb(X)
        
        # Total batches needed for minibatch
        if self.batch_size > self.m:
            total_batches = self.m
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
            rand_indx = np.random.choice(range(X.shape[0]), X.shape[0])
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
                costf = self.activations[self.model['output']['cost']]
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
                    if i == 0:
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
        '''
        Predict the Y values based on the input X.

        Parameters
        ----------
        X : numpy array
            The input data.

        Returns
        -------
        hz : numpy array
            The predicted Y values.

        '''
        hz, _, _, _, _, _ = self.forward_prop(X, self.wb, self.batch_norm,
                                           train=False, sample=False)
        return hz
    
    def draw_predictive_samples(self, X, n_samples=1000, n_outputs=1):
        '''
        Draws predictive samples for each observation from the distribution
        created by using dropout.
        
        The rows of the output represent the observations in X, and the columns
        of the output represent the number of samples drawn. If Y has multiple
        outputs or classes for a single observation, then the shape of the
        output will be (X.shape[0], Y.shape[1]*n_samples). For example, if
        there are four observations in X, two classes in Y, and you want three
        samples, then the output will be:
            
            [[Y11_c1, Y11_c2] [Y12_c1, Y12_c2], [Y13_c1, Y13_c2]
             [Y21_c1, Y21_c2] [Y22_c1, Y22_c2], [Y23_c1, Y23_c2]
             [Y31_c1, Y31_c2] [Y32_c1, Y32_c2], [Y33_c1, Y33_c2]
             [Y41_c1, Y41_c2] [Y42_c1, Y42_c2], [Y43_c1, Y43_c2]]

        Parameters
        ----------
        X : numpy array
            The input data.
        n_samples : int, optional
            The number of samples to draw. The default is 1000.
        n_outputs : int, optional
            The number of outputs or classes in Y. The default is 1.

        Returns
        -------
        draws : numpy array
            The sampled predicted values.

        '''
        draws = np.zeros(shape=(X.shape[0], n_samples*n_outputs))
        for i in range(n_samples):
            hz, _, _, _, _ = self.forward_prop(X, self.wb, self.batch_norm,
                                               train=True, sample=True)
            draws[:, (i*n_outputs):((i+1)*n_outputs)] = hz
        return draws
    
    def score(self, X, y):
        '''
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

        '''
        if isinstance(self.scorer, str):
            score_func = self.scorer_list[self.scorer]
        else:
            score_func = self.scorer
        y_score = self.predict(X, self.wb)
        return score_func(y, y_score)
