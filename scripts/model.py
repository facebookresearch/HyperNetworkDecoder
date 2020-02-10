# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
import numpy as np


class HyperNetworkDecoder():

    def __init__(self, args, data):

        self.data = data
        self.args = args

        # F network
        init_xav = tf.contrib.layers.xavier_initializer()
        self.f_weights  = {
            'f_h1': tf.get_variable('f_h1', [args.sf_n_input, args.sf_n_hidden_1], initializer=init_xav),
            'f_h2': tf.get_variable('f_h2', [args.sf_n_hidden_1, args.sf_n_hidden_2], initializer=init_xav),
            'f_h3': tf.get_variable('f_h3', [args.sf_n_hidden_2, args.sf_n_hidden_3], initializer=init_xav),
            'f_h4': tf.get_variable('f_h4', [args.sf_n_hidden_3, args.sf_n_hidden_3], initializer=init_xav),
            'f_head1': tf.get_variable('f_head1',[args.sf_n_hidden_3, args.n_input*args.n_hidden_1], initializer=init_xav),
            'f_head2': tf.get_variable('f_head2',[args.sf_n_hidden_3, args.n_hidden_2], initializer=init_xav)
        }

        # Input/Output placeholders
        self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.code_n])
        self.y = tf.placeholder(tf.float32, shape=[args.batch_size, args.code_n])

        # Generate matrices for graph construction
        self.generate_graph_matrix(args)
        self.W_odd2even_graphnn_var = tf.Variable(self.W_odd2even_graphnn)
        self.W_output_var = tf.Variable(self.W_output.copy())

        # Input layer
        self.x_tile = tf.tile(self.x, multiples=[1, args.n_odd])
        self.W_input = tf.reshape(self.W_input.transpose(), [-1])
        self.x_tile = tf.multiply(self.x_tile, self.W_input)
        self.x_tile = tf.reshape(self.x_tile, [args.batch_size, args.n_odd, args.code_n])
        self.u_i = tf.tanh(0.5*tf.clip_by_value(self.x_tile, clip_value_min=-10, clip_value_max=10))
        self.u_i = tf.add(self.u_i, 1-tf.to_float(tf.abs(self.u_i) > 0))
        self.z_input = tf.reduce_prod(self.u_i, reduction_indices=2)
        self.x_hv = tf.log(tf.div(1+self.z_input, 1-self.z_input))

        self.net_dict = {}
        self.arg_loss = 0
        for i in range(0, args.num_hidden_layers-1, 1):

            # parity layer
            fw1, fw2 = self.f_hyper(tf.abs(self.x_hv), self.f_weights)
            self.x_hv_c = tf.expand_dims(self.x_hv, 1)

            f_weights = {'h1':tf.reshape(fw1, [args.batch_size, args.n_input, args.n_hidden_1]), 'out':tf.reshape(fw2, [args.batch_size, args.n_hidden_2, 1])}
            self.W_odd2even_graphnn_var = tf.multiply(self.W_odd2even_graphnn, self.W_odd2even_graphnn_var)
            self.x_input_tile = tf.einsum('aij,bjk->abik', self.x_hv_c, self.W_odd2even_graphnn_var)
            self.x_hv_c = tf.squeeze(self.x_input_tile, 2)

            self.x_sc = tf.expand_dims(tf.matmul(self.x, self.W_skipconn2even), 2)
            self.x_all = tf.concat([self.x_hv_c, self.x_sc], 2)
            self.x_hp = self.mlp_vn(self.x_all, f_weights)

            # check layer
            self.x_hv_c = tf.tile(self.x_hp, multiples=[1, args.n_odd])
            self.x_hv_c = tf.multiply(self.x_hv_c, tf.reshape(self.W_even2odd.transpose(), [-1]))
            self.x_hv_c = tf.reshape(self.x_hv_c,[args.batch_size, args.n_odd, args.n_even])
            self.x_hv_c = tf.add(self.x_hv_c, 1 - tf.to_float(tf.abs(self.x_hv_c) > 0))
            self.x_hv_c = tf.reduce_prod(self.x_hv_c, reduction_indices=2)
            self.x_hv = 2*self.arc_tanh_like(self.x_hv_c, order=1005)

            # marginalization & loss calculating
            self.W_output_var = tf.multiply(self.W_output_var, self.W_output)
            self.out_i = tf.add(self.x, tf.matmul(self.x_hv, self.W_output))
            self.arg_loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_i, labels=1-self.y)

        self.y_output = self.out_i
        self.loss = tf.reduce_mean(self.arg_loss)
        self.train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(self.loss)

    def mlp_vn(self, x, weights):
        """Short summary.

        Parameters
        ----------
        x : type
            Description of parameter `x`.
        weights : type
            Description of parameter `weights`.

        Returns
        -------
        type
            Description of returned object.

        """

        # First Layer
        layer_1 = tf.einsum('aij,ajb->aib', x, weights['h1'])
        layer_1 = tf.nn.tanh(layer_1)

        # Second layer
        out_layer = tf.einsum('aij,ajb->aib', layer_1, weights['out'])
        out_layer = tf.nn.tanh(out_layer)

        out_layer = tf.squeeze(out_layer, 2)

        return out_layer

    def f_hyper(self, x, f_weights):
        """Short summary.

        Parameters
        ----------
        x : type
            Description of parameter `x`.
        f_weights : type
            Description of parameter `f_weights`.

        Returns
        -------
        type
            Description of returned object.

        """

        layer_1 = tf.einsum('aj,jb->ab', x, f_weights['f_h1'])
        layer_1 = tf.nn.tanh(layer_1)

        # Hidden layer with tanh activation
        layer_2 = tf.einsum('aj,jb->ab', layer_1, f_weights['f_h2'])
        layer_2 = tf.nn.tanh(layer_2)

        # Hidden layer with tanh activation
        layer_3 = tf.einsum('aj,jb->ab', layer_2, f_weights['f_h3'])
        layer_3 = tf.nn.tanh(layer_3)

        # Hidden layer with tanh activation
        layer_4 = tf.einsum('aj,jb->ab', layer_3, f_weights['f_h4'])
        layer_4 = tf.nn.tanh(layer_4)

        # Output layer with linear activation
        out_1 = tf.einsum('aj,jb->ab', layer_4, f_weights['f_head1'])
        out_2 = tf.einsum('aj,jb->ab', layer_4, f_weights['f_head2'])

        return out_1, out_2


    def arc_tanh_like(self, x, order):
        """Short summary.

        Parameters
        ----------
        x : type
            Description of parameter `x`.
        order : type
            Description of parameter `order`.

        Returns
        -------
        type
            Description of returned object.

        """

        out = x
        for i in range(3, order+1):
            if (i-1) % 2 == 0:
                out += (1.0/i)*tf.pow(x, i*tf.ones_like(x))

        return out

    def generate_graph_matrix(self, args):
        """Short summary.

        Parameters
        ----------
        args : type
            Description of parameter `args`.

        Returns
        -------
        type
            Description of returned object.

        """

        self.W_input = np.zeros((args.code_n, args.n_odd), dtype=args.var_type)
        self.W_odd2even = np.zeros((args.n_odd, args.n_even), dtype=args.var_type)
        self.W_odd2even_graphnn = np.zeros((args.n_odd, args.n_even, args.n_odd), dtype=args.var_type)
        self.W_skipconn2even = np.zeros((args.code_n,args.n_even), dtype=args.var_type)
        self.W_even2odd = np.zeros((args.n_even, args.n_odd), dtype=args.var_type)
        self.W_output = np.zeros((args.n_odd, args.code_n), dtype=args.var_type)

        # init W_input
        k = 0
        for i in range(0,self.data.code_parityCheckMatrix.shape[0],1):
            for j in range(0,self.data.code_parityCheckMatrix.shape[1],1):
                if(self.data.code_parityCheckMatrix[i,j] == 1):
                    vec = self.data.code_parityCheckMatrix[i,:].copy()
                    vec[j] = 0
                    self.W_input[:,k] = vec
                    k += 1

        # init W_odd2even & W_skipconn2even
        k = 0
        vec_tmp = np.zeros((args.n_odd),dtype=args.var_type)
        for j in range(0,self.data.code_parityCheckMatrix.shape[1],1):
            for i in range(0,self.data.code_parityCheckMatrix.shape[0],1):
                if(self.data.code_parityCheckMatrix[i,j] == 1):

                    num_of_conn = np.sum(self.data.code_parityCheckMatrix[:,j])        # get the number of connection of the variable node
                    idx = np.argwhere(self.data.code_parityCheckMatrix[:,j] ==1)       # get the indexes
                    for l in range(0, num_of_conn, 1):                                 # adding num_of_conn columns to W
                        vec_tmp = np.zeros((args.n_odd),dtype=args.var_type)
                        for r in range(0, self.data.code_parityCheckMatrix.shape[0], 1):         # adding one to the right place
                            if(self.data.code_parityCheckMatrix[r,j] == 1 and idx[l][0] != r):
                                idx_vec = np.cumsum(self.data.code_parityCheckMatrix[r,0:j+1])[-1] - 1
                                vec_tmp[int(idx_vec + np.sum(self.data.code_parityCheckMatrix[:r,:]))] = 1.0
                        self.W_odd2even[:,k] = vec_tmp.transpose()
                        k += 1
                    break

        # init W_odd2even_graphnn
        for j in range(0,self.W_odd2even.shape[1],1):
            for i in range(0,self.W_odd2even.shape[0],1):
                    self.W_odd2even_graphnn[j, i, i] = self.W_odd2even[i, j]

        # init W_even2odd, W_skipconn2even & W_output
        k, m = 0, 0
        for j in range(0,self.data.code_parityCheckMatrix.shape[1],1):
            for i in range(0,self.data.code_parityCheckMatrix.shape[0],1):
                if(self.data.code_parityCheckMatrix[i,j] == 1):

                    # W_even2odd
                    idx_row = np.cumsum(self.data.code_parityCheckMatrix[i,0:j+1])[-1] - 1
                    till_d_c = np.sum(self.data.code_parityCheckMatrix[:i,:])
                    this_d_c = np.sum(self.data.code_parityCheckMatrix[:(i+1),:])
                    self.W_even2odd[k,int(till_d_c):int(this_d_c)] = 1.0
                    self.W_even2odd[k,int(till_d_c+idx_row)] = 0.0

                    # W_skipconn2even
                    self.W_skipconn2even[j,k] = 1.0

                    # W_output
                    idx_row = np.cumsum(self.data.code_parityCheckMatrix[i,0:j+1])[-1] - 1
                    till_d_c = np.sum(self.data.code_parityCheckMatrix[:i,:])
                    self.W_output[int(till_d_c+idx_row), m] = 1.0

                    k += 1
            m += 1
