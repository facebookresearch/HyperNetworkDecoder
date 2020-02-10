# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
import numpy as np
from utils import *



class Solver():

    def __init__(self, args, data, model):

        self.args = args
        self.data = data
        self.model = model

    def train(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """

        # Init Session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        # Load model
        if self.args.weights_path:
            saver.restore(sess, self.args.weights_path)

        for i in range(self.args.num_of_batch):

            # Generate data
            training_data, training_labels = self.data.get_batch(self.args.train_on_zero_word)

            # Train
            y_train, train_loss, _ = sess.run(fetches=[self.model.y_output, self.model.loss, self.model.train_step], feed_dict={self.model.x: training_data, self.model.y: training_labels})

            # Evaluate
            if(i%self.args.batch_in_epoch == 0) and (i != 0):

                y_v = np.zeros([1,self.args.code_n], dtype=self.args.var_type)
                y_v_pred = np.zeros([1,self.args.code_n], dtype=self.args.var_type)
                loss_v = np.zeros([1, 1], dtype=self.args.var_type)
                for kk, k_sf in enumerate(self.data.scaling_factor):
                    for j in range(self.args.batches_for_val_per_snr_all[kk]):

                        x_v_j, y_v_j = self.data.get_batch(self.args.test_on_zero_word, self.args.batch_size, [k_sf])
                        y_v_pred_j, loss_v_j = sess.run(fetches = [self.model.y_output, self.model.loss], feed_dict={self.model.x:x_v_j, self.model.y:y_v_j})

                        y_v = np.vstack((y_v,y_v_j))
                        y_v_pred = np.vstack((y_v_pred,y_v_pred_j))
                        loss_v = np.vstack((loss_v, loss_v_j))

                y_v_pred = 1.0 / (1.0 + np.exp(-1.0 * y_v_pred))
                ber_val, fer_val = calc_ber_fer(self.args.snr_db, y_v_pred[1:,:], y_v[1:,:], self.args.batch_size*self.args.batches_for_val_per_snr_all)

                # Print results
                print('Results for epoch - ', int(i/self.args.batch_in_epoch))
                print('SNR range in [dB] - ', self.args.snr_db)
                print('Bit Error rate for validation - ', ber_val)
                print('Frame Error rate for validation - ', fer_val)

                # Save weights
                saver.save(sess, self.args.weights_path_format % i)
