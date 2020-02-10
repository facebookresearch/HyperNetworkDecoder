# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

class DataSet():

    def __init__(self, args):

        self.scaling_factor = np.sqrt(1.0/(2.0*args.snr_lin*args.code_rate))
        self.wordRandom = np.random.RandomState(args.word_seed)
        self.random = np.random.RandomState(args.noise_seed)
        self.numOfWordSim = args.numOfWordSim_train
        self.code_n = args.code_n
        self.code_k = args.code_k
        self.code_parityCheckMatrix = args.code_parityCheckMatrix
        self.code_generatorMatrix = args.code_generatorMatrix


    def get_batch(self, is_zeros_word, numOfWordSim_new=None, scaling_factor_new=None):
        """Short summary.

        Parameters
        ----------
        is_zeros_word : type
            Description of parameter `is_zeros_word`.
        numOfWordSim_new : type
            Description of parameter `numOfWordSim_new`.
        scaling_factor_new : type
            Description of parameter `scaling_factor_new`.

        Returns
        -------
        type
            Description of returned object.

        """

        self.X = np.zeros([1, self.code_n], dtype=np.float32)
        self.Y = np.zeros([1, self.code_n], dtype=np.int32)

        if scaling_factor_new:
            scaling_factor = scaling_factor_new
        else:
            scaling_factor = self.scaling_factor

        if numOfWordSim_new:
            numOfWordSim = numOfWordSim_new
        else:
            numOfWordSim = self.numOfWordSim

        # Build set for epoch
        for sf_i in scaling_factor:
            if is_zeros_word:
                infoWord_i = 0*self.wordRandom.randint(0, 2, size=(numOfWordSim, self.code_k))
            else:
                infoWord_i = self.wordRandom.randint(0, 2, size=(numOfWordSim, self.code_k))

            Y_i = np.dot(infoWord_i, self.code_generatorMatrix) % 2
            X_p_i = self.random.normal(0.0, 1.0, Y_i.shape)*sf_i + (-1)**(Y_i)
            x_llr_i = 2*X_p_i/(sf_i**2)
            self.X = np.vstack((self.X, x_llr_i))
            self.Y = np.vstack((self.Y, Y_i))

        self.X = self.X[1:]
        self.Y = self.Y[1:]

        return self.X, self.Y
