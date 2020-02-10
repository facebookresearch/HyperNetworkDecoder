# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.getcwd()+'/scripts')
from data import DataSet
from model import HyperNetworkDecoder
from solver import Solver

import argparse
parser = argparse.ArgumentParser("Hyper-Graph-Network Decoders for Block Codes")


# General Configuration
parser.add_argument('--code_parityCheckMatrix_path', type=str, default='./data/BCH_63_51_H.npy',
                    help='Parity check matrix file')
parser.add_argument('--code_generatorMatrix_path', type=str, default='./data/BCH_63_51_G.npy',
                    help='Generator matrix file')
parser.add_argument('--code_n', type=int, default=63,
                    help='Block length - N')
parser.add_argument('--code_k', type=int, default=51,
                    help='Number of information bits - K')
parser.add_argument('--start_snr', default=1.0, type=float,
                    help='SNR start value, in dB.')
parser.add_argument('--stop_snr', default=8.0, type=float,
                    help='SNR stop value, in dB.')
parser.add_argument('--step', default=1.0, type=float,
                    help='SNR step, in dB.')
parser.add_argument('--word_seed', default=786000, type=int,
                    help='Seed for word generator')
parser.add_argument('--noise_seed', default=345000, type=int,
                    help='Seed for noise generator')
parser.add_argument('--weights_path', default='', type=str,
                    help='Path for checkpoint')
parser.add_argument('--num_hidden_layers', default=5, type=int,
                    help='Number of iteration of BP')
parser.add_argument('--numOfWordSim_train', default=15, type=int,
                    help='Number of word simulated')
parser.add_argument('--batches_for_val_per_snr', default=500, type=int,
                    help='Number batches per SNR')
parser.add_argument('--batch_in_epoch', default=500, type=int,
                    help='Number of batches in epoch')
parser.add_argument('--num_of_batch', default=100000000, type=int,
                    help='Total number of batches in training')
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--gpu_mem_fraction', default=0.99, type=float,
                    help='Percentage of memory GPU useage')
parser.add_argument('--train_on_zero_word', action='store_true',
                    help='Does train on zero codeword only?')
parser.add_argument('--test_on_zero_word', action='store_true',
                    help='Does test on zero codeword only?')
parser.add_argument('--n_hidden_1', default=16, type=int,
                    help='Number of neurons in the first layer of g')
parser.add_argument('--n_hidden_2', default=16, type=int,
                    help='Number of neurons in the second layer of g')
parser.add_argument('--sf_n_hidden_1', default=32, type=int,
                    help='Number of neurons in the first layer of f')
parser.add_argument('--sf_n_hidden_2', default=32, type=int,
                    help='Number of neurons in the second layer of f')
parser.add_argument('--sf_n_hidden_3', default=32, type=int,
                    help='Number of neurons in the third layer of f')
parser.add_argument('--weights_path_save', default=os.getcwd(), type=str,
                    help='Path to save checkpoint')
parser.add_argument('--var_type', default=np.float32, type=type,
                    help='Variable type')


def main(args):

    # Loading Dataset
    dataLoader = DataSet(args)

    # Build Model
    model = HyperNetworkDecoder(args, dataLoader)

    # Init GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_fraction)

    # Init Solver
    solver = Solver(args, dataLoader, model)

    # Training & Evaluating
    solver.train()


if __name__ == '__main__':

    args = parser.parse_args()

    ### General Parameters
    args.code_parityCheckMatrix = np.load(args.code_parityCheckMatrix_path)
    args.code_generatorMatrix = np.load(args.code_generatorMatrix_path)
    args.code_rate = 1.0*args.code_k/args.code_n
    args.snr_db = np.arange(args.start_snr, args.stop_snr+args.step, args.step, dtype=np.float32)
    args.batch_size = args.numOfWordSim_train * len(args.snr_db)
    args.batches_for_val_per_snr_all = np.ones((len(args.snr_db),)) * args.batches_for_val_per_snr
    args.batches_for_val_per_snr_all = args.batches_for_val_per_snr_all.astype(int)
    args.n_odd = int(np.sum(args.code_parityCheckMatrix))
    args.n_even = args.n_odd
    args.n_input = args.n_odd + 1 # plus 1 for skip connection
    args.sf_n_input = args.n_odd
    args.snr_lin = 10.0**(args.snr_db/10.0)
    args.weights_path_format = args.weights_path_save + '/weights/weights_epoch_%03d.ckpt'

    main(args)
