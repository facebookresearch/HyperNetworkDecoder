# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def calc_ber_fer(snr_db, Y_v_pred, Y_v, batches_for_val_per_snr):
    """Short summary.

    Parameters
    ----------
    snr_db : type
        Description of parameter `snr_db`.
    Y_v_pred : type
        Description of parameter `Y_v_pred`.
    Y_v : type
        Description of parameter `Y_v`.
    batches_for_val_per_snr : type
        Description of parameter `batches_for_val_per_snr`.

    Returns
    -------
    type
        Description of returned object.

    """

    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    last_ind = 0

    for i in range(0,snr_db.shape[0]):

        numOfWordSim = int(batches_for_val_per_snr[i]*1.0)
        Y_v_pred_i = Y_v_pred[last_ind:(last_ind + numOfWordSim),:]
        Y_v_i = Y_v[last_ind:(last_ind + numOfWordSim),:]
        ber_test[i] = np.abs(((Y_v_pred_i<0.5)-Y_v_i)).sum()/(Y_v_i.shape[0]*Y_v_i.shape[1])
        fer_test[i] = (np.abs(np.abs(((Y_v_pred_i<0.5)-Y_v_i))).sum(axis=1)>0).sum()*1.0/Y_v_i.shape[0]
        last_ind = last_ind + numOfWordSim

    return ber_test, fer_test
