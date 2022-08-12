# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Utility functions for Weighted BP decoding

import tensorflow as tf
from tensorflow.keras import Model
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import ebnodb2no
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from gnn import * # load ecc utility functions

class WeightedBP(tf.keras.Model):
    """System model for BER simulations of Weighted BP decoding.

    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.

    Parameters
    ----------
        pcm: ndarray
            The parity-check matrix of the code under investigation.

        num_iter: int
            Number of BP decoding iterations.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (c, c_hat, loss):
            Tuple:

        c: tf.float32
            A tensor of shape `[batch_size, n]` of 0s and 1s containing the
            transmitted codeword bits.

        c_hat: tf.float32
            A tensor of shape `[batch_size, n]` of 0s and 1s containing the
            estimated codeword bits.

        loss: tf.float32
            Binary cross-entropy loss between `c` and `c_hat`.
    """
    def __init__(self, pcm, num_iter=5):
        super().__init__()
        print("Note that WBP requires Sionna > v0.11.")
        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1, # iterations are done via
                                     # outer loop (to access intermediate
                                     # results for multi-loss)
                                     stateful=True,
                                     hard_out=False, # we need to access
                                     # soft-information
                                     cn_type="boxplus",
                                     trainable=True) # the decoder must be
                                     # trainable, otherwise no weights are
                                     # generated

        # used to generate llrs during training (see example notebook on
        # all-zero codeword trick)
        self._llr_source = GaussianPriorSource()
        self._num_iter = num_iter

        self._n = pcm.shape[1]
        self._coderate = 1 - pcm.shape[0]/pcm.shape[1]
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        # batch_size = tf.constant(batch_size, dtype=tf.int32)
        # ebno_db = tf.constant(ebno_db, dtype=tf.float32)
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)

        # all-zero CW to calculate loss / BER
        c = tf.zeros([batch_size, self._n])

        # Gaussian LLR source
        llr = self._llr_source([[batch_size, self._n], noise_var])

        # implement multi-loss as proposed by Nachmani et al.
        loss = 0
        msg_vn = None
        for _ in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn)) # perform one decoding iteration; decoder
            # returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration

        loss /= self._num_iter # scale loss by number of iterations

        return c, c_hat, loss

    def train_wbp(self, train_param):

        # check training params for consistency
        assert len(train_param["wbp_batch_size"])==\
               len(train_param["wbp_train_iter"]),\
               "wbp_batch_size must have same lengths as wbp_train_iter."
        assert len(train_param["wbp_batch_size"])==\
               len(train_param["wbp_learning_rate"]),\
                "wbp_batch_size must have same lengths as wbp_learning_rate."
        assert len(train_param["wbp_batch_size"])==\
               len(train_param["wbp_ebno_train"]),\
               "wbp_batch_size must have same lengths as wbp_ebno_train."

        # bmi is used as metric to evaluate the intermediate results
        bmi = BitwiseMutualInformation()

        # init optimizer
        optimizer = tf.keras.optimizers.Adam(
                                        train_param["wbp_learning_rate"][0])

        # and run the training
        for idx, batch_size in enumerate(train_param["wbp_batch_size"]):
            # set new learning rate
            optimizer.learning_rate = train_param["wbp_learning_rate"][idx]

            for it in range(train_param["wbp_train_iter"][idx]):
                with tf.GradientTape() as tape:
                    b, llr, loss = self(batch_size,
                                        train_param["wbp_ebno_train"][idx])

                grads = tape.gradient(loss, self.trainable_variables)
                grads = tf.clip_by_value(grads,
                                         -train_param["wbp_clip_value_grad"],
                                         train_param["wbp_clip_value_grad"],
                                         name=None)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # calculate and print intermediate metrics
                # only for information, this has no impact on the training
                if it%50==0: # evaluate every 10 iterations
                    b, llr, loss = self(train_param["wbp_batch_size_val"],
                                        train_param["wbp_ebno_val"])
                    b_hat = hard_decisions(llr) # hard decided LLRs first
                    ber = compute_ber(b, b_hat)
                    # and print results
                    mi = bmi(b, llr).numpy() # calculate bit-wise mutual information
                    l = loss.numpy() # copy loss to numpy for printing
                    print(f"Iter: {it} loss: {l:3f} ber: {ber:.6f} "\
                          f"bmi: {mi:.3f}".format())
                    bmi.reset_states() # reset the BMI metric


######### Utility functions #######
def evaluate_wbp(params, pcm, encoder, ebno_dbs, ber_plot):
    """Train and evaluate a weighted BP end-to-end model.

    Parameters
    ----------
    params: dict
        Containing the system/training hyperparameters.

    pcm : [num_ch, num_vn], numpy.array
        The parity-check matrix.

    encoder: Layer or None
        Encoder layer, no encoding applied if None.

    ebno_dbs: ndarray
        Containing the SNR points to be evaluated.

    ber_plot: PlotBER
        Instance of PlotBER for BER simulations.
    """
    model_wbp = WeightedBP(pcm=pcm, num_iter=10) # only used for training

    # and run the training loop
    model_wbp.train_wbp(params)

    # generate new decoder object (with 20 iterations) for evaluation
    bp_decoder_wbp = LDPCBPDecoder(pcm,
                                   num_iter=20,
                                   hard_out=False,
                                   trainable=True)

    # copy weights from trained decoder
    bp_decoder_wbp.set_weights(model_wbp.decoder.get_weights())
    e2e_wbp = E2EModel(encoder, bp_decoder_wbp, params["k"], params["n"])

    ber_plot.simulate(e2e_wbp,
                     ebno_dbs=ebno_dbs,
                     batch_size=params["mc_batch_size"],
                     num_target_block_errors=params["num_target_block_errors"],
                     legend=f"Weighted BP {bp_decoder_wbp._num_iter.numpy()} iter.",
                     soft_estimates=True,
                     max_mc_iter=params["mc_iters"],
                     forward_keyboard_interrupt=False,
                     show_fig=False);
