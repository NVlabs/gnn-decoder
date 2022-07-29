# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


##### Utility functions for graph neural networks #####

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
import pickle
from sionna.utils.metrics import compute_ber
from time import time
import warnings # ignore some internal TensorFlow warnings

# for e2e model
from sionna.utils import BinarySource, ebnodb2no
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder


class MLP(Layer):
    """Simple MLP layer.

    Parameters
    ----------
    units : List of int
        Each element of the list describes the number of units of the
        corresponding layer.

    activations : List of activations
        Each element of the list contains the activation to be used
        by the corresponding layer.

    use_bias : List of booleans
        Each element of the list indicates if the corresponding layer
        should use a bias or not.
    """
    def __init__(self, units, activations, use_bias):
        super().__init__()
        self._num_units = units
        self._activations = activations
        self._use_bias = use_bias

    def build(self, input_shape):
        self._layers = []
        for i, units in enumerate(self._num_units):
            self._layers.append(Dense(units,
                                      self._activations[i],
                                      use_bias=self._use_bias[i]))

    def call(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs

class GNN_BP(Layer):
    """GNN-based message passing decoder.

    Parameters
    ---------
    pcm : [num_nc, num_vn], numpy.array
        The parity-check matrix.

    num_embed_dims: int
        Number of dimensions of the vertex embeddings.

    num_msg_dims: int
        Number of dimensions of a message.

    num_hidden_units: int
        Number of hidden units of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_mlp_layers: int
        Number of layers of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_iter: int
        Number of iterations.

    reduce_op: str
        A string defining the vertex aggregation function.
        Currently, "mean", "max", "min" and "sum" is supported.

    activation: str
        A string defining the activation function of the hidden MLP layers to
        be used. Defaults to "relu".

    output_all_iter: Bool
        Indicates if the LLRs of all iterations should be returned as list
        or if only the LLRs of the last iteration should be returned.

    clip_llr_to: float or None
        If set, the absolute value of the input LLRs will be clipped to this value.

    use_attributes: Boolean
        Defaults to False. If True, trainable node and edge attributes will be
        applied per node/edge, respectively.

    node_attribute_dims: int
        Number of dimensions of each node attribute.

    msg_attribute_dims: int
        Number of dimensions of each message attribute.

    use_bias: Boolean
        Defaults to False. Indicates if the MLPs should use a bias or not.

    Input
    -----
    llr : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs of all bits.

    Output
    ------
    llr_hat: : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs at the decoder output.
        If `output_all_iter`==True, a list of such tensors will be returned.
    """
    def __init__(self,
                 pcm,
                 num_embed_dims,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 num_iter,
                 reduce_op="mean",
                 activation="tanh",
                 output_all_iter=False,
                 clip_llr_to=None,
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,
                 use_bias=False):

        super().__init__()

        self._pcm = pcm # Parity check matrix
        self._num_cn = pcm.shape[0] # Number of check nodes
        self._num_vn = pcm.shape[1] # Number of variables nodes
        self._num_edges = int(np.sum(pcm)) # Number of edges

        # Array of shape [num_edges, 2]
        # 1st col = CN id, 2nd col = VN id
        # The ith row of this array defines the ith edge.
        self._edges = np.stack(np.where(pcm), axis=1)

        # Create 2D ragged tensor of shape [num_cn,...]
        # cn_edges[i] contains the edge ids for CN i
        cn_edges = []
        for i in range(self._num_cn):
            cn_edges.append(np.where(self._edges[:,0]==i)[0])
        self._cn_edges = tf.ragged.constant(cn_edges)

        # Create 2D ragged tensor of shape [num_vn,...]
        # vn_edges[i] contains the edge ids for VN i
        vn_edges = []
        for i in range(self._num_vn):
            vn_edges.append(np.where(self._edges[:,1]==i)[0])
        self._vn_edges = tf.ragged.constant(vn_edges)

        # Number of dimensions for vertex embeddings
        self._num_embed_dims = num_embed_dims

        # Number of dimensions for messages
        self._num_msg_dims = num_msg_dims

        # Number of hidden units for MLPs computing messages and embeddings
        self._num_hidden_units = num_hidden_units

        # Number of layers for MLPs computing messages and embeddings
        self._num_mlp_layers = num_mlp_layers

        # Number of BP iterations, can be modified
        self._num_iter = num_iter

        # Reduce operation for message aggregation
        self._reduce_op = reduce_op

        # Activation function of the hidden MLP layers
        self._activation = activation

        # if True, the model returns intermediate llrs
        self._output_all_iter = output_all_iter

        # Defines the (internal) LLR clipping value
        self._clip_llr_to = clip_llr_to

        # Actives (trainable) attributes
        self._use_attributes = use_attributes

        # Node /Edge attribute dimensions
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims

        # Activate bias of MLP layers
        self._use_bias = use_bias

        # Internal state for initialization
        self._is_built=False

    @property
    def num_iter(self):
        return self._num_iter

    @num_iter.setter # no retracing of graph (=no effect in graph mode)
    def num_iter(self, value):
        self._num_iter = value

    def build(self, input_shape):
        if not self._is_built: # only build once
            self._is_built=True
            # NN to transform input LLR to VN embedding
            self._llr_embed = Dense(self._num_embed_dims,
                                    use_bias=self._use_bias)

            # NN to transform VN embedding to output LLR
            self._llr_inv_embed = Dense(1,
                                        use_bias=self._use_bias)

            # CN embedding update function
            self.update_h_cn = UpdateEmbeddings(self._num_msg_dims,
                                                self._num_hidden_units,
                                                self._num_mlp_layers,
                                                # Flip columns: "from VN to CN"
                                                np.flip(self._edges, 1),
                                                self._cn_edges,
                                                self._reduce_op,
                                                self._activation,
                                                self._use_attributes,
                                                self._node_attribute_dims,
                                                self._msg_attribute_dims,
                                                self._use_bias)

            # VN embedding update function
            self.update_h_vn = UpdateEmbeddings(self._num_msg_dims,
                                                self._num_hidden_units,
                                                self._num_mlp_layers,
                                                self._edges, # "from CN to VN"
                                                self._vn_edges,
                                                self._reduce_op,
                                                self._activation,
                                                self._use_attributes,
                                                self._node_attribute_dims,
                                                self._msg_attribute_dims,
                                                self._use_bias)

    def llr_to_embed(self, llr):
        """Transform LLRs to VN embeddings."""
        return self._llr_embed(tf.expand_dims(llr, -1))

    def embed_to_llr(self, h_vn):
        """Transform VN embeddings to LLRs."""
        return tf.squeeze(self._llr_inv_embed(h_vn), axis=-1)

    def call(self, llr):
        """Run the decoder."""
        batch_size = tf.shape(llr)[0]

        # Initialize vertex embeddings
        if self._clip_llr_to is not None:
            llr = tf.clip_by_value(llr, -self._clip_llr_to, self._clip_llr_to)

        h_vn = self.llr_to_embed(llr)
        h_cn = tf.zeros([batch_size, self._num_cn, self._num_embed_dims])

        # BP iterations
        if self._output_all_iter:
            llr_hat = []
        for i in range(self._num_iter):

            # Update CN embeddings
            h_cn = self.update_h_cn(h_vn, h_cn)

            # Update VNs
            h_vn = self.update_h_vn(h_cn, h_vn)

            if self._output_all_iter:
                llr_hat.append(self.embed_to_llr(h_vn))

        if not self._output_all_iter:
            llr_hat = self.embed_to_llr(h_vn)

        return llr_hat

class UpdateEmbeddings(Layer):
    """Update vertex embeddings of the GNN BP decoder.

    This layer computes first the messages that are sent across the edges
    of the graph, then sums the incoming messages at each vertex, finally and
    updates their embeddings.

    Parameters
    ----------
    num_msg_dims: int
        Number of dimensions of a message.

    num_hidden_units: int
        Number of hidden units of MLPs used to compute
        messages and to update the vertex embeddings.

    num_mlp_layers: int
        Number of layers of the MLPs used to compute
        messages and to update the vertex embeddings.

    from_to_ind: [num_egdes, 2], np.array
        Two dimensional array containing in each row the indices of the
        originating and receiving vertex for an edge.

    gather_ind: [`num_vn` or `num_cn`, None], tf.ragged.constant
        Ragged tensor that contains for each receiving vertex the list of
        edge indices from which to aggregate the incoming messages. As each
        vertex can have a different degree, a ragged tensor is used.

    reduce_op: str
        A string defining the vertex aggregation function.
        Currently, "mean", "max", "min" and "sum" is supported.

    activation: str
        A string defining the activation function of the hidden MLP layers to
        be used. Defaults to "relu".

    use_attributes: Boolean
        Defaults to False. If True, trainable node and edge attributes will be
        applied per node/edge, respectively.

    node_attribute_dims: int
        Number of dimensions of each node attribute.

    msg_attribute_dims: int
        Number of dimensions of each message attribute.

    use_bias: Boolean
        Defaults to False. Indicates if the MLP should use a bias or not.

    Input
    -----
    h_from : [batch_size, num_cn or num_vn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "transmitting" vertices.

    h_to : [batch_size, num_vn or num_cn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "receiving" vertices.

    Output
    ------
    h_to_new : Same shape and type as `h_to`
        Tensor containing the updated embeddings of the "receiving" vertices.
    """
    def __init__(self,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 from_to_ind,
                 gather_ind,
                 reduce_op="sum",
                 activation="relu",
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,
                 use_bias=False):

        super().__init__()
        self._num_msg_dims = num_msg_dims
        self._num_hidden_units = num_hidden_units
        self._num_mlp_layers = num_mlp_layers
        self._from_ind = from_to_ind[:,0]
        self._to_ind = from_to_ind[:,1]
        self._gather_ind = gather_ind
        self._reduce_op = reduce_op
        self._activation = activation
        self._use_attributes = use_attributes
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims
        self._use_bias = use_bias

        # add node attributes
        if self._use_attributes:
            num_nodes = self._gather_ind.shape[0]
            num_edges = self._from_ind.shape[0]
            # node attributes
            self._g_node = tf.Variable(tf.zeros((num_nodes,
                                       self._node_attribute_dims),tf.float32),
                                       trainable=True)
            # edge attributes
            self._g_msg = tf.Variable(tf.zeros((num_edges,
                                      self._msg_attribute_dims), tf.float32),
                                      trainable=True)

    def build(self, input_shape):

        num_embed_dims = input_shape[-1]

        # MLP to compute messages
        units = [self._num_hidden_units]*(self._num_mlp_layers-1) + [self._num_msg_dims]
        activations = [self._activation]*(self._num_mlp_layers-1) + [None]
        use_bias = [self._use_bias]*self._num_mlp_layers
        self._msg_mlp = MLP(units, activations, use_bias)

        # MLP to update embeddings from accumulated messages
        units[-1] = num_embed_dims
        self._embed_mlp = MLP(units, activations, use_bias)

    def call(self, h_from, h_to):

        # Concatenate embeddings of the transmitting (from) and receiving (to) vertex for each edge
        features = tf.concat([tf.gather(h_from, self._from_ind, axis=1),
                              tf.gather(h_to, self._to_ind, axis=1)],
                             axis=-1)

        # Add message attribute
        if self._use_attributes:
            attr = tf.tile(tf.expand_dims(self._g_msg, axis=0),
                          [tf.shape(features)[0], 1, 1])
            features = tf.concat((features, attr), axis=-1)


        # Compute messsages for all edges
        messages = self._msg_mlp(features)

        # Reduce messages at each receiving (to) vertex
        # note: bring batch dim to last dim for improved performance
        # with ragged tensors
        messages = tf.transpose(messages, (1,2,0))
        m_ragged = tf.gather(messages, self._gather_ind, axis=0)
        if self._reduce_op=="sum":
            m = tf.reduce_sum(m_ragged, axis=1)
        elif self._reduce_op=="mean":
            m = tf.reduce_mean(m_ragged, axis=1)
        elif self._reduce_op=="max":
            m = tf.reduce_max(m_ragged, axis=1)
        elif self._reduce_op=="min":
            m = tf.reduce_min(m_ragged, axis=1)
        else:
            raise ValueError("unknown reduce operation")
        m = tf.transpose(m, (2,0,1)) # batch-dim back to first dim

        # Add node attribute
        if self._use_attributes:
            # tile to bs dim
            attr = tf.tile(tf.expand_dims(self._g_node, axis=0),
                          [tf.shape(m)[0], 1, 1])
            m = tf.concat((m, attr), axis=-1)

        # Compute new embeddings
        h_to_new = self._embed_mlp(tf.concat([m, h_to], axis=-1))

        return h_to_new

######### Utility functions #########

def save_weights(system, model_path):
    """Save model weights.

    This function saves the weights of a Keras model ``system`` to the
    path as provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            A model containing the weights to be stored.

        model_path: str
            Defining the path where the weights are stored.

    """
    weights = system.get_weights()
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(system, model_path):
    """Load model weights.

    This function loads the weights of a Keras model ``system`` from a file
    provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            The target model into which the weights are loaded.

        model_path: str
            Defining the path where the weights are stored.

    """
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    system.set_weights(weights)

def train_gnn(model, params):
    """Training function for the GNN decoder model.

    This function also generates log files and save the weights every N
    iterations.

    Parameters
    ----------
        system: Keras model
            The system model that should be trained.

        params: dict
            Defining all required training/model parameters.
    """
    # we ignore TF warnings related to sparse indexed slices
    # These warnings are related to raggedTensors and cannot be easily fixed.
    warnings.filterwarnings('ignore', '.*Converting sparse IndexedSlices.*', )

    # We use the BCE loss
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # This function logs the training progress into an external file
    log_dir = params["save_dir"] \
              + params["run_name"] \
              + "_log.txt"
    # generate log/result folder
    try:
        os.makedirs(params["save_dir"])
    except FileExistsError:
        pass

    with open(log_dir, 'a') as f:
        f.write("\n----Starting Training----\n")
        f.write(str(params) + "\n")

    # SGD update iteration
    @tf.function()
    def train_step(batch_size):
        # train for random SNRs within a pre-defined interval
        ebno_db = tf.random.uniform([batch_size, 1],
                                    minval=params["ebno_db_min"],
                                    maxval=params["ebno_db_max"])
        with tf.GradientTape() as tape:
            c, llr_hat = model(batch_size, ebno_db)
            loss_value = 0
            # we use a multi-loss averaged over all iterations
            for _, l in enumerate(llr_hat):
                loss_value += loss(c, l)

        # and apply the SGD updates
        weights = model.trainable_weights
        grads = tape.gradient(loss_value, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return c, llr_hat

    # init the optimizer; we use Adam throughout this work
    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=params["learning_rate"][0])

    # run the training iterations
    iter_total = 0
    time_start = time() # measure time per 1000 iters

    # for each list-element in the training parameters we run the SGD-updates
    for idx,_ in enumerate(params["batch_size"]):
        batch_size = tf.constant(params["batch_size"][idx], tf.int32)
        lr = params["learning_rate"][idx]
        train_iter = params["train_iter"][idx]

        # set new learning rate
        optimizer.lr.assign(lr)

        # and log the training
        log_str = f"New training parameters - bs: {batch_size}, "\
                  f"lr: {lr}, iters: {train_iter}"
        print(log_str)
        with open(log_dir, 'a') as f:
            f.write(log_str + "\n")

        # run pre-defined number of training iterations
        for it in range(train_iter):
            iter_total += 1 # total number of iters to log training progress
            train_step(batch_size) # and train

            # evaluate intermediate training results
            if iter_total%params["eval_train_steps"]==0:
                ebno_db = tf.random.uniform([params["batch_size_eval"], 1],
                                             minval=params["ebno_db_eval"],
                                             maxval=params["ebno_db_eval"])
                c, llr_hat = model(params["batch_size_eval"], ebno_db)
                loss_value = 0
                for l in llr_hat:
                    loss_value += loss(c, l)
                # for BER calculations only consider last decoder iterations
                # i.e., [-1] axis)
                c_hat = tf.cast(tf.greater(llr_hat[-1], 0), tf.float32)
                ber = compute_ber(c, c_hat).numpy()

                # measure required time since last evaluation
                duration = time() - time_start # in s
                time_start = time() # reset counter
                log_str = f"Iteration {iter_total}, " \
                          f"loss = {loss_value.numpy():.3f}, " \
                          f"ber = {ber:.5f}, " \
                          f"duration: {duration:.2f}s"
                print(log_str)
                # and write intermediate results in file
                with open(log_dir, 'a') as f:
                    f.write(log_str +"\n")

            # save weights of model every X iters
            # keep in mind that this may require a log of memory
            if iter_total%params["save_weights_iter"]==0:
                model_path = params["save_dir"] \
                              + params["run_name"] \
                              + "_" + str(iter_total) + ".npy"
                save_weights(model, model_path)

    # and save the final weights
    model_path = params["save_dir"] + params["run_name"] + "_final.npy"
    save_weights(model, model_path)

class E2EModel(tf.keras.Model):
    """End-to-end model for (GNN-)decoder evaluation.

    Parameters
    ----------
    encoder: Layer or None
        Encoder layer, no encoding applied if None.

    decoder: Layer or None
        Decoder layer, no decoding applied if None.

    k: int
        Number of information bits per codeword.

    n: int
        Codeword lengths.

    return_infobits: Boolean
        Defaults to False. If True, only the ``k`` information bits are
        returned. Must be supported be the decoder as well.

    es_no: Boolean
        Defaults to False. If True, the SNR is not rate-adjusted (i.e., Es/N0).

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (c, llr):
            Tuple:

        c: tf.float32
            A tensor of shape `[batch_size, n] of 0s and 1s containing the
            transmitted codeword bits.

        llr: tf.float32
            A tensor of shape `[batch_size, n] of llrs containing estimated on
            the codeword bits.
    """

    def __init__(self, encoder, decoder, k, n, return_infobits=False, es_no=False):
        super().__init__()

        self._n = n
        self._k = k

        self._binary_source = BinarySource()
        self._num_bits_per_symbol = 2
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._channel = AWGN()
        self._decoder = decoder
        self._encoder = encoder
        self._return_infobits = return_infobits
        self._es_no = es_no

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):

        # no rate-adjustment for uncoded transmission or es_no scenario
        if self._decoder is not None and self._es_no==False:
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._k/self._n)
        else: #for uncoded transmissions the rate is 1
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, 1)

        b = self._binary_source([batch_size, self._k])
        if self._encoder is not None:
            c = self._encoder(b)
        else:
            c = b

        # check that rate calculations are correct
        assert self._n==c.shape[-1], "Invalid value of n."

        # zero padding to support odd codeword lengths
        if self._n%2==1:
            c_pad = tf.concat([c, tf.zeros([batch_size, 1])], axis=1)
        else: # no padding
            c_pad = c
        x = self._mapper(c_pad)

        y = self._channel([x, no])
        llr = self._demapper([y, no])

        # remove zero padded bit at the end
        if self._n%2==1:
            llr = llr[:,:-1]

        # and run the decoder
        if self._decoder is not None:
            llr = self._decoder(llr)

        if self._return_infobits:
            return b, llr
        else:
            return c, llr

def export_pgf(ber_plot, col_names):
    """Export results as table for for pgfplots compatible imports.

    Parameters
    ----------
    ber_plot: PlotBER
        An object of PlotBER containing the BER simulations to be exported

    col_names: list of str
        Column names of the exported BER curves
    """
    s = "snr, \t"
    for idx, var_name in enumerate(col_names):
        s += var_name + ", \t"
    s += "\n"

    for idx_snr,snr in enumerate(ber_plot._snrs[0]):
        s += f"{snr:.3f},\t"
        for idx_dec, _ in enumerate(col_names):
            s += f"{ber_plot._bers[idx_dec][idx_snr].numpy():.6E},\t"
        s += "\n"
    print(s)

def generate_pruned_pcm_5g(decoder, n, verbose=True):
    """Utility function to get the pruned parity-check matrix of the 5G code.

    Identifies the pruned and shortened positions.
    Hereby, '0' indicates an pruned codeword position
    '1' indicates an codeword position
    '2' indicates a shortened position.

    Parameters
    ---------
    decoder: LDPC5GDecoder
        An instance of the decoder object.

    n: int
        The codeword lengths including rate-matching.

    verbose: Boolean
        Defaults to True. If True, status information during pruning is
        provided.
    """

    enc = decoder._encoder

    # transmitted positions
    pos_tx = np.ones(n)

    # undo puncturing of the first 2*z information bits
    pos_punc = np.concatenate([np.zeros([2*enc.z]),pos_tx], axis=0)

    # puncturing of the last positions
    # total length must be n_ldpc, while pos_tx has length n
    # first 2*z positions are already added
    # -> add n_ldpc - n - 2Z punctured positions
    k_short = enc.k_ldpc - enc.k # number of shortend bits
    num_punc_bits = ((enc.n_ldpc - k_short) - enc.n - 2*enc.z)
    pos_punc2 = np.concatenate(
               [pos_punc, np.zeros([num_punc_bits - decoder._nb_pruned_nodes])])

    # shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
    # the first k positions are the systematic bits
    pos_info = pos_punc2[0:enc.k]

    # parity part
    num_par_bits = (enc.n_ldpc-k_short-enc.k-decoder._nb_pruned_nodes)
    pos_parity = pos_punc2[enc.k:enc.k+num_par_bits]
    pos_short = 2 * np.ones([k_short]) # "2" indicates shortened position

    # and concatenate final pattern
    rm_pattern = np.concatenate([pos_info, pos_short, pos_parity], axis=0)

    # and prune matrix (remove shortend positions from pcm)
    pcm_pruned = np.copy(decoder.pcm.todense())
    idx_short = np.where(rm_pattern==2)
    idx_pruned = np.setdiff1d(np.arange(pcm_pruned.shape[1]), idx_short)
    pcm_pruned = pcm_pruned[:,idx_pruned]
    num_shortened = np.size(idx_short)

    # print information if enabled
    if verbose:
        print("using bg: ", enc._bg)
        print("# information bits:", enc.k)
        print("CW length after rate-matching:", n)
        print("CW length without rm (incl. first 2*Z info bits):",
                                    pcm_pruned.shape[1])
        print("# punctured bits:", num_punc_bits)
        print("# pruned nodes:", decoder._nb_pruned_nodes)
        print("# parity bits", num_par_bits)
        print("# shortened bits", num_shortened)
        print("pruned pcm dimension:", pcm_pruned.shape)
    return pcm_pruned, rm_pattern[idx_pruned]

class LDPC5GGNN(GNN_BP):
    """GNN-based Decoder for 5G LDPC codes incl. internal rate-matching.

    This layer inherits from the GNN_BP decoder and extends its functionality
    by the LDPC rate-matching.

    Parameters
    ---------
    encoder : LDPC5GEncoder
        Instance of LDPC5GEncoder used for encoding.

    num_embed_dims: int
        Number of dimensions of the vertex embeddings.

    num_msg_dims: int
        Number of dimensions of a message.

    num_hidden_units: int
        Number of hidden units of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_mlp_layers: int
        Number of layers of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_iter: int
        Number of iterations.

    reduce_op: str
        A string defining the vertex aggregation function.
        Currently, "mean", "max", "min" and "sum" is supported.

    activation: str
        A string defining the activation function of the hidden MLP layers to
        be used. Defaults to "relu".

    output_all_iter: Bool
        Indicates if the LLRs of all iterations should be returned as list
        or if only the LLRs of the last iteration should be returned.

    clip_llr_to: float or None
        If set, the absolute value of the input LLRs will be clipped to this value.

    use_attributes: Boolean
        Defaults to False. If True, trainable node and edge attributes will be
        applied per node/edge, respectively.

    node_attribute_dims: int
        Number of dimensions of each node attribute.

    msg_attribute_dims: int
        Number of dimensions of each message attribute.

    return_infobits: Boolean
        Defaults to False. Indicates if only the `k` information bits are
        returned.

    use_bias: Boolean
        Defaults to True. Indicates if the MLPs should use a bias or not.

    Input
    -----
    llr : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs of all bits.

    Output
    ------
    llr_hat: : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs at the decoder output.
        If `output_all_iter`==True, a list of such tensors will be returned.
    """

    def __init__(self,
                 encoder,
                 num_embed_dims,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 num_iter,
                 reduce_op="mean",
                 activation="tanh",
                 output_all_iter=False,
                 clip_llr_to=None,
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,
                 return_infobits=False,
                 use_bias=True,
                 **kwargs):

        self._encoder = encoder
        self._return_infobits = return_infobits
        self._llr_max = 20 # internal max value for LLR initialization

        # instantiate internal decoder object to access pruned pcm
        # Remark: this object is NOT used for decoding!
        decoder = LDPC5GDecoder(encoder, prune_pcm=True)

        # access pcm and code properties
        self._n_pruned = decoder._n_pruned
        self._num_pruned_nodes = decoder._nb_pruned_nodes
        # prune and remove shortened positions
        self._pcm, self._rm_pattern = generate_pruned_pcm_5g(decoder,
                                                             encoder.n,
                                                             verbose=False)
        # precompute pruned positions
        gather_ind = encoder.n * np.ones(np.size(self._rm_pattern))
        gather_ind_inv = np.zeros(np.size(np.where(self._rm_pattern==1)))
        for idx, pos in enumerate(np.where(self._rm_pattern==1)[0]):
            gather_ind[pos] = idx
            gather_ind_inv[idx] = pos

        self._rm_ind = tf.constant(gather_ind, tf.int32)
        self._rm_inv_ind = tf.constant(gather_ind_inv, tf.int32)


        # init GNN decoder
        super().__init__(self._pcm,
                         num_embed_dims,
                         num_msg_dims,
                         num_hidden_units,
                         num_mlp_layers,
                         num_iter,
                         reduce_op,
                         activation,
                         output_all_iter,
                         clip_llr_to,
                         use_attributes,
                         node_attribute_dims,
                         msg_attribute_dims,
                         use_bias,
                         **kwargs)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def llr_max(self):
        """Max LLR value used for rate-matching."""
        return self._llr_max

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build model."""
        super().build(input_shape)

    def call(self, inputs):
        """Iterative MPNN decoding function."""

        llr_ch = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]

        # add punctured positions
        # append one zero pos
        llr_in = tf.concat([llr_ch, tf.zeros([batch_size, 1], tf.float32)],
                           axis=1)
        llr_rm = tf.gather(llr_in, self._rm_ind, axis=1)

        # and execute the decoder
        x_hat_dec = super().call(llr_rm)

        # we need to de-ratematch for all iterations individually (for training)
        if not self._output_all_iter:
            x_hat_list = [x_hat_dec]
        else:
            x_hat_list = x_hat_dec

        u_out = []
        x_out =[]

        for idx,x_hat in enumerate(x_hat_list):
            if self._return_infobits: # return only info bits
                # reconstruct u_hat # code is systematic
                u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
                u_out.append(u_hat)

            else: # return all codeword bits
                x_short = tf.gather(x_hat, self._rm_inv_ind, axis=1)
                x_out.append(x_short)

        # return no list
        if not self._output_all_iter:
            if self._return_infobits:
                return u_out[-1]
            else:
                return x_out[-1]

        # return list of all iterations
        if self._return_infobits:
            return u_out
        else:
            return x_out
