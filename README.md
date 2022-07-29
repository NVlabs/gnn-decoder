<!-- SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited. -->

# Graph Neural Networks for Channel Decoding

Implementation of the graph neural network (GNN)-based decoder experiments from
[[A *PLACEHOLDER*]](*PLACEHOLDER*) using the
[Sionna link-level simulator](https://nvlabs.github.io/sionna/).

## Abstract

In this work, we propose a fully differentiable graph neural network (GNN)-based framework for channel decoding and showcase a competitive decoding performance for various coding schemes, such as low-density parity-check (LDPC) and BCH codes with significantly less decoding iterations. The idea is to let an NN learn a generalized message passing algorithm over a given graph, representing the FEC code structure, by replacing node and edge message updates with trainable functions. Contrary to many other deep learning-based decoding approaches, the proposed solution enjoys scalability to arbitrary block lengths and the training is not limited by the curse of dimensionality. We benchmark our proposed decoder against state-of-the-art in conventional channel decoding as well as against recent deep learning-based results. For the (63,45) BCH code, our solution outperforms weighted belief propagation (BP) decoding by approximately 0.5dB and even for 5G NR LDPC codes, we observe a competitive performance when compared to conventional BP decoding. For the BCH codes, the resulting GNN decoder can be fully parametrized with only 9900 weights.

## Setup

Running this code requires [Sionna 0.10](https://nvlabs.github.io/sionna/).
To run the notebooks on your machine, you also need [Jupyter](https://jupyter.org).
We recommend Ubuntu 20.04, Python 3.8, and TensorFlow 2.8.

## Structure of this repository

The following notebooks may serve as starting point:

* [GNN_decoder_standalone.ipynb](GNN_decoder_standalone.ipynb) : Implements a standalone tutorial-style GNN-based decoder. This notebook can be directly executed in Google Colab. Please note that for the sake of simplicity not all functions of the library are available; please use the notebooks below for more sophisticated experiments.
* [GNN_decoder_universal.ipynb](GNN_decoder_universal.ipynb) : Implements a universal GNN-based decoder that can be configured for own experiments.
* [GNN_decoder_BCH.ipynb](GNN_decoder_BCH.ipynb) : Trains the GNN-decoder for the (63,45) BCH code. Reproduces Fig. 3 in [[A *PLACEHOLDER*]](*PLACEHOLDER*)
* [GNN_decoder_reg_LDPC.ipynb](GNN_decoder_reg_LDPC.ipynb) : Trains the GNN-decoder for regular (3,6) LDPC codes. Reproduces Fig. 4 in [[A *PLACEHOLDER*]](*PLACEHOLDER*)
* [GNN_decoder_LDPC_5G.ipynb](GNN_decoder_LDPC_5G.ipynb) : Trains the GNN-decoder to decode 5G NR LDPC codes. Reproduces Fig. 5 in [[A *PLACEHOLDER*]](*PLACEHOLDER*)

These notebooks rely on the following modules:

* [gnn.py](gnn.py) : Implements the GNN-based decoder and utility functions.
* [wbp.py](wbp.py) : Implements the weighted BP decoder and training for comparison [[B]](https://arxiv.org/abs/1607.04793). See [Sionna example notebook](https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html) for further details.

In addition, pre-trained weights are available in the [weights/](weights/) directory. This allows reproducing the results from [[A *PLACEHOLDER*]](*PLACEHOLDER*) without retraining the neural network. The result of each simulation can be found in [results/](results/).

## References

[A] [S. Cammerer, J. Hoydis, F. Aït Aoudia, and A. Keller, "Graph Neural Networks for Channel Decoding", 2022](*PLACEHOLDER*)

[B] [E. Nachmani, Y. Beery, D. Burshtein, "Learning to Decode Linear Codes Using Deep Learning", Allerton, 2016](https://arxiv.org/abs/1607.04793)

## License

Copyright &copy; 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA License](LICENSE.txt).
