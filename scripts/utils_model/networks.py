#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: networks.py
# Created Date: Friday, April 17th 2020, 11:11:26 am
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2020 Amelia Villegas-Morcillo
###


import torch
import torch.nn as nn
from utils_model.modules import MLP, GRU, CNN1D, ResCNN1D


class MultiClassCNNGRU(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Define CNN layers
        self.cnn_layers = CNN1D(
            input_dim=hparams.input_dim, channel_dims=hparams.channel_dims,
            kernel_sizes=hparams.kernel_sizes, activation=hparams.activation,
            drop_prob=hparams.drop_prob
        )
        # Define GRU layers
        self.gru_layers = GRU(
            input_dim=hparams.channel_dims[-1], gru_dim=hparams.gru_dim,
            gru_bidirec=hparams.gru_bidirec, gru_layers=hparams.gru_layers
        )
        # Define fully-connected layers
        self.mlp_layers = MLP(
            input_dim=hparams.gru_dim * (hparams.gru_bidirec + 1),
            hidden_dims=hparams.hidden_dims, activation=hparams.activation_last,
            drop_prob=hparams.drop_prob, batch_norm=hparams.batch_norm
        )
        # Define output layer
        self.out_layer = nn.Linear(hparams.hidden_dims[-1],
                                   hparams.num_classes)

    def forward(self, inp):
        x = inp.x.transpose(1, 2)   # (batch_size, num_channels, seq_len)
        x = self.cnn_layers(x, inp.seq_mask)
        x = x.transpose(1, 2)       # (batch_size, seq_len, num_channels)
        _, x = self.gru_layers(x, inp.seq_len)  # last hidden state
        emb = self.mlp_layers(x)
        out = self.out_layer(emb)
        return emb, out


class MultiClassResCNNGRU(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Define ResCNN layers
        self.cnn_layers = ResCNN1D(
            input_dim=hparams.input_dim, channel_dims=hparams.channel_dims,
            kernel_sizes=hparams.kernel_sizes, activation=hparams.activation,
            drop_prob=hparams.drop_prob
        )
        # Define GRU layers
        self.gru_layers = GRU(
            input_dim=hparams.channel_dims[-1], gru_dim=hparams.gru_dim,
            gru_bidirec=hparams.gru_bidirec, gru_layers=hparams.gru_layers
        )
        # Define fully-connected layers
        self.mlp_layers = MLP(
            input_dim=hparams.gru_dim * (hparams.gru_bidirec + 1),
            hidden_dims=hparams.hidden_dims, activation=hparams.activation_last,
            drop_prob=hparams.drop_prob, batch_norm=hparams.batch_norm
        )
        # Define output layer
        self.out_layer = nn.Linear(hparams.hidden_dims[-1],
                                   hparams.num_classes)

    def forward(self, inp):
        x = inp.x.transpose(1, 2)   # (batch_size, num_channels, seq_len)
        x = self.cnn_layers(x, inp.seq_mask)
        x = x.transpose(1, 2)       # (batch_size, seq_len, num_channels)
        _, x = self.gru_layers(x, inp.seq_len)  # last hidden state
        emb = self.mlp_layers(x)
        out = self.out_layer(emb)
        return emb, out
