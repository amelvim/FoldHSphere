#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: modules.py
# Created Date: Tuesday, April 28th 2020, 4:13:42 pm
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2020 Amelia Villegas-Morcillo
###


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def set_activation(activation):
    # Define activation functions
    activation_functions = nn.ModuleDict([
        ["relu", nn.ReLU()], ["lrelu", nn.LeakyReLU()],
        ["sigmoid", nn.Sigmoid()], ["tanh", nn.Tanh()]
    ])
    return activation_functions[activation]


class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512], activation="relu",
                 drop_prob=0, batch_norm=False):
        super().__init__()
        # Define fully-connected layers
        dims = [input_dim] + hidden_dims
        mlp_layers = []
        for m, n in zip(dims, dims[1:]):
            mlp_layers += [nn.Linear(m, n), set_activation(activation)]
            mlp_layers += [nn.BatchNorm1d(n)] if batch_norm else []
            mlp_layers += [nn.Dropout(drop_prob)]

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class GRU(nn.Module):
    def __init__(self, input_dim=1024, gru_dim=512, gru_bidirec=False,
                 gru_layers=1):
        super().__init__()
        # Define GRU layers
        self.num_gru_direc = 2 if gru_bidirec else 1
        self.num_gru_layers = gru_layers
        self.gru = nn.GRU(input_dim, batch_first=True, hidden_size=gru_dim,
                          bidirectional=gru_bidirec, num_layers=gru_layers)

    def forward(self, x, seq_len):
        # Compute GRU part, input of shape (batch_size, seq_len, num_channels)
        x_pack = pack_padded_sequence(x, seq_len, batch_first=True,
                                      enforce_sorted=False)
        hs_pack, h_n = self.gru(x_pack) # all hidden states (t=1,...,seq_len)
                                        # hidden state for t=seq_len

        # Unpack hidden states (add zero padding)
        # (batch_size, seq_len, hidden_size * num_directions)
        hs, _ = pad_packed_sequence(hs_pack, batch_first=True)

        # Take last part of hidden state vector (last GRU layer)
        # (num_layers, num_directions, batch_size, hidden_size)
        h_n = h_n.view(self.num_gru_layers, self.num_gru_direc,
                       h_n.size(1), h_n.size(2))
        h_n = h_n[-1, :, :, :]

        # Remove direction dimension or concatenate two directions
        # (batch_size, hidden_size * num_directions)
        if self.num_gru_direc == 1:
            return (hs, h_n[0, :, :])
        elif self.num_gru_direc == 2:
            return (hs, torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1))


class CNN1D(nn.Module):
    def __init__(self, input_dim=1024, channel_dims=[128, 256],
                 kernel_sizes=[5, 5], dilations=[1, 1], activation="relu",
                 drop_prob=0):
        super().__init__()
        # Define 1D-convolutional layers
        dims = [input_dim] + channel_dims
        cnn_layers = []
        for m, n, k, d in zip(dims, dims[1:], kernel_sizes, dilations):
            cnn_layers += [nn.Sequential(
                nn.Conv1d(m, n, kernel_size=k, padding=int((k-1)/2)*d, dilation=d),
                set_activation(activation),
                nn.BatchNorm1d(n),
                nn.Dropout(drop_prob)
            )]
        self.conv = nn.ModuleList(cnn_layers)

    def forward(self, x, seq_mask):
        for conv_layer in self.conv:
            x = conv_layer(x)
            x = torch.mul(x, seq_mask)  # apply mask
        return x


class ResCNN1D(nn.Module):
    def __init__(self, input_dim=1024, channel_dims=[64, 256, 64, 256],
                 kernel_sizes=[5, 5], dilations=[1, 1], activation="relu",
                 drop_prob=0):
        super().__init__()
        # Define dimensions
        btneck_dim = channel_dims[0]
        conv_dim = channel_dims[1]
        # Define initial 1D-convolutional layer (upsampling / downsampling)
        self.conv_init = CNN1D(
            input_dim, channel_dims=[conv_dim], kernel_sizes=[1],
            dilations=[1], activation=activation, drop_prob=drop_prob
        )
        # Define 1D-convolutional residual blocks with dilation
        residual_blocks = []
        for k, d in zip(kernel_sizes, dilations):
            # Bottleneck residual block (two 1D-convolutions)
            residual_blocks += [CNN1D(
                conv_dim, channel_dims=[btneck_dim, conv_dim],
                kernel_sizes=[k, k], dilations=[d, d], activation=activation,
                drop_prob=drop_prob
            )]
        self.conv_res = nn.ModuleList(residual_blocks)

    def forward(self, x, seq_mask):
        # Compute initial convolution
        x = self.conv_init(x, seq_mask)
        # Compute residual blocks
        res = x     # initial residual
        for block in self.conv_res:     # each residual block
            # Compute two convolutional layers
            out = block(res, seq_mask)
            # Compute new residual
            res = out + res
        return res
