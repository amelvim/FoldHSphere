#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: datasets.py
# Created Date: Thursday, September 24th 2020, 12:58:05 pm
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2020 Amelia Villegas-Morcillo
###


import numpy as np
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, data_file, feats_dir, sep=".", fold_label_file=None):
        # Initialize data
        data = np.loadtxt(data_file, dtype=str)
        self.names = list(data[:, 0])
        self.lengths = data[:, 1].astype("int")
        self.feats_dir = feats_dir

        # Get family, superfamily, and fold categories
        self.fams = data[:, 2]
        self.supfams = np.array([item.rsplit(sep, 1)[0] for item in self.fams])
        self.folds = data[:, 3]

        # Convert folds to labels (only for training)
        if fold_label_file is not None:
            fold_label = np.loadtxt(fold_label_file, dtype=str)
            fold_label_dict = {item[0]: int(item[1]) for item in fold_label[1:]}
            self.labels = [fold_label_dict[f] for f in self.folds]
        else:
            self.labels = np.zeros((len(self.names)))

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample (name, length and label)
        name = self.names[index]
        seq_len = self.lengths[index]
        label = self.labels[index]

        # Load embedding feature matrix
        x = np.load("%s/%s.npy" % (self.feats_dir, name))  # length x dim

        return x, label, seq_len, name
