#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: dataloaders.py
# Created Date: Thursday, September 24th 2020, 12:51:33 pm
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2020 Amelia Villegas-Morcillo
###


import numpy as np
import torch
from torch.utils.data import DataLoader
from utils_data.data import Data
from utils_data.datasets import SingleDataset


def single_sequence_collate(batch):
    # Get features, label, length and name (from a list of arrays)
    (x, y, lens, names) = zip(*batch)

    # Pad features to max sequence length in the batch
    max_len = max(lens)
    x_pad = [np.pad(item, ((0, max_len-lens[i]), (0, 0)), "constant") \
             for i, item in enumerate(x)]

    # Create mask for each sample
    masks = [np.expand_dims([1]*i + [0]*(max_len - i), axis=0) for i in lens]

    # Convert to tensors and return Data object
    return Data(x=torch.from_numpy(np.array(x_pad)),
                y=torch.from_numpy(np.array(y)),
                seq_len=torch.from_numpy(np.array(lens)),
                seq_mask=torch.from_numpy(np.array(masks, dtype=np.float32)),
                name=np.array(names))


def get_train_loader_class(args):
    ''' Method to get the train dataset and dataloaders '''
    train_set = SingleDataset(
        args.train_file, args.feats_dir, sep=args.scop_separation,
        fold_label_file=args.fold_label_file
    )
    return DataLoader(train_set, batch_size=args.batch_size_class,
                      shuffle=True, num_workers=args.ndata_workers,
                      drop_last=True, collate_fn=single_sequence_collate)


def get_valid_loader_class(args):
    ''' Method to get the validation dataset and dataloader '''
    valid_set = SingleDataset(
        args.valid_file, args.feats_dir, sep=args.scop_separation,
        fold_label_file=args.fold_label_file
    )
    return DataLoader(valid_set, batch_size=args.batch_size_class,
                      num_workers=args.ndata_workers,
                      collate_fn=single_sequence_collate)


def get_test_loader_class(args):
    ''' Method to get the test dataset and dataloader '''
    test_set = SingleDataset(
        args.test_file, args.feats_dir_test, sep=args.scop_separation
    )
    return DataLoader(test_set, batch_size=1, num_workers=args.ndata_workers,
                      collate_fn=single_sequence_collate)
