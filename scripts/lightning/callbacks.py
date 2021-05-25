#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: callbacks.py
# Created Date: Wednesday, October 5th 2020, 11:35:17 am
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2020 Amelia Villegas-Morcillo
###


import pickle
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class MyModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        pass
    def on_epoch_end(self, trainer, pl_module):
        self.save_checkpoint(trainer, pl_module)


class TestSerializer(Callback):
    def __init__(self, emb_path=None):
        self.emb_path = emb_path
    def on_test_end(self, trainer, pl_module):
        if self.emb_path is not None:
            with open(self.emb_path, "wb") as f:
                pickle.dump(pl_module.embeddings, f)
