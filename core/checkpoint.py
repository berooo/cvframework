#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os
import copy

import torch
from config.defaults import _C as cfg


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))



def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike

    startepoch=0
    min_loss=0
    step=0

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['loss']
        step = checkpoint['step']
        print("load checkpoint successful!", flush=True)
    else:
        print("load checkpoint unsuccessful!",flush=True)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print('optimizer load successfully.')
    #return checkpoint["epoch"]
    return startepoch,min_loss,step
