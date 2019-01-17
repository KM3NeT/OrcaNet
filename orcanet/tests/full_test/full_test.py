#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a test of the entire code on dummy data. """
import numpy as np
import h5py
import os
from orcanet.utilities.input_output_utilities import Settings
from orcanet.run_nn import orca_train
from orcanet.model_setup import build_nn_model


def example_run():
    """
    This shows how to use OrcaNet.
    """
    # Set up the cfg object with the input data
    cfg = Settings(main_folder, list_file, config_file)
    # If this is the start of the training, a compiled model needs to be handed to the orca_train function
    if cfg.get_latest_epoch() == (0, 1):
        # Add Info for building a model with OrcaNet to the cfg object
        cfg.set_from_model_file(model_file)
        # Build it
        initial_model = build_nn_model(cfg)
    else:
        # No model is required if the training is continued, as it will be loaded automatically
        initial_model = None
    orca_train(cfg, initial_model)


def make_dummy_data(name, delete=False):
    filepath = "temp/" + name + ".h5py"
    if not delete:
        x = np.concatenate([np.ones((100, 10, 10)), np.zeros((100, 10, 10))])
        y = np.ones((200, 1))
        h5f = h5py.File(filepath, 'w')
        h5f.create_dataset('x', data=x, dtype='uint8')
        h5f.create_dataset('y', data=y, dtype='float32')
        h5f.close()
        print("Created file ", filepath)
    else:
        os.remove(filepath)
        print("Deleted file ", filepath)


def preperation(delete):
    make_dummy_data("train", delete=delete)
    make_dummy_data("test", delete=delete)
