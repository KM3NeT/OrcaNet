#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a test of the entire code on dummy data. """
import numpy as np
import h5py
import os
from keras.models import Model
from keras.layers import Dense, Input
from orcanet.utilities.input_output_utilities import Settings
from orcanet.run_nn import orca_train
# from orcanet.model_setup import build_nn_model


def example_run():
    """
    This shows how to use OrcaNet.
    """
    temp_dir = "temp/"

    main_folder = temp_dir + "model/"
    list_file = temp_dir + "test_list.toml"
    config_file = temp_dir + "test_config.toml"
    # model_file = temp_dir + "test_model.toml"

    cfg = Settings(main_folder, list_file, config_file)
    initial_model = preperation(temp_dir)

    orca_train(cfg, initial_model)

# os.remove(filepath)


def make_dummy_data(name, temp_dir):
    filepath = temp_dir + name + ".h5"
    shape = (10, )

    x = np.concatenate([np.ones((1000,) + shape), np.zeros((1000,) + shape)])
    y = np.ones((2000, 1))

    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset('x', data=x, dtype='uint8')
    h5f.create_dataset('y', data=y, dtype='float32')
    h5f.close()
    print("Created file ", filepath)
    return shape


def make_dummy_model(shape):
    inp = Input(shape=shape)
    x = Dense(10)(inp)
    model = Model(inp, x)
    return model


def preperation(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    shape = make_dummy_data("train", temp_dir)
    make_dummy_data("val", temp_dir)
    model = make_dummy_model(shape)
    return model
