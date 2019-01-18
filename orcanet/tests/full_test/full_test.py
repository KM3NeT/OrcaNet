#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a test of the entire code on dummy data. """
import numpy as np
import h5py
import os
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from orcanet.utilities.input_output_utilities import Settings
from orcanet.run_nn import orca_train
from orcanet.model_setup import build_nn_model


def test_run():
    """
    Test Orcatrain.
    """
    temp_dir = "temp/"

    main_folder = temp_dir + "model/"
    list_file = "test_list.toml"
    config_file = "test_config.toml"
    model_file = "test_model.toml"

    initial_model = preperation(temp_dir)
    cfg = Settings(main_folder, list_file, config_file)
    cfg.zero_center_folder = temp_dir
    cfg.set_from_model_file(model_file)
    initial_model = build_nn_model(cfg)

    orca_train(cfg, initial_model)


def make_dummy_data(name, temp_dir):
    filepath = temp_dir + name + ".h5"
    shape = (10, 10, 10, 10)

    x = np.concatenate([np.ones((500,) + shape), np.zeros((500,) + shape)])
    y = np.ones((1000, 14))

    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset('x', data=x, dtype='uint8')
    h5f.create_dataset('y', data=y, dtype='float32')
    h5f.close()
    print("Created file ", filepath)
    return shape


def make_dummy_model(shape):
    inp = Input(shape=shape)
    x = Flatten()(inp)
    x = Dense(5)(x)
    model = Model(inp, x)
    model.compile("sgd", loss="mse")
    return model


def preperation(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    shape = make_dummy_data("train", temp_dir)
    make_dummy_data("val", temp_dir)
    model = make_dummy_model(shape)
    return model


if __name__ == "__main__":
    test_run()
    # os.remove(filepath)
