#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a test of the entire code on dummy data. """
import numpy as np
import h5py
import os
import shutil
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from unittest import TestCase
from orcanet.utilities.input_output_utilities import Settings
from orcanet.run_nn import orca_train
from orcanet.model_setup import build_nn_model


class DatasetTest(TestCase):
    """ Tests which require a dataset. """
    def setUp(self):
        """ Generate some dummy data for the models in a temp folder, and a list file. """
        self.temp_dir = os.path.join(os.path.dirname(__file__), ".temp/")
        print("Using temp directory: " + self.temp_dir)
        os.makedirs(self.temp_dir)
        shape = (5, 5, 5, 5)
        train_filepath = self.temp_dir + "train.h5"
        val_filepath = self.temp_dir + "val.h5"
        make_dummy_data(train_filepath, shape)
        make_dummy_data(val_filepath, shape)

        self.list_file_path = self.temp_dir + "list.toml"
        with open(self.list_file_path, "w") as list_file:
            list_content = '[[input]]\ntrain_files = ["{}",]\nvalidation_files = ["{}",]'.format(train_filepath, val_filepath)
            list_file.write(list_content)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_multi_input_model(self):
        """ Make a model and train it with the test toml files provided. """
        list_file = self.list_file_path
        config_file = os.path.join(os.path.dirname(__file__), "test_config.toml")
        model_file = os.path.join(os.path.dirname(__file__), "test_model.toml")

        main_folder = self.temp_dir + "model/"
        cfg = Settings(main_folder, list_file, config_file)
        cfg.zero_center_folder = self.temp_dir
        cfg.set_from_model_file(model_file)
        initial_model = build_nn_model(cfg)
        orca_train(cfg, initial_model)


def make_dummy_data(filepath, shape):
    x = np.concatenate([np.ones((500,) + shape), np.zeros((500,) + shape)])
    y = np.ones((1000, 14))

    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset('x', data=x, dtype='uint8')
    h5f.create_dataset('y', data=y, dtype='float32')
    h5f.close()


def make_dummy_model(shape):
    inp = Input(shape=shape)
    x = Flatten()(inp)
    x = Dense(5)(x)
    model = Model(inp, x)
    model.compile("sgd", loss="mse")
    return model
