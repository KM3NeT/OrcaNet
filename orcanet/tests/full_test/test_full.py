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
from orcanet.eval_nn import orca_eval


class DatasetTest(TestCase):
    """ Tests which require a dataset. """
    def setUp(self):
        """
        Make a .temp directory in the current working directory, generate dummy data in it and also a list .toml file which points to these datasets.
        """
        self.temp_dir = os.path.join(os.getcwd(), ".temp/")
        self.list_file_path = self.temp_dir + "list.toml"

        assert not os.path.exists(self.temp_dir)
        os.makedirs(self.temp_dir)

        shape = (3, 3, 3, 3)
        train_filepath = self.temp_dir + "train.h5"
        val_filepath = self.temp_dir + "val.h5"
        make_dummy_data(train_filepath, shape)
        make_dummy_data(val_filepath, shape)
        with open(self.list_file_path, "w") as list_file:
            list_content = '[[input]]\ntrain_files = ["{}",]\nvalidation_files = ["{}",]'.format(train_filepath, val_filepath)
            list_file.write(list_content)

    def tearDown(self):
        """ Remove the .temp directory. """
        shutil.rmtree(self.temp_dir)

    def test_multi_input_model(self):
        """
        Make a model and train it with the test toml files provided to check if it throws an error.
        Also resumes training after the first epoch to check if that works.
        """
        list_file = self.list_file_path
        config_file = os.path.join(os.path.dirname(__file__), "test_config.toml")
        model_file = os.path.join(os.path.dirname(__file__), "test_model.toml")

        main_folder = self.temp_dir + "model/"
        cfg = Settings(main_folder, list_file, config_file)
        cfg.zero_center_folder = self.temp_dir
        cfg.class_type = ['None', 'energy_dir_bjorken-y_vtx_errors']
        cfg.set_from_model_file(model_file)
        initial_model = build_nn_model(cfg)
        orca_train(cfg, initial_model)
        orca_train(cfg)
        orca_eval(cfg)


def make_dummy_data(filepath, shape):
    x = np.concatenate([np.ones((100,) + shape), np.zeros((100,) + shape)])
    y = np.ones((200, 14))

    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset('x', data=x, dtype='uint8')
    h5f.create_dataset('y', data=y, dtype='float32')
    h5f.close()
