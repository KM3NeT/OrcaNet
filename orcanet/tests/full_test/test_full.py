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

from orcanet.core import Configuration, orca_train
from orcanet.model_archs.model_setup import build_nn_model


class DatasetTest(TestCase):
    """ Tests which require a dataset. """
    def setUp(self):
        """
        Make a .temp directory in the current working directory, generate dummy data in it and also a list .toml file which points to these datasets.
        """
        self.temp_dir = os.path.join(os.getcwd(), ".temp/")
        self.list_file_path = self.temp_dir + "list.toml"

        # Make sure temp dir does either not exist or is empty
        assert not os.path.exists(self.temp_dir) or len(os.listdir('/home/varun/temp')) == 0
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
        Also resumes training after the first epoch with a custom lr to check if that works.
        """
        list_file = self.list_file_path
        config_file = os.path.join(os.path.dirname(__file__), "config_test.toml")
        model_file = os.path.join(os.path.dirname(__file__), "model_test.toml")

        main_folder = self.temp_dir + "model/"
        cfg = Configuration(main_folder, list_file, config_file)
        cfg.zero_center_folder = self.temp_dir
        cfg.set_from_model_file(model_file)
        initial_model = build_nn_model(cfg)
        orca_train(cfg, initial_model)

        def test_learning_rate(epoch, fileno, cfg):
            lr = (1 + epoch)*(1 + fileno) * 0.001
            return lr
        cfg.learning_rate = test_learning_rate
        orca_train(cfg)
        # orca_eval(cfg)


def make_dummy_data(filepath, shape):
    xs = np.concatenate([np.ones((100,) + shape), np.zeros((100,) + shape)])

    y = np.ones((200, 16))
    dtypes = [('event_id', '<f8'), ('particle_type', '<f8'), ('energy', '<f8'), ('is_cc', '<f8'), ('bjorkeny', '<f8'),
           ('dir_x', '<f8'), ('dir_y', '<f8'), ('dir_z', '<f8'), ('time_interaction', '<f8'), ('run_id', '<f8'),
           ('vertex_pos_x', '<f8'), ('vertex_pos_y', '<f8'), ('vertex_pos_z', '<f8'), ('time_residual_vertex', '<f8'),
           ('prod_ident', '<f8'), ('group_id', '<i8')]
    ys = y.ravel().view(dtype=dtypes)

    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset('x', data=xs, dtype='uint8')
    h5f.create_dataset('y', data=ys, dtype=dtypes)
    h5f.close()


def make_dummy_model(shape):
    inp = Input(shape=shape)
    x = Flatten()(inp)
    x = Dense(5)(x)
    model = Model(inp, x)
    model.compile("sgd", loss="mse")
    return model
