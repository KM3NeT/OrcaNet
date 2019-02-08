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

from orcanet.core import OrcaHandler
from orcanet.model_archs.model_setup import OrcaModel

from orcanet.utilities.nn_utilities import load_zero_center_data


class DatasetTest(TestCase):
    """ Tests which require a dataset. """
    def setUp(self):
        """
        Make a .temp directory in the current working directory, generate dummy data in it and set up the
        cfg object.

        """
        self.temp_dir = os.path.join(os.getcwd(), ".temp/")
        # Make sure temp dir does either not exist or is empty
        if os.path.exists(self.temp_dir):
            assert len(os.listdir(self.temp_dir)) == 0
        else:
            os.makedirs(self.temp_dir)

        # Make dummy data of given shape
        self.shape = (3, 3, 3, 3)
        train_inp = (self.temp_dir + "train1.h5", self.temp_dir + "train2.h5")
        self.train_pathes = {"input_1": train_inp}
        val_inp = (self.temp_dir + "val1.h5", self.temp_dir + "val2.h5")
        self.val_pathes = {"input_1": val_inp}
        for path1, path2 in (train_inp, val_inp):
            make_dummy_data(path1, path2, self.shape)

        # Set up the configuration object
        config_file = os.path.join(os.path.dirname(__file__), "config_test.toml")
        output_folder = self.temp_dir + "model/"

        orca = OrcaHandler(output_folder, config_file=config_file)
        orca.cfg._train_files = self.train_pathes
        orca.cfg._val_files = self.val_pathes
        orca.cfg._list_file = "test.toml"
        orca.cfg.zero_center_folder = self.temp_dir

        self.orca = orca

    def tearDown(self):
        """ Remove the .temp directory. """
        shutil.rmtree(self.temp_dir)

    def test_zero_center(self):
        """ Calculate the zero center image and check if it works properly. """
        orca = self.orca
        xs_mean = load_zero_center_data(orca)
        target_xs_mean = np.ones(self.shape)/4
        self.assertTrue(np.allclose(xs_mean["input_1"], target_xs_mean))

        file = orca.cfg.zero_center_folder + orca.cfg._list_file + '_input_' + "input_1" + '.npz'
        zero_center_used_ip_files = np.load(file)['zero_center_used_ip_files']
        self.assertTrue(np.array_equal(zero_center_used_ip_files, orca.cfg._train_files["input_1"]))

    def test_multi_input_model(self):
        """
        Make a model and train it with the test toml files provided to check if it throws an error.
        Also resumes training after the first epoch with a custom lr to check if that works.
        """
        orca = self.orca
        model_file = os.path.join(os.path.dirname(__file__), "model_test.toml")

        orcamodel = OrcaModel(model_file)
        initial_model = orcamodel.build(orca)
        orcamodel.update_orca(orca)

        orca.train(initial_model)

        def test_learning_rate(epoch, fileno, cfg):
            lr = (1 + epoch)*(1 + fileno) * 0.001
            return lr

        def test_modifier(xs):
            xs = {key: xs[key] * 2 for key in xs}
            return xs

        orca.cfg.learning_rate = test_learning_rate
        orca.cfg.sample_modifier = test_modifier
        orca.train()
        orca.predict()


def make_dummy_data(filepath1, filepath2, shape):
    """
    Make a total of 100 ones vs 300 zeroes of dummy data over two files.

    Parameters
    ----------
    filepath1 : str
        Path to file 1.
    filepath2 : str
        Path to file 2.
    shape : tuple
        Shape of the data, not including sample dimension.

    """
    xs1 = np.concatenate([np.ones((75,) + shape), np.zeros((75,) + shape)])
    xs2 = np.concatenate([np.ones((25,) + shape), np.zeros((225,) + shape)])

    dtypes = [('event_id', '<f8'), ('particle_type', '<f8'), ('energy', '<f8'), ('is_cc', '<f8'), ('bjorkeny', '<f8'),
              ('dir_x', '<f8'), ('dir_y', '<f8'), ('dir_z', '<f8'), ('time_interaction', '<f8'), ('run_id', '<f8'),
              ('vertex_pos_x', '<f8'), ('vertex_pos_y', '<f8'), ('vertex_pos_z', '<f8'), ('time_residual_vertex', '<f8'),
              ('prod_ident', '<f8'), ('group_id', '<i8')]
    ys1 = np.ones((150, 16)).ravel().view(dtype=dtypes)
    ys2 = np.ones((250, 16)).ravel().view(dtype=dtypes)

    for xs, ys, filepath in ((xs1, ys1, filepath1), (xs2, ys2, filepath2)):
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
