#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import shutil
import h5py
from unittest import TestCase
from keras.models import Model
from keras.layers import Dense, Input, Concatenate

from orcanet.core import Organizer
from orcanet.utilities.nn_utilities import load_zero_center_data, get_layer_output


class TetstZeroCenter(TestCase):
    def setUp(self):
        # Pathes
        # Temporary output folder
        self.temp_dir = os.path.join(os.path.dirname(__file__),
                                     ".temp", "nn_utilities")
        self.output_folder = os.path.join(self.temp_dir, "model")
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")

        config_file = os.path.join(self.data_folder, "config_test.toml")

        # Make sure temp dir does either not exist or is empty
        if os.path.exists(self.temp_dir):
            assert len(os.listdir(self.temp_dir)) == 0
        else:
            os.makedirs(self.temp_dir)

        # Make dummy data
        self.train_inp_A = (
            self.temp_dir + "/train1_A.h5",
            self.temp_dir + "/train2_A.h5"
        )
        self.train_inp_B = (
            self.temp_dir + "/train1_B.h5",
            self.temp_dir + "/train2_B.h5",
        )

        self.train_pathes = {
            "testing_input_A": self.train_inp_A,
            "testing_input_B": self.train_inp_B,
        }
        self.shape = (3, 3, 3)

        make_dummy_data(self.train_inp_A[0], self.train_inp_A[1], self.shape, mode=1)
        make_dummy_data(self.train_inp_B[0], self.train_inp_B[1], self.shape, mode=2)

        orga = Organizer(self.output_folder, config_file=config_file)
        orga.cfg._files_dict["train"] = self.train_pathes
        orga.cfg._list_file = "test.toml"
        orga.cfg.zero_center_folder = self.temp_dir

        self.orga = orga

    def tearDown(self):
        """ Remove the .temp directory. """
        shutil.rmtree(self.temp_dir)

    def test_load_zero_center(self):
        """ Calculate the zero center image and check if it works properly. """
        orga = self.orga
        xs_mean = load_zero_center_data(orga, logging=False)

        target = {
            "testing_input_A": np.ones(self.shape) * 0.25,
            "testing_input_B": np.ones(self.shape) * 0.75,
        }

        self._check_dict_ndarray(xs_mean, target)

        file_A = orga.cfg.zero_center_folder + "/" + orga.cfg._list_file + \
            '_input_testing_input_A.npz'
        file_B = orga.cfg.zero_center_folder + "/" + orga.cfg._list_file + \
            '_input_testing_input_B.npz'
        used_files_A = np.load(file_A)['zero_center_used_ip_files']
        used_files_B = np.load(file_B)['zero_center_used_ip_files']
        self.assertTrue(np.array_equal(used_files_A, self.train_inp_A))
        self.assertTrue(np.array_equal(used_files_B, self.train_inp_B))

        # test loading of saved xs_mean
        os.remove(file_B)
        xs_mean = load_zero_center_data(orga, logging=False)

        self._check_dict_ndarray(xs_mean, target)
        used_files_A = np.load(file_A)['zero_center_used_ip_files']
        used_files_B = np.load(file_B)['zero_center_used_ip_files']
        self.assertTrue(np.array_equal(used_files_A, self.train_inp_A))
        self.assertTrue(np.array_equal(used_files_B, self.train_inp_B))

    def _check_dict_ndarray(self, val, target):
        self.assertSetEqual(set(val.keys()), set(target.keys()))
        for key in val.keys():
            np.testing.assert_array_almost_equal(val[key], target[key])


class TestModel(TestCase):
    def setUp(self):
        self.layer_names = ["inp1", "inp2", "mid", "outp"]

        inp1 = Input(shape=(2, ), name=self.layer_names[0])
        inp2 = Input(shape=(3, ), name=self.layer_names[1])

        conc = Concatenate()([inp1, inp2])

        intermed = Dense(2,
                         name=self.layer_names[2],
                         bias_initializer="Ones",
                         kernel_initializer="Ones")(conc)
        outp = Dense(1, name=self.layer_names[3])(intermed)

        self.model = Model([inp1, inp2], outp)

    def test_get_layer_output_list_input(self):
        samples = [np.ones((2, 2)), np.ones((2, 3))]

        args = {
            "model": self.model,
            "samples": samples,
            "layer_name": self.layer_names[2],
            "mode": "test",
        }
        value = get_layer_output(**args)
        target = [[6., 6.], [6., 6.]]
        np.testing.assert_array_equal(value, target)

    def test_get_layer_output_dict_input(self):
        samples = {self.layer_names[0]: np.ones((2, 2)),
                   self.layer_names[1]: np.ones((2, 3))}

        args = {
            "model": self.model,
            "samples": samples,
            "layer_name": self.layer_names[2],
            "mode": "test",
        }
        value = get_layer_output(**args)
        target = [[6., 6.], [6., 6.]]
        np.testing.assert_array_equal(value, target)


def make_dummy_data(filepath1, filepath2, shape, mode=1):
    """
    Make two dummy data files.

    mode 1:
      Make a total of 100 ones vs 300 zeroes of dummy data over two files.
      -- > xs_mean = 0.25
    mode 2:
      Make a total of 300 ones vs 100 zeroes of dummy data over two files.
      -- > xs_mean = 0.75

    Parameters
    ----------
    filepath1 : str
        Path to file 1.
    filepath2 : str
        Path to file 2.
    shape : tuple
        Shape of the data, not including sample dimension.

    """
    xs1 = np.ones((150,) + shape)
    xs2 = np.ones((250,) + shape)

    if mode == 1:
        xs1[100:, ...] = 0
        xs2[:, ...] = 0

    elif mode == 2:
        xs2[-100:, ...] = 0

    else:
        raise AssertionError

    dtypes = [('event_id', '<f8'), ('particle_type', '<f8'), ('energy', '<f8'),
              ('is_cc', '<f8'), ('bjorkeny', '<f8'), ('dir_x', '<f8'),
              ('dir_y', '<f8'), ('dir_z', '<f8'), ('time_interaction', '<f8'),
              ('run_id', '<f8'), ('vertex_pos_x', '<f8'), ('vertex_pos_y', '<f8'),
              ('vertex_pos_z', '<f8'), ('time_residual_vertex', '<f8'),
              ('prod_ident', '<f8'), ('group_id', '<i8')]
    ys1 = np.ones((150, 16)).ravel().view(dtype=dtypes)
    ys2 = np.ones((250, 16)).ravel().view(dtype=dtypes)

    for xs, ys, filepath in ((xs1, ys1, filepath1), (xs2, ys2, filepath2)):
        h5f = h5py.File(filepath, 'w')
        h5f.create_dataset('x', data=xs, dtype='uint8')
        h5f.create_dataset('y', data=ys, dtype=dtypes)
        h5f.close()
