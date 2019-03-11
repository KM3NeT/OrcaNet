#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import os
import shutil
import h5py
from unittest import TestCase

from orcanet.core import Organizer
from orcanet.utilities.nn_utilities import load_zero_center_data


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
        self.train_inp = (
            self.temp_dir + "/train1.h5",
            self.temp_dir + "/train2.h5"
        )
        self.train_pathes = {"testing_input": self.train_inp}
        self.shape = (3, 3, 3, 3)

        make_dummy_data(self.train_inp[0], self.train_inp[1], self.shape)

        orga = Organizer(self.output_folder, config_file=config_file)
        orga.cfg._train_files = self.train_pathes
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

        target = np.ones(self.shape)/4
        self.assertTrue(np.allclose(xs_mean["testing_input"], target))

        file = orga.cfg.zero_center_folder + "/" + orga.cfg._list_file + \
            '_input_' + "testing_input" + '.npz'
        used_files = np.load(file)['zero_center_used_ip_files']
        self.assertTrue(np.array_equal(used_files, self.train_inp))


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
