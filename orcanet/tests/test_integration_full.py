#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a test of the entire code on dummy data. """
import tempfile
import numpy as np
import h5py
import os
import sys
import shutil

import tensorflow as tf
import orcanet
from orcanet.core import Organizer
from orcanet.model_builder import ModelBuilder
from orcanet_contrib.custom_objects import get_custom_objects


class TestIntegration(tf.test.TestCase):
    """ Run the actual training to see if it throws any errors. """
    def setUp(self):
        """
        Make a .temp directory in the current working directory, generate
        dummy data in it and set up the cfg object.

        """
        # Pathes
        # Temporary output folder
        self.tdir = tempfile.TemporaryDirectory()
        self.temp_dir = self.tdir.name
        self.output_folder = os.path.join(self.temp_dir, "model")

        # Pathes to temp dummy data that will get generated
        train_inp = (self.temp_dir + "/train1.h5", self.temp_dir + "/train2.h5")
        self.train_pathes = {"testing_input": train_inp}

        val_inp = (self.temp_dir + "/val1.h5", self.temp_dir + "/val2.h5")
        self.val_pathes = {"testing_input": val_inp}

        # The config file to load
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        config_file = os.path.join(self.data_folder, "config_test.toml")

        # Make dummy data of given shape
        self.shape = (3, 3, 3, 3)
        for path1, path2 in (train_inp, val_inp):
            make_dummy_data(path1, path2, self.shape)

        def make_orga():
            orga = Organizer(self.output_folder, config_file=config_file)
            orga.cfg._files_dict["train"] = self.train_pathes
            orga.cfg._files_dict["val"] = self.val_pathes
            orga.cfg._list_file = "test.toml"
            orga.cfg.zero_center_folder = self.temp_dir
            orga.cfg.label_modifier = label_modifier
            orga.cfg.custom_objects = get_custom_objects()
            return orga

        self.make_orga = make_orga

    def tearDown(self):
        """ Remove the .temp directory. """
        self.tdir.cleanup()

    def test_integration_zero_center(self):
        """ Calculate the zero center image and check if it works properly. """
        orga = self.make_orga()
        xs_mean = orga.get_xs_mean()
        target_xs_mean = np.ones(self.shape)/4
        self.assertTrue(np.allclose(xs_mean["testing_input"], target_xs_mean))

        file = orga.cfg.zero_center_folder + "/" + orga.cfg._list_file + \
            '_input_' + "testing_input" + '.npz'
        zero_center_used_ip_files = np.load(file)['zero_center_used_ip_files']
        self.assertTrue(np.array_equal(zero_center_used_ip_files,
                                       orga.cfg._files_dict["train"]["testing_input"]))

    def test_integration_multi_input_model(self):
        """
        Run whole script on dummy data to see if it throws an error.

        Build a model with the ModelBuilder.
        Train for 2 epochs.
        Reset organizer.
        Resume for 1 epoch with different lr and sample modifier.
        Use orga.train and validate once each.
        Predict.

        """
        orga = self.make_orga()

        model_file = os.path.join(self.data_folder, "model_test.toml")
        builder = ModelBuilder(model_file)
        initial_model = builder.build(orga)

        orga.train_and_validate(initial_model, epochs=2)

        def test_learning_rate(epoch, fileno):
            lr = 0.001 * (epoch + 0.1*fileno)
            return lr

        def test_modifier(info_blob):
            xs = info_blob["x_values"]
            xs = {key: xs[key] for key in xs}
            return xs

        orga = self.make_orga()
        orga.cfg.learning_rate = test_learning_rate
        orga.cfg.sample_modifier = test_modifier
        orga.train_and_validate(epochs=1)
        orga.train()
        orga.validate()
        orga.predict()


class TestFullExample(tf.test.TestCase):
    """
    Test the full example in the examples folder, as it is used on the docs.
    Must go thorugh without any errors.
    """
    def test_full_example(self):
        init_dir = os.getcwd()

        example_dir = os.path.join(
            os.path.dirname(orcanet.__path__[0]), "examples/full_example")
        os.chdir(example_dir)
        sys.path.append(example_dir)
        output_dir = os.path.join(example_dir, "output")
        if os.path.exists(output_dir):
            raise FileExistsError(output_dir)
        try:
            import full_example
        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            os.chdir(init_dir)


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

    dtypes = [('dir_x', '<f8'), ]
    ys1 = np.ones((150, len(dtypes))).ravel().view(dtype=dtypes)
    ys2 = np.ones((250, len(dtypes))).ravel().view(dtype=dtypes)

    for xs, ys, filepath in ((xs1, ys1, filepath1), (xs2, ys2, filepath2)):
        h5f = h5py.File(filepath, 'w')
        h5f.create_dataset('x', data=xs, dtype='uint8')
        h5f.create_dataset('y', data=ys, dtype=dtypes)
        h5f.close()


def label_modifier(info_blob):
    y_values = info_blob["y_values"]
    ys = dict()

    ys['dx'], ys['dx_err'] = y_values['dir_x'], y_values['dir_x']

    for key_label in ys:
        ys[key_label] = ys[key_label].astype(np.float32)
    return ys
