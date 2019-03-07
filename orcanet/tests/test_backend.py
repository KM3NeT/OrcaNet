from unittest import TestCase
import os
import h5py
import numpy as np

from orcanet.core import Organizer
from orcanet.backend import hdf5_batch_generator, get_datasets, get_learning_rate


class TestBatchGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "test_backend")
        os.mkdir(cls.temp_dir)

        # make some dummy data
        cls.n_bins = {'input_A': (2, 3), 'input_B': (2, 3)}
        cls.train_sizes = [3, 5]
        cls.train_A_file_1 = {
            "path": cls.temp_dir + "/input_A_train_1.h5",
            "shape": cls.n_bins["input_A"],
            "size": cls.train_sizes[0],
        }

        cls.train_B_file_1 = {
            "path": cls.temp_dir + "/input_B_train_1.h5",
            "shape": cls.n_bins["input_B"],
            "size": cls.train_sizes[0],
        }

        cls.filepaths_file_1 = {
            "input_A": cls.train_A_file_1["path"],
            "input_B": cls.train_B_file_1["path"],
        }

        cls.train_A_file_1_ctnt = save_dummy_h5py(**cls.train_A_file_1)
        cls.train_B_file_1_ctnt = save_dummy_h5py(**cls.train_B_file_1)

    def setUp(self):
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.output_folder = self.data_folder + "/dummy_model"

        list_file = self.data_folder + "/in_out_test_list.toml"
        config_file = None

        self.orga = Organizer(self.output_folder, list_file, config_file)
        self.batchsize = 2
        self.orga.cfg.batchsize = self.batchsize
        self.orga.cfg.label_modifier = label_modifier

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.train_A_file_1["path"])
        os.remove(cls.train_B_file_1["path"])
        os.rmdir(cls.temp_dir)

    def test_batch(self):
        filepaths = self.filepaths_file_1
        gene = hdf5_batch_generator(self.orga, filepaths)

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2],
            "input_B": self.train_B_file_1_ctnt[0][:2],
        }

        target_ys_batch_1 = label_modifier(self.train_A_file_1_ctnt[1][:2])

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:],
            "input_B": self.train_B_file_1_ctnt[0][2:],
        }

        target_ys_batch_2 = label_modifier(self.train_A_file_1_ctnt[1][2:])

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

        # appended batches
        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

        with self.assertRaises(StopIteration):
            next(gene)


class TestFunctions(TestCase):
    def test_get_datasets(self):
        mc_info = "mc_info gets simply passed forward"
        y_true = {
            "out_A": 1,
            "out_B": 2,
        }
        y_pred = {
            "out_pred_A": 3,
            "out_pred_B": 4,
        }
        target = {
            "mc_info": mc_info,
            "label_out_A": 1,
            "label_out_B": 2,
            "pred_out_pred_A": 3,
            "pred_out_pred_B": 4,
        }
        datasets = get_datasets(mc_info, y_true, y_pred)
        self.assertDictEqual(datasets, target)

    def test_get_learning_rate_float(self):
        user_lr = 0.1
        no_train_files = 3

        for fileno in range(no_train_files):
            lr = get_learning_rate((1, fileno), user_lr, no_train_files)
            self.assertEqual(lr, user_lr)

        for fileno in range(no_train_files):
            lr = get_learning_rate((6, fileno), user_lr, no_train_files)
            self.assertEqual(lr, user_lr)

    def test_get_learning_rate_tuple(self):
        user_lr = (0.1, 0.2)
        no_train_files = 3

        rates_epoch_1 = [0.1, 0.08, 0.064]
        for fileno in range(no_train_files):
            lr = get_learning_rate((1, fileno+1), user_lr, no_train_files)
            self.assertAlmostEqual(lr, rates_epoch_1[fileno])

        rates_epoch_2 = [0.0512, 0.04096, 0.032768]
        for fileno in range(no_train_files):
            lr = get_learning_rate((2, fileno + 1), user_lr, no_train_files)
            self.assertAlmostEqual(lr, rates_epoch_2[fileno])

    def test_get_learning_rate_function(self):
        def get_lr(epoch, filenos):
            return epoch+filenos

        user_lr = get_lr
        no_train_files = 3

        rates_epoch_1 = [2, 3, 4]
        for fileno in range(no_train_files):
            lr = get_learning_rate((1, fileno+1), user_lr, no_train_files)
            self.assertAlmostEqual(lr, rates_epoch_1[fileno])

        rates_epoch_2 = [3, 4, 5]
        for fileno in range(no_train_files):
            lr = get_learning_rate((2, fileno + 1), user_lr, no_train_files)
            self.assertAlmostEqual(lr, rates_epoch_2[fileno])


def save_dummy_h5py(path, shape, size):
    """
    xs[0] is np.zeros, xs[1] is np.ones, etc
    ys[0] is ndarray with 0.5s, ys[1] with 1.5s etc
    """
    xs = np.ones((size,) + shape)
    for sample_no in range(len(xs)):
        xs[sample_no, ...] = sample_no

    dtypes = [('mc_A', '<f8'), ('mc_B', '<f8'), ]
    ys = np.ones((size, 2))
    for sample_no in range(len(xs)):
        ys[sample_no, ...] = sample_no + 0.5

    ys = ys.ravel().view(dtype=dtypes)

    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('x', data=xs, dtype='<f8')
        h5f.create_dataset('y', data=ys, dtype=dtypes)

    return xs, ys


def label_modifier(y_values):
    ys = dict()
    for name in y_values.dtype.names:
        ys[name] = y_values[name]
    return ys


def assert_dict_arrays_equal(a, b):
    for key, value in a.items():
        assert key in b
        np.testing.assert_array_almost_equal(a[key], b[key])
