from unittest import TestCase
import os
import h5py
import numpy as np

from orcanet.core import Configuration
from orcanet.in_out import HistoryHandler, IOHandler


class TestIOHandler(TestCase):
    @classmethod
    def setUpClass(cls):
        # super(TestIOHandler, cls).setUpClass()
        cls.temp_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "test_in_out")
        os.mkdir(cls.temp_dir)
        cls.init_dir = os.getcwd()
        os.chdir(cls.temp_dir)
        # make some dummy data
        cls.n_bins = {'input_A': (2, 3), 'input_B': (2, 3)}
        cls.train_sizes = [30, 50]
        cls.train_A_file_1 = {
            "path": cls.temp_dir + "/input_A_train_1.h5",
            "shape": cls.n_bins["input_A"],
            "value_xs": 1.1,
            "value_ys": 1.2,
            "size": cls.train_sizes[0],
        }
        cls.train_A_file_2 = {
            "path": cls.temp_dir + "/input_A_train_2.h5",
            "shape": cls.n_bins["input_A"],
            "value_xs": 1.3,
            "value_ys": 1.4,
            "size": cls.train_sizes[1],
        }
        cls.train_B_file_1 = {
            "path": cls.temp_dir + "/input_B_train_1.h5",
            "shape": cls.n_bins["input_B"],
            "value_xs": 2.1,
            "value_ys": 2.2,
            "size": cls.train_sizes[0],
        }
        cls.train_B_file_2 = {
            "path": cls.temp_dir + "/input_B_train_2.h5",
            "shape": cls.n_bins["input_B"],
            "value_xs": 2.3,
            "value_ys": 2.4,
            "size": cls.train_sizes[1],
        }
        cls.train_A_file_1_ctnt = save_dummy_h5py(**cls.train_A_file_1)
        cls.train_A_file_2_ctnt = save_dummy_h5py(**cls.train_A_file_2)
        cls.train_B_file_1_ctnt = save_dummy_h5py(**cls.train_B_file_1)
        cls.train_B_file_2_ctnt = save_dummy_h5py(**cls.train_B_file_2)

    def setUp(self):
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.output_folder = self.data_folder + "/dummy_model"

        list_file = self.data_folder + "/in_out_test_list.toml"
        config_file = None

        cfg = Configuration(self.output_folder, list_file, config_file)
        self.batchsize = 3
        cfg.batchsize = self.batchsize
        self.io = IOHandler(cfg)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.train_A_file_1["path"])
        os.remove(cls.train_A_file_2["path"])
        os.remove(cls.train_B_file_1["path"])
        os.remove(cls.train_B_file_2["path"])

        os.chdir(cls.init_dir)
        os.rmdir(cls.temp_dir)

    def test_get_n_bins(self):
        value = self.io.get_n_bins()
        self.assertSequenceEqual(value, self.n_bins)

    def test_get_file_sizes_train(self):
        value = self.io.get_file_sizes("train")
        self.assertSequenceEqual(value, self.train_sizes)

    def test_get_batch_xs(self):
        value = self.io.get_batch()
        target = {
            "input_A": self.train_A_file_1_ctnt[0][:self.batchsize],
            "input_B": self.train_B_file_1_ctnt[0][:self.batchsize],
        }
        assert_dict_arrays_equal(value[0], target)

    def test_get_batch_ys(self):
        value = self.io.get_batch()
        target = {
            "input_A": self.train_A_file_1_ctnt[1][:self.batchsize],
            "input_B": self.train_B_file_1_ctnt[1][:self.batchsize],
        }
        assert_equal_struc_array(value[1], target["input_A"])

    def test_get_latest_epoch(self):
        value = self.io.get_latest_epoch()
        target = (2, 1)
        self.assertSequenceEqual(value, target)

    def test_get_latest_epoch_epoch_1(self):
        value = self.io.get_latest_epoch(epoch=1)
        target = (1, 2)
        self.assertSequenceEqual(value, target)

    def test_get_latest_epoch_no_files_but_epoch_given(self):
        with self.assertRaises(ValueError):
            self.io.get_latest_epoch(epoch=3)

    def test_get_next_epoch_none(self):
        value = self.io.get_next_epoch(None)
        target = (1, 1)
        self.assertSequenceEqual(value, target)

    def test_get_next_epoch_1_1(self):
        value = self.io.get_next_epoch((1, 1))
        target = (1, 2)
        self.assertSequenceEqual(value, target)

    def test_get_next_epoch_1_2(self):
        value = self.io.get_next_epoch((1, 2))
        target = (2, 1)
        self.assertSequenceEqual(value, target)

    def test_get_previous_epoch_2_1(self):
        value = self.io.get_previous_epoch((2, 1))
        target = (1, 2)
        self.assertSequenceEqual(value, target)

    def test_get_previous_epoch_1_2(self):
        value = self.io.get_previous_epoch((1,2))
        target = (1, 1)
        self.assertSequenceEqual(value, target)

    def test_get_model_path(self):
        value = self.io.get_model_path(1, 1)
        target = self.output_folder + '/saved_models/model_epoch_1_file_1.h5'
        self.assertEqual(value, target)

    def test_get_model_path_latest(self):
        value = self.io.get_model_path(-1, -1)
        target = self.output_folder + '/saved_models/model_epoch_2_file_1.h5'
        self.assertEqual(value, target)

    def test_get_pred_path(self):
        value = self.io.get_pred_path(2, 1)
        target = self.output_folder + '/predictions/pred_model_epoch_2_file_1' \
                                      '_on_in_out_test_list_val_files.h5'
        self.assertEqual(value, target)

    def test_get_local_files_train(self):
        value = self.io.get_local_files("train")
        target = {
            'input_A': ('input_A_train_1.h5', 'input_A_train_2.h5'),
            'input_B': ('input_B_train_1.h5', 'input_B_train_2.h5'),
        }
        self.assertDictEqual(value, target)

    def test_get_local_files_val(self):
        value = self.io.get_local_files("val")
        target = {
            'input_A': ('input_A_val_1.h5', 'input_A_val_2.h5', 'input_A_val_3.h5'),
            'input_B': ('input_B_val_1.h5', 'input_B_val_2.h5', 'input_B_val_3.h5')
        }
        self.assertDictEqual(value, target)

    def test_get_no_of_files_train(self):
        value = self.io.get_no_of_files("train")
        target = 2
        self.assertEqual(value, target)

    def test_get_no_of_files_val(self):
        value = self.io.get_no_of_files("val")
        target = 3
        self.assertEqual(value, target)

    def test_yield_files_train(self):
        file_paths = self.io.yield_files("train")
        target = (
            {
                'input_A': 'input_A_train_1.h5',
                'input_B': 'input_B_train_1.h5',
            },
            {
                'input_A': 'input_A_train_2.h5',
                'input_B': 'input_B_train_2.h5',
            },
        )
        for i, value in enumerate(file_paths):
            self.assertDictEqual(value, target[i])

    def test_yield_files_val(self):
        file_paths = self.io.yield_files("val")
        target = (
            {
                'input_A': 'input_A_val_1.h5',
                'input_B': 'input_B_val_1.h5',
            },
            {
                'input_A': 'input_A_val_2.h5',
                'input_B': 'input_B_val_2.h5',
            },
            {
                'input_A': 'input_A_val_3.h5',
                'input_B': 'input_B_val_3.h5',
            },
        )
        for i, value in enumerate(file_paths):
            self.assertDictEqual(value, target[i])

    def test_get_file_train(self):
        value = self.io.get_file("train", 2)
        target = {
                'input_A': 'input_A_train_2.h5',
                'input_B': 'input_B_train_2.h5',
        }
        self.assertDictEqual(value, target)

    def test_get_file_val(self):
        value = self.io.get_file("val", 1)
        target = {
                'input_A': 'input_A_val_1.h5',
                'input_B': 'input_B_val_1.h5',
        }
        self.assertDictEqual(value, target)


class TestHistoryHandler(TestCase):
    def setUp(self):
        self.output_folder = os.path.join(os.path.dirname(__file__),
                                          "data", "dummy_model")
        self.summary_filename = os.path.join(self.output_folder, "summary.txt")
        self.summary_filename_2 = os.path.join(self.output_folder, "summary_2.txt")
        self.train_log_folder = os.path.join(self.output_folder, "train_log")

        self.history = HistoryHandler(self.summary_filename,
                                      self.train_log_folder)

    def test_get_metrics(self):
        metrics = self.history.get_metrics()
        target = ["loss", "acc"]
        np.testing.assert_array_equal(metrics, target)

    def test_get_summary_data(self):
        summary_data = self.history.get_summary_data()
        target = np.array(
            [(0.03488, 0.005, 0.07776, 0.05562, 0.971, 0.9808),
             (0.06971, 0.00465, 0.05034, np.nan, 0.9822, np.nan)],
            dtype=[('Epoch', '<f8'), ('LR', '<f8'), ('train_loss', '<f8'),
                   ('val_loss', '<f8'), ('train_acc', '<f8'), ('val_acc', '<f8')])
        assert_equal_struc_array(summary_data, target)

    def test_get_train_data(self):
        train_data = self.history.get_train_data()
        target = np.array(
            [(250., 0.00018755, 0.48725819, 0.7566875),
             (500., 0.00056265, 0.29166106, 0.8741875),
             (250., 0.03506906, 0.05898428, 0.9779375),
             (500., 0.03544416, 0.05545885, 0.980375)],
            dtype=[('Batch', '<f8'), ('Batch_float', '<f8'),
                   ('loss', '<f8'), ('acc', '<f8')])
        assert_equal_struc_array(train_data, target)

    def test_get_column_names(self):
        column_names = self.history.get_column_names()
        target = ('Epoch', 'LR', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
        self.assertSequenceEqual(column_names, target)

    def test_get_state(self):
        self.history = HistoryHandler(self.summary_filename_2,
                                      self.train_log_folder)
        state = self.history.get_state()
        print(state)
        target = [
            {'epoch': 0.1, 'is_trained': True, 'is_validated': True},
            {'epoch': 0.2, 'is_trained': False, 'is_validated': True},
            {'epoch': 0.3, 'is_trained': True, 'is_validated': False},
            {'epoch': 0.4, 'is_trained': False, 'is_validated': False},
        ]
        self.assertSequenceEqual(state, target)


def assert_equal_struc_array(a, b):
    """  np.testing.assert_array_equal does not work for arrays containing nans...
     so test individual instead. """
    np.testing.assert_array_equal(a.dtype, b.dtype)
    for name in a.dtype.names:
        np.testing.assert_almost_equal(a[name], b[name])


def save_dummy_h5py(path, shape, value_xs=1., value_ys=1., size=50):
    xs = np.ones((size,) + shape) * value_xs

    dtypes = [('mc_A', '<f8'), ('mc_B', '<f8'), ]
    ys = (np.ones((size, 2))*value_ys).ravel().view(dtype=dtypes)

    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('x', data=xs, dtype='<f8')
        h5f.create_dataset('y', data=ys, dtype=dtypes)

    return xs, ys


def assert_dict_arrays_equal(a, b):
    for key, value in a.items():
        assert key in b
        np.testing.assert_array_almost_equal(a[key], b[key])

