from unittest import TestCase
import os
import numpy as np

from orcanet.core import Configuration
from orcanet.in_out import HistoryHandler, IOHandler


class TestIOHandlerNoFiles(TestCase):
    """ For io test that dont require a h5 file. """
    def setUp(self):
        self.temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.output_folder = self.data_folder + "/dummy_model"

        list_file = self.data_folder + "/in_out_test_list.toml"
        config_file = None

        cfg = Configuration(self.output_folder, list_file, config_file)
        self.io = IOHandler(cfg)

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

    def test_get_epoch_float(self):
        value = self.io.get_epoch_float(1, 2)
        target = 3
        self.assertEqual(value, target)


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
