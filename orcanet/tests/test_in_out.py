from unittest import TestCase
import os
import numpy as np

from orcanet.in_out import HistoryHandler


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
