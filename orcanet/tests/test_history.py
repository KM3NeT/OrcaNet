from unittest import TestCase
import os
import numpy as np
from orcanet.history import HistoryHandler


class TestHistoryHandler(TestCase):
    """
    Test the in out Handler on 2 dummy summary files
    (summary.txt, summary_2.txt).
    """
    def setUp(self):
        self.output_folder = os.path.join(os.path.dirname(__file__),
                                          "data", "dummy_model")
        self.history = HistoryHandler(self.output_folder)

        self.summary_filename_2 = "summary_2.txt"

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
        print(train_data)
        target = np.array(
            [(250., 0.00018755, 0.487258, 0.756687),
             (500., 0.00056265, 0.291661, 0.874188),
             (250., 0.0350691, 0.0589843, 0.977938),
             (500., 0.0354442, 0.0554589, 0.980375)],
            dtype=[('Batch', '<f8'), ('Batch_float', '<f8'),
                   ('loss', '<f8'), ('acc', '<f8')])
        assert_equal_struc_array(train_data, target)

    def test_get_column_names(self):
        column_names = self.history.get_column_names()
        target = ('Epoch', 'LR', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
        self.assertSequenceEqual(column_names, target)

    def test_get_state(self):
        self.history.summary_filename = self.summary_filename_2

        state = self.history.get_state()
        print(state)
        target = [
            {'epoch': 0.1, 'is_trained': True, 'is_validated': True},
            {'epoch': 0.2, 'is_trained': False, 'is_validated': True},
            {'epoch': 0.3, 'is_trained': True, 'is_validated': False},
            {'epoch': 0.4, 'is_trained': False, 'is_validated': False},
        ]
        self.assertSequenceEqual(state, target)

    def test_plot_metric_unknown_metric(self):
        with self.assertRaises(ValueError):
            self.history.plot_metric("test")
    """
    @patch('orcanet.history.plot_history')
    def test_plot_metric_loss(self, mock_plot_history):
        def plot_history(train_data, val_data, **kwargs):
            return train_data, val_data, kwargs
        mock_plot_history.side_effect = plot_history

        value_train, value_val, value_kwargs = self.history.plot_metric("loss")

        target_train = [
            np.array([0.000187551, 0.000562653, 0.0350691, 0.0354442]),
            np.array([0.487258, 0.291661, 0.0589843, 0.0554589]),
        ]

        target_val = [
            np.array([0.03488, 0.06971]),
            np.array([0.05562, np.nan]),
        ]
        target_kwargs = {'y_label': 'loss'}

        np.testing.assert_array_almost_equal(target_train, value_train)
        np.testing.assert_array_almost_equal(target_val, value_val)
        self.assertDictEqual(target_kwargs, value_kwargs)

    @patch('orcanet.history.plot_history')
    def test_plot_metric_acc(self, mock_plot_history):
        def plot_history(train_data, val_data, **kwargs):
            return train_data, val_data, kwargs
        mock_plot_history.side_effect = plot_history

        value_train, value_val, value_kwargs = self.history.plot_metric("acc")

        target_train = [
            np.array([0.000187551, 0.000562653, 0.0350691, 0.0354442]),
            np.array([0.756687, 0.874188, 0.977938, 0.980375]),
        ]

        target_val = [
            np.array([0.03488, 0.06971]),
            np.array([0.9808, np.nan]),
        ]
        target_kwargs = {'y_label': 'acc'}

        np.testing.assert_array_almost_equal(target_train, value_train)
        np.testing.assert_array_almost_equal(target_val, value_val)
        self.assertDictEqual(target_kwargs, value_kwargs)

    @patch('orcanet.history.plot_history')
    def test_plot_lr(self, mock_plot_history):
        def plot_history(train_data, val_data, **kwargs):
            return train_data, val_data, kwargs
        mock_plot_history.side_effect = plot_history

        value_train, value_val, value_kwargs = self.history.plot_lr()

        target_train = None

        target_val = [
            np.array([0.03488, 0.06971]),
            np.array([0.005, 0.00465])
        ]
        target_kwargs = {
            'y_label': 'Learning rate',
            'legend': False
        }

        self.assertEqual(target_train, value_train)
        np.testing.assert_array_almost_equal(target_val, value_val)
        self.assertDictEqual(target_kwargs, value_kwargs)
    """
    def test_get_best_epoch_info(self):
        self.history.summary_filename = self.summary_filename_2
        value = self.history.get_best_epoch_info()
        target = np.array(
            [(0.1, 0.11, 0.12, 0.13, 0.14, 0.15)],
            dtype=[('Epoch', '<f8'), ('LR', '<f8'), ('train_loss', '<f8'),
                   ('val_loss', '<f8'), ('train_acc', '<f8'), ('val_acc', '<f8')])
        np.testing.assert_array_equal(target, value)

    def test_get_best_epoch_fileno(self):
        self.history.summary_filename = self.summary_filename_2
        value = self.history.get_best_epoch_fileno()
        target = (1, 1)
        self.assertEqual(target, value)


def assert_equal_struc_array(a, b):
    """  np.testing.assert_array_equal does not work for arrays containing nans...
     so test individual instead. """
    np.testing.assert_array_equal(a.dtype, b.dtype)
    for name in a.dtype.names:
        np.testing.assert_almost_equal(a[name], b[name])
