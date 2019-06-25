from unittest import TestCase
import numpy as np

from orcanet.utilities.visualization import get_ylims, get_epoch_xticks, sort_metrics


class TestFunctions(TestCase):
    def setUp(self):
        train_epoch_data = np.linspace(0, 2, num=100)
        train_y_data = np.linspace(1, 2, num=100)

        val_epoch_data = np.linspace(0, 2, num=6)
        val_y_data = np.linspace(2, 3, num=6)

        self.train_data = (train_epoch_data, train_y_data)
        self.val_data = (val_epoch_data, val_y_data)

    def test_get_ylims_frac_25(self):
        y_lims = get_ylims(self.train_data[1], self.val_data[1])
        target = (0.5, 3.5)
        self.assertSequenceEqual(y_lims, target)

    def test_get_ylims_frac_0(self):
        y_lims = get_ylims(self.train_data[1], self.val_data[1], fraction=0)
        target = (1, 3)
        self.assertSequenceEqual(y_lims, target)

    def test_get_ylims_no_val(self):
        y_lims = get_ylims(self.train_data[1], None)
        target = (0.75, 2.25)
        self.assertSequenceEqual(y_lims, target)

    def test_get_ylims_no_train(self):
        y_lims = get_ylims(None, self.val_data[1])
        target = (1.75, 3.25)
        self.assertSequenceEqual(y_lims, target)

    def test_get_ylims_one_point_val(self):
        val_data = np.array([[1., ], [3., ]])
        y_lims = get_ylims(None, val_data[1])
        target = (2.925, 3.075)
        self.assertSequenceEqual(y_lims, target)

    def test_get_epoch_xticks(self):
        x_points = np.concatenate((self.train_data[0], self.val_data[0]))
        ticks = get_epoch_xticks(x_points)
        target = np.array([0, 1, 2])
        np.testing.assert_array_equal(ticks, target)

    def test_get_epoch_xticks_large(self):
        train_data = ([0.9, 21.9], [1, 2])
        ticks = get_epoch_xticks(train_data[0])
        target = np.arange(0, 23, 2)
        np.testing.assert_array_equal(ticks, target)

    def test_sort_metrics(self):
        metrics = ['e_loss', 'loss', 'e_err_loss', 'dx_err_loss']
        value = sort_metrics(metrics)
        target = ['e_loss', 'e_err_loss', 'loss', 'dx_err_loss']
        self.assertSequenceEqual(value, target)

    def test_sort_metrics_err_first(self):
        metrics = ['e_err_loss', 'dx_err_loss', 'e_loss', 'loss', ]
        value = sort_metrics(metrics)
        target = ['dx_err_loss', 'e_loss', 'e_err_loss', 'loss']
        self.assertSequenceEqual(value, target)

    def test_sort_metrics_no_err(self):
        metrics = ['e_loss', 'loss', 'blub']
        value = sort_metrics(metrics)
        target = ['e_loss', 'loss', 'blub']
        self.assertSequenceEqual(value, target)

    def test_sort_metrics_only_err(self):
        metrics = ['e_err_loss', 'dx_err_loss']
        value = sort_metrics(metrics)
        target = ['e_err_loss', 'dx_err_loss']
        self.assertSequenceEqual(value, target)

