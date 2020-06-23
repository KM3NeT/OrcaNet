from unittest import TestCase
from unittest.mock import MagicMock
import os
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import numpy as np
import shutil

from orcanet.logging import SummaryLogger, merge_arrays, BatchLogger, gen_line_str
from orcanet.core import Organizer


class TestSummaryLogger(TestCase):
    """
    Test writing of the SummaryLogger by generating a summary file into
    a temp directory. Will then check if its contents match the given target.

    """
    def setUp(self):
        # Size of train files. No actual files get generated (they are mocked).
        file_sizes = [100, 66]

        self.temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
        self.summary_file = os.path.join(self.temp_dir, "summary.txt")

        self.orga = Organizer(self.temp_dir)
        self.orga.io.get_file_sizes = MagicMock(return_value=file_sizes)
        model = build_test_model()
        self.smry = SummaryLogger(self.orga, model)
        # tf 2.2: train the model to set metric_names
        model.train_on_batch(np.ones((2, 1)), np.ones((2, 1)))

        self.metrics = model.metrics_names

    def _check_file(self, target_lines):
        verbose = False  # for debugging
        with open(self.summary_file) as file:
            for i, line in enumerate(file):
                if verbose:
                    print("\n", [line])
                    print([target_lines[i]])
                self.assertEqual(line, target_lines[i])

    def test_writing_summary_file_update_epoch_1(self):
        history_train_0 = dict(zip(self.metrics, [0.0, 0.5]))
        history_train_1 = dict(zip(self.metrics, [1.0, 1.5]))
        history_val = dict(zip(self.metrics, [2.0, 2.5]))

        target = [
            "Epoch       | LR          | train_loss  | val_loss    | train_mae   | val_mae    \n",
            "------------+-------------+-------------+-------------+-------------+------------\n",
            "0.60241     | 0.001       | 0           | n/a         | 0.5         | n/a        \n",
            "1           | 0.002       | 1           | n/a         | 1.5         | n/a        \n",
        ]
        filled_line = "1           | 0.002       | 1           | 2           | 1.5         | 2.5        \n"

        """
        target = [
            "Epoch       | LR          | train_loss  | val_loss    | train_mean_absolute_error | val_mean_absolute_error\n",
            "------------+-------------+-------------+-------------+---------------------------+------------------------\n",
            "0.60241     | 0.001       | 0           | n/a         | 0.5                       | n/a                    \n",
            "1           | 0.002       | 1           | n/a         | 1.5                       | n/a                    \n",
        ]
        filled_line = "1           | 0.002       | 1           | 2           | 1.5                       | 2.5                    \n"
        """

        epoch = (1, 1)
        lr = 0.001
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_train=history_train_0)
        self._check_file(target)

        epoch = (1, 2)
        lr = 0.002
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_train=history_train_1)
        self._check_file(target)

        target[-1] = filled_line
        lr = np.nan
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_val=history_val)
        self._check_file(target)

    def test_writing_summary_file_update_epoch_2(self):
        history_train_0 = dict(zip(self.metrics, [0.0, 0.5]))
        history_train_1 = dict(zip(self.metrics, [1.0, 1.5]))
        history_val = dict(zip(self.metrics, [2.0, 2.5]))

        target = [
            "Epoch       | LR          | train_loss  | val_loss    | train_mae   | val_mae    \n",
            "------------+-------------+-------------+-------------+-------------+------------\n",
            "1.60241     | 0.001       | 0           | n/a         | 0.5         | n/a        \n",
            "2           | 0.002       | 1           | n/a         | 1.5         | n/a        \n",
        ]
        filled_line = "1.60241     | 0.001       | 0           | 2           | 0.5         | 2.5        \n"

        """
        target = [
            "Epoch       | LR          | train_loss  | val_loss    | train_mean_absolute_error | val_mean_absolute_error\n",
            "------------+-------------+-------------+-------------+---------------------------+------------------------\n",
            "1.60241     | 0.001       | 0           | n/a         | 0.5                       | n/a                    \n",
            "2           | 0.002       | 1           | n/a         | 1.5                       | n/a                    \n",
        ]
        filled_line = "1.60241     | 0.001       | 0           | 2           | 0.5                       | 2.5                    \n"
        """

        epoch = (2, 1)
        lr = 0.001
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_train=history_train_0)
        self._check_file(target)

        target[2] = filled_line
        lr = "n/a"
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_val=history_val)
        self._check_file(target)

        epoch = (2, 2)
        lr = 0.002
        self.smry.write_line(self.orga.io.get_epoch_float(*epoch), lr,
                             history_train=history_train_1)
        self._check_file(target)

    def tearDown(self):
        os.remove(self.summary_file)


class TestLoggingUtil(TestCase):
    def test_merge_arrays(self):
        a = [1, 2, np.nan, np.nan, "n/a", np.nan]
        b = [np.nan, 2, np.nan, 3, 4, "n/a"]
        target = [1, 2, np.nan, 3, 4, "n/a"]
        merged = merge_arrays(a, b)
        self.assertSequenceEqual(merged, target)

    def test_merge_arrays_error(self):
        a = [1, 1]
        b = [2, 1]
        with self.assertRaises(ValueError):
            merge_arrays(a, b)

    def test_merge_arrays_exclude(self):
        a = [1, 1]
        b = [2, 1]
        target = [1, 1]
        merged = merge_arrays(a, b, 0)
        self.assertSequenceEqual(merged, target)


class TestGenLineStr(TestCase):
    def setUp(self):
        self.widths = [9, 9]
        self.long = [1.23456789e-9, 2.0]
        self.short = ["blub", 2]

        self.kwargs = {
            "seperator": " | ",
            "float_precision": 4,
            "minimum_cell_width": 9,
        }

    def test_ok(self):
        line_l, widths_l = gen_line_str(self.long, self.widths, **self.kwargs)
        line_s, widths_s = gen_line_str(self.short, self.widths, **self.kwargs)

        #          "1---5---9_|_1---5---9"
        target_s = "blub      | 2        "
        target_l = "1.235e-09 | 2        "
        self.assertEqual(line_l, target_l)
        self.assertEqual(line_s, target_s)
        self.assertSequenceEqual(self.widths, widths_l)
        self.assertSequenceEqual(self.widths, widths_s)


class TestBatchLogger(TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(
            os.path.dirname(__file__), ".temp", "batch_logger")

        self.model = build_test_model()
        samples_file1 = 40
        samples_file2 = 60
        self.data_file1 = get_test_data(samples_file1)
        self.data_file2 = get_test_data(samples_file2)

        self.orga = Organizer(self.temp_dir)
        self.orga.io.get_file_sizes = MagicMock(
            return_value=[samples_file1, samples_file2])
        self.orga.io.get_subfolder("train_log", create=True)

        self.orga.cfg.batchsize = 10
        self.orga.cfg.train_logger_display = 4

    def tearDown(self):
        """ Remove the .temp directory. """
        shutil.rmtree(self.temp_dir)

    def test_batch_logger_epoch_1_logfile_1(self):
        epoch = (1, 1)
        lines = self._make_and_get_lines(epoch)

        target_file1 = [
            'Batch       | Batch_float | loss        | mae        \n',
            '------------+-------------+-------------+------------\n',
            '4           | 0.2         | 0.25        | 0.5        \n',
        ]

        """
        target_file1 = [
            'Batch       | Batch_float | loss        | mean_absolute_error\n',
            '------------+-------------+-------------+--------------------\n',
            '4           | 0.2         | 0.25        | 0.5                \n',
        ]
        """

        for line_no in range(len(lines)):
            self.assertEqual(target_file1[line_no], lines[line_no])

    def test_batch_logger_epoch_1_logfile_2(self):
        epoch = (1, 2)
        lines = self._make_and_get_lines(epoch)

        target_file1 = [
            'Batch       | Batch_float | loss        | mae        \n',
            '------------+-------------+-------------+------------\n',
            '4           | 0.6         | 0.25        | 0.5        \n',
            '6           | 0.8         | 0.125       | 0.25       \n',
        ]

        """
        target_file1 = [
            'Batch       | Batch_float | loss        | mean_absolute_error\n',
            '------------+-------------+-------------+--------------------\n',
            '4           | 0.6         | 0.25        | 0.5                \n',
            '6           | 0.8         | 0.125       | 0.25               \n',
        ]
        """

        for line_no in range(len(lines)):
            self.assertEqual(target_file1[line_no], lines[line_no])

    def test_batch_logger_epoch_2_logfile_1(self):
        epoch = (2, 1)
        lines = self._make_and_get_lines(epoch)

        target_file1 = [
            'Batch       | Batch_float | loss        | mae        \n',
            '------------+-------------+-------------+------------\n',
            '4           | 1.2         | 0.25        | 0.5        \n',
        ]
        """
        target_file1 = [
            'Batch       | Batch_float | loss        | mean_absolute_error\n',
            '------------+-------------+-------------+--------------------\n',
            '4           | 1.2         | 0.25        | 0.5                \n',
        ]
        """

        for line_no in range(len(lines)):
            self.assertEqual(target_file1[line_no], lines[line_no])

    def _make_and_get_lines(self, epoch):
        """ Create some train log file with the batch logger. """
        batch_logger = BatchLogger(self.orga, epoch)

        if epoch[1] == 1:
            train_file = self.data_file1
        elif epoch[1] == 2:
            train_file = self.data_file2
        else:
            raise AssertionError(epoch)

        self.model.fit(train_file[0], train_file[1],
                       batch_size=self.orga.cfg.batchsize,
                       callbacks=[batch_logger], verbose=0)
        logfile_path = '{}/log_epoch_{}_file_{}.txt'.format(
            os.path.join(self.temp_dir, "train_log"),
            epoch[0], epoch[1])

        with open(logfile_path) as file:
            lines = file.readlines()
        return lines


def build_test_model():
    input_shape = (1,)
    inp = layers.Input(input_shape, name="inp")
    x = layers.Dense(1, kernel_initializer="Ones",
                     bias_initializer="Zeros", trainable=False)(inp)
    outp = layers.Dense(1, name="out", kernel_initializer="Ones",
                        bias_initializer="Zeros", trainable=False)(x)

    test_model = Model(inp, outp)
    test_model.compile("sgd", loss={"out": "mse"}, metrics={"out": "mae"})

    return test_model


def get_test_data(samples):
    """
    Make model input data.

    xs : [0.5, ]
    ys : [1, ]

    MSE: 1/4
    MAE: 1/2

    """
    xs = np.ones(samples,) * 0.5
    ys = np.ones((samples, 1))
    return xs, ys
