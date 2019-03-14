from unittest import TestCase
from unittest.mock import patch
import os
import h5py
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Flatten
import shutil

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
        cls.val_sizes = [40, ]
        cls.file_names = (
            "/input_A_train_1.h5",
            "/input_A_train_2.h5",
            "/input_B_train_1.h5",
            "/input_B_train_2.h5",
            "/input_A_val_1.h5",
            "/input_B_val_1.h5",
        )
        cls.train_A_file_1 = {
            "path": cls.temp_dir + cls.file_names[0],
            "shape": cls.n_bins["input_A"],
            "value_xs": 1.1,
            "value_ys": 1.2,
            "size": cls.train_sizes[0],
        }
        cls.train_A_file_2 = {
            "path": cls.temp_dir + cls.file_names[1],
            "shape": cls.n_bins["input_A"],
            "value_xs": 1.3,
            "value_ys": 1.4,
            "size": cls.train_sizes[1],
        }
        cls.train_B_file_1 = {
            "path": cls.temp_dir + cls.file_names[2],
            "shape": cls.n_bins["input_B"],
            "value_xs": 2.1,
            "value_ys": 2.2,
            "size": cls.train_sizes[0],
        }
        cls.train_B_file_2 = {
            "path": cls.temp_dir + cls.file_names[3],
            "shape": cls.n_bins["input_B"],
            "value_xs": 2.3,
            "value_ys": 2.4,
            "size": cls.train_sizes[1],
        }
        cls.val_A_file_1 = {
            "path": cls.temp_dir + cls.file_names[4],
            "shape": cls.n_bins["input_A"],
            "value_xs": 3.1,
            "value_ys": 3.2,
            "size": cls.val_sizes[0],
        }
        cls.val_B_file_1 = {
            "path": cls.temp_dir + cls.file_names[5],
            "shape": cls.n_bins["input_B"],
            "value_xs": 4.1,
            "value_ys": 4.2,
            "size": cls.val_sizes[0],
        }
        cls.train_A_file_1_ctnt = save_dummy_h5py(**cls.train_A_file_1)
        cls.train_A_file_2_ctnt = save_dummy_h5py(**cls.train_A_file_2)
        cls.train_B_file_1_ctnt = save_dummy_h5py(**cls.train_B_file_1)
        cls.train_B_file_2_ctnt = save_dummy_h5py(**cls.train_B_file_2)
        cls.val_A_file_1_ctnt = save_dummy_h5py(**cls.val_A_file_1)
        cls.val_B_file_1_ctnt = save_dummy_h5py(**cls.val_B_file_1)

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
        os.remove(cls.val_A_file_1["path"])
        os.remove(cls.val_B_file_1["path"])

        os.chdir(cls.init_dir)
        os.rmdir(cls.temp_dir)

    def test_use_local_node(self):
        temp_temp_dir = self.temp_dir + "/scratch"
        os.mkdir(temp_temp_dir)
        if "TMPDIR" in os.environ:
            tempdir_environ = os.environ["TMPDIR"]
        else:
            tempdir_environ = None
        try:
            os.environ["TMPDIR"] = temp_temp_dir
            scratch_dir = temp_temp_dir

            target_dirs_train = {
                 "input_A": (scratch_dir + self.file_names[0], scratch_dir + self.file_names[1]),
                 "input_B": (scratch_dir + self.file_names[2], scratch_dir + self.file_names[3]),
            }
            target_dirs_val = {
                 "input_A": (scratch_dir + self.file_names[4], ),
                 "input_B": (scratch_dir + self.file_names[5], ),
            }

            self.io.use_local_node()

            value = self.io.get_local_files("train")
            self.assertDictEqual(target_dirs_train, value)

            value = self.io.get_local_files("val")
            self.assertDictEqual(target_dirs_val, value)
        finally:
            if tempdir_environ is not None:
                os.environ["TMPDIR"] = tempdir_environ
            else:
                os.environ.pop("TMPDIR")

            shutil.rmtree(temp_temp_dir)

    def test_check_connections_no_sample(self):
        input_shapes = self.n_bins
        output_shapes = {
            "out_A": 1,
            "out_B": 1,
        }

        self.io.cfg.label_modifier = get_dummy_label_modifier(output_shapes.keys())
        model = build_dummy_model(input_shapes, output_shapes)

        self.io.check_connections(model)

    def test_check_connections_ok_sample(self):
        input_shapes = self.n_bins
        output_shapes = {
            "out_A": 1,
            "out_B": 1,
        }

        def sample_modifier(samples):
            return {'input_A': samples["input_A"], 'input_B': samples["input_B"]}

        self.io.cfg.label_modifier = get_dummy_label_modifier(
            output_shapes.keys())
        self.io.cfg.sample_modifier = sample_modifier

        model = build_dummy_model(input_shapes, output_shapes)
        self.io.check_connections(model)

    def test_check_connections_wrong_sample(self):
        input_shapes = self.n_bins
        output_shapes = {
            "out_A": 1,
            "out_B": 1,
        }

        def sample_modifier(samples):
            return {'input_A': samples["input_A"]}

        self.io.cfg.label_modifier = get_dummy_label_modifier(
            output_shapes.keys())
        self.io.cfg.sample_modifier = sample_modifier

        model = build_dummy_model(input_shapes, output_shapes)
        with self.assertRaises(ValueError):
            self.io.check_connections(model)

    def test_check_connections_wrong_label(self):
        input_shapes = self.n_bins
        output_shapes = {
            "out_A": 1,
            "out_B": 1,
        }

        self.io.cfg.label_modifier = get_dummy_label_modifier(["out_A"])
        model = build_dummy_model(input_shapes, output_shapes)

        with self.assertRaises(ValueError):
            self.io.check_connections(model)

    def test_check_connections_no_label(self):
        input_shapes = self.n_bins
        output_shapes = {
            "out_A": 1,
            "out_B": 1,
        }

        model = build_dummy_model(input_shapes, output_shapes)

        with self.assertRaises(ValueError):
            self.io.check_connections(model)

    def test_check_connections_auto_label(self):
        input_shapes = self.n_bins
        output_shapes = {
            "mc_A": 1,
            "mc_B": 1,
        }

        model = build_dummy_model(input_shapes, output_shapes)

        self.io.check_connections(model)

    def test_get_n_bins(self):
        value = self.io.get_n_bins()
        self.assertSequenceEqual(value, self.n_bins)

    def test_get_input_shape(self):
        value = self.io.get_input_shapes()
        self.assertSequenceEqual(value, self.n_bins)

        def sample_modifier(samples):
            return {'input_A': samples["input_A"], }
        self.io.cfg.sample_modifier = sample_modifier

        value = self.io.get_input_shapes()
        self.assertEqual(value, {"input_A": self.n_bins["input_A"]})

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

    def test_get_all_epochs(self):
        epochs = self.io.get_all_epochs()
        target = [
            (1, 1), (1, 2), (2, 1),
        ]
        self.assertSequenceEqual(epochs, target)

    def test_get_latest_epoch(self):
        value = self.io.get_latest_epoch()
        target = (2, 1)
        self.assertSequenceEqual(value, target)

    def test_get_latest_epoch_no_epochs(self):
        self.io.cfg.output_folder = "./missing/"
        value = self.io.get_latest_epoch()
        self.assertEqual(value, None)

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

    def test_get_model_path_local(self):
        value = self.io.get_model_path(1, 1, local=True)
        target = 'saved_models/model_epoch_1_file_1.h5'
        self.assertEqual(value, target)

    def test_get_model_path_latest(self):
        value = self.io.get_model_path(-1, -1)
        target = self.output_folder + '/saved_models/model_epoch_2_file_1.h5'
        self.assertEqual(value, target)

    def test_get_model_path_latest_invalid(self):
        with self.assertRaises(ValueError):
            self.io.get_model_path(1, -1)

        with self.assertRaises(ValueError):
            self.io.get_model_path(-1, 1)

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
            'input_A': ('input_A_val_1.h5',),
            'input_B': ('input_B_val_1.h5',)
        }
        self.assertDictEqual(value, target)

    def test_get_no_of_files_train(self):
        value = self.io.get_no_of_files("train")
        target = 2
        self.assertEqual(value, target)

    def test_get_no_of_files_val(self):
        value = self.io.get_no_of_files("val")
        target = 1
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

    def test_plot_metric_onknown_metric(self):
        with self.assertRaises(ValueError):
            self.history.plot_metric("test")

    @patch('orcanet.in_out.plot_history')
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

    @patch('orcanet.in_out.plot_history')
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

    @patch('orcanet.in_out.plot_history')
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


def build_dummy_model(input_shapes, output_shapes):
    """

    Parameters
    ----------
    input_shapes : dict
    output_shapes : dict

    Returns
    -------
    model : keras model

    """
    inputs = {}
    for name, shape in input_shapes.items():
        inputs[name] = Input(shape, name=name)
    conc = Concatenate()(list(inputs.values()))
    flat = Flatten()(conc)

    outputs = {}
    for name, shape in output_shapes.items():
        outputs[name] = Dense(shape, name=name)(flat)

    model = Model(list(inputs.values()), list(outputs.values()))
    return model


def get_dummy_label_modifier(output_names):
    def label_modifier(mc_info):
        particle = mc_info["mc_A"]
        y_true = dict()
        for output_name in output_names:
            y_true[output_name] = particle
        return y_true
    return label_modifier


def assert_dict_arrays_equal(a, b):
    for key, value in a.items():
        assert key in b
        np.testing.assert_array_almost_equal(a[key], b[key])

