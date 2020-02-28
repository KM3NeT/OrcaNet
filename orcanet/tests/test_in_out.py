from unittest import TestCase
from unittest.mock import MagicMock
import os
import h5py
import numpy as np
import shutil
from pathlib import Path
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

from orcanet.core import Configuration
from orcanet.in_out import IOHandler, split_name_of_predfile


class TestIOHandler(TestCase):
    @classmethod
    def setUpClass(cls):
        # super(TestIOHandler, cls).setUpClass()
        cls.temp_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "test_in_out")
        cls.pred_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "test_in_out", "predictions")
        os.mkdir(cls.temp_dir)
        os.mkdir(cls.pred_dir)

        # make dummy pred files
        Path(cls.pred_dir + '/pred_model_epoch_2_file_2_on_listname_val_file_1.h5').touch()
        Path(cls.pred_dir + '/pred_model_epoch_2_file_2_on_listname_val_file_2.h5').touch()

        cls.pred_filepaths = [cls.pred_dir + '/pred_model_epoch_2_file_2_on_listname_val_file_1.h5',
                              cls.pred_dir + '/pred_model_epoch_2_file_2_on_listname_val_file_2.h5']

        cls.init_dir = os.getcwd()
        os.chdir(cls.temp_dir)
        # make some dummy data
        cls.n_bins = {'input_A': (2, 3), 'input_B': (2, 3)}
        cls.train_sizes = [30, 50]
        cls.val_sizes = [40, 60]
        cls.file_names = (
            "/input_A_train_1.h5",
            "/input_A_train_2.h5",
            "/input_B_train_1.h5",
            "/input_B_train_2.h5",
            "/input_A_val_1.h5",
            "/input_A_val_2.h5",
            "/input_B_val_1.h5",
            "/input_B_val_2.h5",
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
        cls.val_A_file_2 = {
            "path": cls.temp_dir + cls.file_names[5],
            "shape": cls.n_bins["input_A"],
            "value_xs": 3.1,
            "value_ys": 3.2,
            "size": cls.val_sizes[0],
        }
        cls.val_B_file_1 = {
            "path": cls.temp_dir + cls.file_names[6],
            "shape": cls.n_bins["input_B"],
            "value_xs": 4.1,
            "value_ys": 4.2,
            "size": cls.val_sizes[1],
        }
        cls.val_B_file_2 = {
            "path": cls.temp_dir + cls.file_names[7],
            "shape": cls.n_bins["input_B"],
            "value_xs": 4.1,
            "value_ys": 4.2,
            "size": cls.val_sizes[1],
        }
        cls.train_A_file_1_ctnt = save_dummy_h5py(**cls.train_A_file_1)
        cls.train_A_file_2_ctnt = save_dummy_h5py(**cls.train_A_file_2)
        cls.train_B_file_1_ctnt = save_dummy_h5py(**cls.train_B_file_1)
        cls.train_B_file_2_ctnt = save_dummy_h5py(**cls.train_B_file_2)
        cls.val_A_file_1_ctnt = save_dummy_h5py(**cls.val_A_file_1)
        cls.val_A_file_2_ctnt = save_dummy_h5py(**cls.val_A_file_2)
        cls.val_B_file_1_ctnt = save_dummy_h5py(**cls.val_B_file_1)
        cls.val_B_file_2_ctnt = save_dummy_h5py(**cls.val_B_file_2)

    def setUp(self):
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.output_folder = self.data_folder + "/dummy_model"

        list_file = self.data_folder + "/in_out_test_list.toml"
        config_file = None

        cfg = Configuration(self.output_folder, list_file, config_file)
        self.batchsize = 3
        cfg.batchsize = self.batchsize
        self.io = IOHandler(cfg)

        # mock get_subfolder, but only in case of predictions argument
        original_get_subfolder = self.io.get_subfolder
        mocked_result = self.pred_dir

        def side_effect(key, create=False):
            if key == 'predictions':
                return mocked_result
            else:
                return original_get_subfolder(key, create)

        self.io.get_subfolder = MagicMock(side_effect=side_effect)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.train_A_file_1["path"])
        os.remove(cls.train_A_file_2["path"])
        os.remove(cls.train_B_file_1["path"])
        os.remove(cls.train_B_file_2["path"])
        os.remove(cls.val_A_file_1["path"])
        os.remove(cls.val_A_file_2["path"])
        os.remove(cls.val_B_file_1["path"])
        os.remove(cls.val_B_file_2["path"])

        os.chdir(cls.init_dir)
        shutil.rmtree(cls.temp_dir)

    def test_copy_to_ssd(self):
        self.io.cfg.use_scratch_ssd = True
        # make temporary directory
        temp_temp_dir = self.temp_dir + "/scratch"
        os.mkdir(temp_temp_dir)

        # change env variable TMPDIR to this dir (TMPDIR not defined in gitrunner)
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
                 "input_A": (scratch_dir + self.file_names[4], scratch_dir + self.file_names[5], ),
                 "input_B": (scratch_dir + self.file_names[6], scratch_dir + self.file_names[7], ),
            }

            value = self.io.get_local_files("train")
            self.assertDictEqual(target_dirs_train, value)

            value = self.io.get_local_files("val")
            self.assertDictEqual(target_dirs_val, value)

        finally:
            # reset the env variable
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

        def sample_modifier(info_blob):
            x_values = info_blob["x_values"]
            return {'input_A': x_values["input_A"], 'input_B': x_values["input_B"]}

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

        def sample_modifier(info_blob):
            x_values = info_blob["x_values"]
            return {'input_A': x_values["input_A"]}

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

        def sample_modifier(info_blob):
            x_values = info_blob["x_values"]
            return {'input_A': x_values["input_A"], }
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
        assert_dict_arrays_equal(value["x_values"], target)

    def test_get_batch_ys(self):
        value = self.io.get_batch()
        target = {
            "input_A": self.train_A_file_1_ctnt[1][:self.batchsize],
            "input_B": self.train_B_file_1_ctnt[1][:self.batchsize],
        }
        assert_equal_struc_array(value["y_values"], target["input_A"])

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
        value = self.io.get_previous_epoch((1, 2))
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

    def test_get_pred_files_list(self):
        value = self.io.get_pred_files_list()
        target = self.pred_filepaths
        self.assertEqual(value, target)

    def test_get_pred_files_list_epoch_given(self):
        value = self.io.get_pred_files_list(epoch=2)
        target = self.pred_filepaths
        self.assertEqual(value, target)

    def test_get_pred_files_list_fileno_given(self):
        value = self.io.get_pred_files_list(fileno=2)
        target = self.pred_filepaths
        self.assertEqual(value, target)

    def test_get_pred_files_list_epoch_fileno_given(self):
        value = self.io.get_pred_files_list(epoch=2, fileno=2)
        target = self.pred_filepaths
        self.assertEqual(value, target)

    def test_get_pred_files_list_no_files(self):
        value = self.io.get_pred_files_list(epoch=3)
        target = []
        self.assertEqual(value, target)

    def test_get_latest_prediction_file_no(self):
        value = self.io.get_latest_prediction_file_no(2, 2)
        target = 2
        self.assertEqual(value, target)

    def test_get_pred_path(self):
        value = self.io.get_pred_path(1, 2, 3)
        target = self.io.get_subfolder("predictions") + '/pred_model_epoch_1_file_2_on_in_out_test_list_val_file_3.h5'
        self.assertEqual(value, target)

    def test_get_cumulative_number_of_rows(self):
        h5_file_list = [self.train_A_file_1["path"], self.train_A_file_2["path"]]
        value = self.io.get_cumulative_number_of_rows(h5_file_list)
        target = [0, self.train_sizes[0], self.train_sizes[0] + self.train_sizes[1]]
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
            'input_A': ('input_A_val_1.h5', 'input_A_val_2.h5',),
            'input_B': ('input_B_val_1.h5', 'input_B_val_2.h5',)
        }
        self.assertDictEqual(value, target)

    def test_get_no_of_files_train(self):
        value = self.io.get_no_of_files("train")
        target = 2
        self.assertEqual(value, target)

    def test_get_no_of_files_val(self):
        value = self.io.get_no_of_files("val")
        target = 2
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


class TestIOHandlerLR(TestCase):
    """
    Test the learning rate method of the io handler. The csv part
    will generate files in the temp dir.
    """
    def setUp(self):
        self.temp_dir = os.path.join(
            os.path.dirname(__file__), ".temp", "TestIOHandlerLR")
        os.makedirs(self.temp_dir)

        cfg = Configuration(self.temp_dir, None, None)
        self.io = IOHandler(cfg)
        self.get_learning_rate = self.io.get_learning_rate

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_learning_rate_float(self):
        user_lr = 0.1
        no_train_files = 3

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        for fileno in range(no_train_files):
            lr = self.get_learning_rate((1, fileno))
            self.assertEqual(lr, user_lr)

        for fileno in range(no_train_files):
            lr = self.get_learning_rate((6, fileno))
            self.assertEqual(lr, user_lr)

    def test_get_learning_rate_tuple(self):
        user_lr = (0.1, 0.2)
        no_train_files = 3

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        rates_epoch_1 = [0.1, 0.08, 0.064]
        for fileno in range(no_train_files):
            lr = self.get_learning_rate((1, fileno+1))
            self.assertAlmostEqual(lr, rates_epoch_1[fileno])

        rates_epoch_2 = [0.0512, 0.04096, 0.032768]
        for fileno in range(no_train_files):
            lr = self.get_learning_rate((2, fileno + 1))
            self.assertAlmostEqual(lr, rates_epoch_2[fileno])

    def test_get_learning_rate_function(self):
        def get_lr(epoch, filenos):
            return epoch+filenos

        user_lr = get_lr
        no_train_files = 3

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        rates_epoch_1 = [2, 3, 4]
        for fileno in range(no_train_files):
            lr = self.get_learning_rate((1, fileno+1))
            self.assertAlmostEqual(lr, rates_epoch_1[fileno])

        rates_epoch_2 = [3, 4, 5]
        for fileno in range(no_train_files):
            lr = self.get_learning_rate((2, fileno + 1))
            self.assertAlmostEqual(lr, rates_epoch_2[fileno])

    def test_get_learning_rate_other(self):
        user_lr = print
        no_train_files = 3

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        with self.assertRaises(TypeError):
            self.get_learning_rate((1, 1))

    def test_get_learning_rate_read_csv_file(self):
        user_lr = "lr.csv"
        no_train_files = "not used"

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        csv_content = (
            "1  2   0.1\n"
            "1  3   0.2\n"
            "2  2   0.3\n"
        )

        with open(os.path.join(self.temp_dir, "lr.csv"), "w") as f:
            f.write(csv_content)

        with self.assertRaises(ValueError):
            self.get_learning_rate((1, 1))

        tests = [
            {"inp": (1, 2), "target": 0.1},
            {"inp": (1, 3), "target": 0.2},
            {"inp": [1, 10], "target": 0.2},
            {"inp": (2, 1), "target": 0.2},
            {"inp": (2, 2), "target": 0.3},
            {"inp": (2, 3), "target": 0.3},
            {"inp": (25, 33), "target": 0.3},
        ]
        for test in tests:
            self.assertEqual(
                self.get_learning_rate(test["inp"]), test["target"])

    def test_get_learning_rate_read_csv_file_only_one_line(self):
        user_lr = "lr.csv"
        no_train_files = "not used"

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        csv_content = (
            "0  1   2\n"
        )

        with open(os.path.join(self.temp_dir, "lr.csv"), "w") as f:
            f.write(csv_content)

        tests = [
            {"inp": (0, 1), "target": 2},
        ]
        for test in tests:
            self.assertEqual(
                self.get_learning_rate(test["inp"]), test["target"])

    def test_get_learning_rate_read_csv_file_bad_format_too_short(self):
        user_lr = "lr.csv"
        no_train_files = "not used"

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        csv_content = (
            "0  1\n"
            "0  1\n"
        )

        with open(os.path.join(self.temp_dir, "lr.csv"), "w") as f:
            f.write(csv_content)

        with self.assertRaises(ValueError):
            self.get_learning_rate((1, 1))

    def test_get_learning_rate_read_csv_file_bad_format_too_long(self):
        user_lr = "lr.csv"
        no_train_files = "not used"

        self.io.cfg.learning_rate = user_lr
        self.io.get_no_of_files = MagicMock(return_value=no_train_files)

        csv_content = (
            "0  1   2   4\n"
            "0  1   2   3\n"
        )

        with open(os.path.join(self.temp_dir, "lr.csv"), "w") as f:
            f.write(csv_content)

        with self.assertRaises(ValueError):
            self.get_learning_rate((1, 1))


class TestFunctions(TestCase):
    def test_split_name_of_predfile(self):
        filename = "pred_model_epoch_1_file_2_on_list_val_file_3.h5"
        target = (1, 2, 3)

        self.assertSequenceEqual(split_name_of_predfile(filename), target)


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
        inputs[name] = layers.Input(shape, name=name)
    conc = layers.Concatenate()(list(inputs.values()))
    flat = layers.Flatten()(conc)

    outputs = {}
    for name, shape in output_shapes.items():
        outputs[name] = layers.Dense(shape, name=name)(flat)

    model = Model(list(inputs.values()), list(outputs.values()))
    return model


def get_dummy_label_modifier(output_names):
    def label_modifier(info_blob):
        y_values = info_blob["y_values"]
        particle = y_values["mc_A"]
        y_true = dict()
        for output_name in output_names:
            y_true[output_name] = particle
        return y_true
    return label_modifier


def assert_dict_arrays_equal(a, b):
    for key, value in a.items():
        assert key in b
        np.testing.assert_array_almost_equal(a[key], b[key])

