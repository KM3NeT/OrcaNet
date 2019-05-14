from unittest import TestCase
from unittest.mock import MagicMock
import os
import h5py
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Flatten
from keras.callbacks import LambdaCallback

from orcanet.core import Organizer
from orcanet.backend import hdf5_batch_generator, get_datasets, train_model, validate_model, make_model_prediction, weighted_average
from orcanet.utilities.nn_utilities import get_auto_label_modifier


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

    def test_batch_zero_center(self):
        filepaths = self.filepaths_file_1

        xs_mean = {name: np.ones(shape) * 0.5 for name, shape in self.n_bins.items()}

        self.orga.get_xs_mean = MagicMock(return_value=xs_mean)
        gene = hdf5_batch_generator(self.orga, filepaths, zero_center=True)

        target_xs_batch_1 = {
            "input_A": np.subtract(self.train_A_file_1_ctnt[0][:2], xs_mean["input_A"]),
            "input_B": np.subtract(self.train_B_file_1_ctnt[0][:2], xs_mean["input_B"]),
        }

        target_ys_batch_1 = label_modifier(self.train_A_file_1_ctnt[1][:2])

        target_xs_batch_2 = {
            "input_A": np.subtract(self.train_A_file_1_ctnt[0][2:], xs_mean["input_A"]),
            "input_B": np.subtract(self.train_B_file_1_ctnt[0][2:], xs_mean["input_B"]),
        }

        target_ys_batch_2 = label_modifier(self.train_A_file_1_ctnt[1][2:])

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

    def test_batch_sample_modifier(self):
        filepaths = self.filepaths_file_1

        def sample_modifier(xs_in):
            mod = {name: val*2 for name, val in xs_in.items()}
            return mod

        self.orga.cfg.sample_modifier = sample_modifier
        gene = hdf5_batch_generator(self.orga, filepaths)

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2]*2,
            "input_B": self.train_B_file_1_ctnt[0][:2]*2,
        }

        target_ys_batch_1 = label_modifier(self.train_A_file_1_ctnt[1][:2])

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:]*2,
            "input_B": self.train_B_file_1_ctnt[0][2:]*2,
        }

        target_ys_batch_2 = label_modifier(self.train_A_file_1_ctnt[1][2:])

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

    def test_batch_mc_infos(self):
        filepaths = self.filepaths_file_1

        gene = hdf5_batch_generator(self.orga, filepaths, yield_mc_info=True)

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2],
            "input_B": self.train_B_file_1_ctnt[0][:2],
        }

        target_ys_batch_1 = label_modifier(self.train_A_file_1_ctnt[1][:2])
        target_mc_info_batch_1 = self.train_A_file_1_ctnt[1][:2]

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:],
            "input_B": self.train_B_file_1_ctnt[0][2:],
        }

        target_ys_batch_2 = label_modifier(self.train_A_file_1_ctnt[1][2:])
        target_mc_info_batch_2 = self.train_A_file_1_ctnt[1][2:]

        xs, ys, mc_info = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)
        assert_equal_struc_array(mc_info, target_mc_info_batch_1)

        xs, ys, mc_info = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)
        assert_equal_struc_array(mc_info, target_mc_info_batch_2)


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

    def test_weighted_average(self):
        # metrics [A, B, C]
        histories = [
            [0, 1, 2],  # file 1
            [3, 4, 5],  # file 2
        ]

        file_sizes = [
            1,
            3,
        ]

        target = [
            (0*1 + 3*3)/4, (1*1 + 4*3)/4, (2*1 + 5*3)/4,
        ]
        averaged_histories = weighted_average(histories, file_sizes)

        self.assertSequenceEqual(averaged_histories, target)

    def test_weighted_average_one_file(self):
        # metrics [A, B, C]
        histories = [
            [0, 1, 2],  # file 1
        ]

        file_sizes = [
            1,
        ]

        target = [
            0, 1, 2
        ]
        averaged_histories = weighted_average(histories, file_sizes)

        self.assertSequenceEqual(averaged_histories, target)

    def test_weighted_average_one_metric(self):
        # metrics [A, ]
        histories = [
            [0, ],  # file 1
            [3, ],  # file 2
        ]

        file_sizes = [
            1,
            3,
        ]

        target = [
            (0*1 + 3*3)/4,
        ]
        averaged_histories = weighted_average(histories, file_sizes)

        self.assertSequenceEqual(averaged_histories, target)

    def test_weighted_average_one_metric_one_file(self):
        # metrics [A, ]
        histories = [
            [1, ],  # file 1
        ]

        file_sizes = [
            1,
        ]

        target = [
            1,
        ]
        averaged_histories = weighted_average(histories, file_sizes)

        self.assertSequenceEqual(averaged_histories, target)


class TestTrainValidatePredict(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "test_backend")
        cls.pred_dir = os.path.join(os.path.dirname(__file__), ".temp",
                                    "predictions")
        os.mkdir(cls.temp_dir)
        os.mkdir(cls.pred_dir)
        cls.pred_filepath = cls.pred_dir + '/pred_model_epoch_1_file_3_on_listfilename_val_file_1.h5'
        cls.file_sizes = [500, ]
        # make some dummy data
        cls.inp_A_file = {
            "path": cls.temp_dir + "/input_A_file.h5",
            "shape": (2, 3),
            "size": cls.file_sizes[0],
        }

        cls.inp_B_file = {
            "path": cls.temp_dir + "/input_B_file.h5",
            "shape": (3, 4),
            "size": cls.file_sizes[0],
        }

        cls.filepaths = {
            "input_A": (cls.inp_A_file["path"],),
            "input_B": (cls.inp_B_file["path"],),
        }

        cls.input_shapes = {
            "input_A": cls.inp_A_file["shape"],
            "input_B": cls.inp_B_file["shape"],
        }

        cls.output_shapes = {
            "mc_A": 1,
            "mc_B": 1,
        }

        cls.train_A_file_1_ctnt = save_dummy_h5py(**cls.inp_A_file, mode="half")
        cls.train_B_file_1_ctnt = save_dummy_h5py(**cls.inp_B_file, mode="half")

    def setUp(self):
        self.orga = Organizer("./.temp")
        self.orga.cfg.batchsize = 9

        self.orga.io.get_local_files = MagicMock(return_value=self.filepaths)
        self.orga.io.get_file_sizes = MagicMock(return_value=self.file_sizes)
        self.orga.cfg.get_list_file = MagicMock(return_value='/path/to/a/listfilename.toml')
        self.orga.io.get_next_pred_path = MagicMock(return_value=self.pred_filepath)

        self.model = build_dummy_model(self.input_shapes, self.output_shapes)
        self.model.compile(loss="mse", optimizer="sgd")
        self.orga._auto_label_modifier = get_auto_label_modifier(self.model)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.inp_A_file["path"])
        os.remove(cls.inp_B_file["path"])
        os.rmdir(cls.temp_dir)
        os.rmdir(cls.pred_dir)

    def test_train(self):
        epoch = (1, 1)
        batch_nos = []
        batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: batch_nos.append(batch))
        self.orga.cfg.callback_train = batch_print_callback

        history = train_model(self.orga, self.model, epoch, batch_logger=False)
        target = {
            'loss': 18.105026489263054,
            'mc_A_loss': 9.569378078221458,
            'mc_B_loss': 8.535648400507155,
        }
        self.assertDictEqual(history, target)
        self.assertSequenceEqual(batch_nos, list(range(int(self.file_sizes[0]/self.orga.cfg.batchsize))))

    def test_validate(self):
        history = validate_model(self.orga, self.model)
        # input to model is ones
        # --> after concatenate: 18 ones
        # Output of each output layer = 18
        # labels: 0 and 1
        # --> loss (18-0)^2 and (18-1)^2
        target = {
            'loss': 18**2 + 17**2,
            'mc_A_loss': 18**2,
            'mc_B_loss': 17**2,
        }
        self.model.summary()
        self.assertDictEqual(history, target)

    def test_predict(self):
        # dummy values
        epoch, fileno = 1, 3
        # mock get_latest_prediction_file_no
        self.orga.io.get_latest_prediction_file_no = MagicMock(return_value=None)

        try:
            make_model_prediction(self.orga, self.model, epoch, fileno)

            file_cntn = {}
            with h5py.File(self.pred_filepath, 'r') as file:
                for key in file.keys():
                    file_cntn[key] = np.array(file[key])
        finally:
            os.remove(self.pred_filepath)

        target_datasets = [
            'label_mc_A', 'label_mc_B', 'mc_info', 'pred_mc_A', 'pred_mc_B'
        ]
        target_shapes = [
            (500,), (500,), (500,), (500, 1), (500, 1),
        ]
        target_contents = [
            np.zeros(target_shapes[0]),
            np.ones(target_shapes[1]),
            self.train_A_file_1_ctnt[1],
            np.ones(target_shapes[3]) * 18,
            np.ones(target_shapes[4]) * 18,
        ]
        target_mc_names = ('mc_A', 'mc_B')

        datasets = list(file_cntn.keys())
        shapes = [file_cntn[key].shape for key in datasets]
        mc_dtype_names = file_cntn["mc_info"].dtype.names

        self.assertSequenceEqual(datasets, target_datasets)
        self.assertSequenceEqual(shapes, target_shapes)
        self.assertSequenceEqual(mc_dtype_names, target_mc_names)
        for i, key in enumerate(target_datasets):
            value = file_cntn[key]
            target = target_contents[i]
            np.testing.assert_array_equal(value, target)


def save_dummy_h5py(path, shape, size, mode="asc"):
    """
    if mode == asc:
        xs[0] is np.zeros, xs[1] is np.ones, etc
        ys[0] is ndarray with 0.5s, ys[1] with 1.5s etc

    if mode == half:
        xs is ones,
        ys[:, 0] is 0, ys[:, 1] is 1

    """
    xs = np.ones((size,) + shape)

    dtypes = [('mc_A', '<f8'), ('mc_B', '<f8'), ]
    ys = np.ones((size, 2))

    if mode == "asc":
        for sample_no in range(len(xs)):
            xs[sample_no, ...] = sample_no
            ys[sample_no, ...] = sample_no + 0.5

    elif mode == "half":
        ys[:, 0] = 0.

    else:
        raise AssertionError

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


def assert_equal_struc_array(a, b):
    """  np.testing.assert_array_equal does not work for arrays containing nans...
     so test individual instead. """
    np.testing.assert_array_equal(a.dtype, b.dtype)
    for name in a.dtype.names:
        np.testing.assert_almost_equal(a[name], b[name])


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
    inputs, flattend = [], []
    for name, shape in input_shapes.items():
        inp = Input(shape, name=name)
        flat = Flatten()(inp)
        inputs.append(inp)
        flattend.append(flat)

    conc = Concatenate()(flattend)

    outputs = []
    for name, shape in output_shapes.items():
        outputs.append(Dense(shape, name=name, kernel_initializer="Ones",
                             bias_initializer="Zeros")(conc))

    model = Model(inputs, outputs)
    return model
