import tempfile
from unittest import TestCase
from unittest.mock import MagicMock
import os
import warnings
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as layers

from orcanet.core import Organizer
from orcanet.backend import get_datasets, train_model, validate_model, make_model_prediction, weighted_average
from orcanet.utilities.nn_utilities import get_auto_label_modifier


class TestFunctions(TestCase):
    def test_get_datasets(self):
        y_values = "y_values gets simply passed forward"
        y_true = {
            "out_A": 1,
            "out_B": 2,
        }
        y_pred = {
            "out_pred_A": 3,
            "out_pred_B": 4,
        }
        target = {
            "y_values": y_values,
            "label_out_A": 1,
            "label_out_B": 2,
            "pred_out_pred_A": 3,
            "pred_out_pred_B": 4,
        }
        info_blob = {
            "y_values": y_values,
            "ys": y_true,
            "y_pred": y_pred,
        }

        datasets = get_datasets(info_blob)
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


class TestTrainValidatePredict(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tdir = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.tdir.name
        cls.pred_dir = os.path.join(cls.temp_dir, "predictions")

        os.mkdir(cls.pred_dir)

        cls.pred_filepath = os.path.join(
            cls.pred_dir,
            'pred_model_epoch_1_file_3_on_listfilename_val_file_1.h5')
        cls.file_sizes = [500, ]
        # make some dummy data
        cls.inp_A_file = {
            "path": os.path.join(cls.temp_dir, "input_A_file.h5"),
            "shape": (2, 3),
            "size": cls.file_sizes[0],
        }

        cls.inp_B_file = {
            "path": os.path.join(cls.temp_dir, "input_B_file.h5"),
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
        self.orga = Organizer(self.temp_dir)
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
        cls.tdir.cleanup()

    def test_train(self):
        epoch = (1, 1)
        batch_nos = []
        batch_print_callback = ks.callbacks.LambdaCallback(
            on_batch_begin=lambda batch, logs: batch_nos.append(batch))
        self.orga.cfg.callback_train = batch_print_callback

        history = train_model(self.orga, self.model, epoch, batch_logger=False)
        target = {  # TODO why does this sometimes change?
            'loss': 18.236408802816285,
            'mc_A_loss': 9.647336,
            'mc_B_loss': 8.597588874108167,
        }
        print(history, target)
        assert_dict_arrays_equal(history, target, rtol=1e-1)
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
        # self.orga.io.get_latest_prediction_file_no = MagicMock(return_value=None)

        try:
            make_model_prediction(self.orga, self.model, epoch, fileno)

            file_cntn = {}
            with h5py.File(self.pred_filepath, 'r') as file:
                for key in file.keys():
                    file_cntn[key] = np.array(file[key])
        finally:
            os.remove(self.pred_filepath)

        target_datasets = [
            'label_mc_A', 'label_mc_B', 'pred_mc_A', 'pred_mc_B', 'y_values'
        ]
        target_shapes = [
            (500,), (500,), (500, 1), (500, 1), (500,)
        ]
        target_contents = [
            np.zeros(target_shapes[0]),
            np.ones(target_shapes[1]),
            np.ones(target_shapes[2]) * 18,
            np.ones(target_shapes[3]) * 18,
            self.train_A_file_1_ctnt[1],
        ]
        shapes_dict = dict(zip(target_datasets, target_shapes))
        contents_dict = dict(zip(target_datasets, target_contents))

        target_mc_names = ('mc_A', 'mc_B')

        datasets = list(file_cntn.keys())
        shapes = [file_cntn[key].shape for key in datasets]
        mc_dtype_names = file_cntn["y_values"].dtype.names

        self.assertSequenceEqual(datasets, target_datasets)
        self.assertSequenceEqual(shapes, target_shapes)
        self.assertSequenceEqual(mc_dtype_names, target_mc_names)
        for i, key in enumerate(target_datasets):
            value = file_cntn[key]
            target = contents_dict[key]
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


def assert_dict_arrays_equal(a, b, rtol=1e-4):
    for key, value in a.items():
        assert key in b
        np.testing.assert_allclose(a[key], b[key], rtol=rtol)
        if not np.array_equal(a[key], b[key]):
            warnings.warn(f"Arrays not equal:{a[key]} != {b[key]}")


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
        inp = layers.Input(shape, name=name)
        flat = layers.Flatten()(inp)
        inputs.append(inp)
        flattend.append(flat)

    conc = layers.Concatenate()(flattend)

    outputs = []
    for name, shape in output_shapes.items():
        outputs.append(layers.Dense(
            shape, name=name, kernel_initializer="Ones",
            bias_initializer="Zeros")(conc))

    model = ks.models.Model(inputs, outputs)
    return model
