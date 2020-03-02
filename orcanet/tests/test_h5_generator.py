import tempfile
from unittest import TestCase
from unittest.mock import MagicMock
import os
import numpy as np

from orcanet.core import Organizer
from orcanet.h5_generator import get_h5_generator
from orcanet.tests.test_backend import save_dummy_h5py, assert_dict_arrays_equal, assert_equal_struc_array


class TestBatchGenerator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tdir = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.tdir.name

        # make some dummy data
        cls.n_bins = {'input_A': (2, 3), 'input_B': (2, 3)}
        cls.train_sizes = [3, 5]
        cls.train_A_file_1 = {
            "path": os.path.join(cls.temp_dir, "input_A_train_1.h5"),
            "shape": cls.n_bins["input_A"],
            "size": cls.train_sizes[0],
        }

        cls.train_B_file_1 = {
            "path":  os.path.join(cls.temp_dir, "input_B_train_1.h5"),
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
        cls.tdir.cleanup()

    def test_batch(self):
        filepaths = self.filepaths_file_1
        gene = iter(get_h5_generator(self.orga, filepaths))

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2],
            "input_B": self.train_B_file_1_ctnt[0][:2],
        }

        target_ys_batch_1 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][:2]})

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:],
            "input_B": self.train_B_file_1_ctnt[0][2:],
        }

        target_ys_batch_2 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][2:]})

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
        gene = iter(get_h5_generator(self.orga, filepaths, zero_center=True))

        target_xs_batch_1 = {
            "input_A": np.subtract(self.train_A_file_1_ctnt[0][:2], xs_mean["input_A"]),
            "input_B": np.subtract(self.train_B_file_1_ctnt[0][:2], xs_mean["input_B"]),
        }

        target_ys_batch_1 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][:2]})

        target_xs_batch_2 = {
            "input_A": np.subtract(self.train_A_file_1_ctnt[0][2:], xs_mean["input_A"]),
            "input_B": np.subtract(self.train_B_file_1_ctnt[0][2:], xs_mean["input_B"]),
        }

        target_ys_batch_2 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][2:]})

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

    def test_batch_sample_modifier(self):
        filepaths = self.filepaths_file_1

        def sample_modifier(info_blob):
            xs_in = info_blob["x_values"]
            mod = {name: val*2 for name, val in xs_in.items()}
            return mod

        self.orga.cfg.sample_modifier = sample_modifier
        gene = iter(get_h5_generator(self.orga, filepaths))

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2]*2,
            "input_B": self.train_B_file_1_ctnt[0][:2]*2,
        }

        target_ys_batch_1 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][:2]})

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:]*2,
            "input_B": self.train_B_file_1_ctnt[0][2:]*2,
        }

        target_ys_batch_2 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][2:]})

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_1)
        assert_dict_arrays_equal(ys, target_ys_batch_1)

        xs, ys = next(gene)
        assert_dict_arrays_equal(xs, target_xs_batch_2)
        assert_dict_arrays_equal(ys, target_ys_batch_2)

    def test_batch_mc_infos(self):
        filepaths = self.filepaths_file_1

        gene = iter(get_h5_generator(self.orga, filepaths, keras_mode=False))

        target_xs_batch_1 = {
            "input_A": self.train_A_file_1_ctnt[0][:2],
            "input_B": self.train_B_file_1_ctnt[0][:2],
        }

        target_ys_batch_1 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][:2]})
        target_mc_info_batch_1 = self.train_A_file_1_ctnt[1][:2]

        target_xs_batch_2 = {
            "input_A": self.train_A_file_1_ctnt[0][2:],
            "input_B": self.train_B_file_1_ctnt[0][2:],
        }

        target_ys_batch_2 = label_modifier({"y_values": self.train_A_file_1_ctnt[1][2:]})
        target_mc_info_batch_2 = self.train_A_file_1_ctnt[1][2:]

        info_blob = next(gene)
        assert_dict_arrays_equal(info_blob["xs"], target_xs_batch_1)
        assert_dict_arrays_equal(info_blob["ys"], target_ys_batch_1)
        assert_equal_struc_array(info_blob["y_values"], target_mc_info_batch_1)

        info_blob = next(gene)
        assert_dict_arrays_equal(info_blob["xs"], target_xs_batch_2)
        assert_dict_arrays_equal(info_blob["ys"], target_ys_batch_2)
        assert_equal_struc_array(info_blob["y_values"], target_mc_info_batch_2)


def label_modifier(info_blob):
    y_values = info_blob["y_values"]
    ys = dict()
    for name in y_values.dtype.names:
        ys[name] = y_values[name]
    return ys
