from unittest import TestCase
from unittest.mock import MagicMock, patch
from keras.models import Model
from keras.layers import Dense, Input
import os

from orcanet.core import Organizer, Configuration


class TestOrganizer(TestCase):
    def setUp(self):
        self.orga = Organizer("./")
        self.temp_dir = os.path.join(os.path.dirname(__file__), ".temp", "core")

    def test_load_model_new_model_no_force(self):
        # latest epoch is None aka no model has been trained
        latest_epoch = None
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # no model given = error
        target_model = None
        with self.assertRaises(ValueError):
            self.orga._load_model(target_model, force_model=False)

        # model given = ok
        target_model = "the model is simply handed through"
        model, epoch = self.orga._load_model(target_model, force_model=False)
        self.assertEqual(target_model, model)
        self.assertEqual(latest_epoch, epoch)

    def test_load_model_existing_model_no_force(self):
        # latest epoch is not None
        latest_epoch = (1, 1)
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # model given = error
        target_model = "the model is simply handed through"
        with self.assertRaises(ValueError):
            self.orga._load_model(target_model, force_model=False)

        model_file = self.temp_dir + "test_model.h5"
        self.orga.io.get_model_path = MagicMock(return_value=model_file)

        try:
            os.mkdir(self.temp_dir)
            # no model given = ok
            target_model = build_test_model()
            target_model.save(model_file)
            model, epoch = self.orga._load_model(model=None, force_model=False)
            self.assertEqual(latest_epoch, epoch)
        finally:
            os.remove(model_file)
            os.rmdir(self.temp_dir)

    def test_load_model_new_model_force(self):
        # latest epoch is None aka no model has been trained
        latest_epoch = None
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # no model given = error
        target_model = None
        with self.assertRaises(ValueError):
            self.orga._load_model(target_model, force_model=True)

        # model given = ok
        target_model = "the model is simply handed through"
        model, epoch = self.orga._load_model(target_model, force_model=True)
        self.assertEqual(target_model, model)
        self.assertEqual(latest_epoch, epoch)

    def test_load_model_existing_model_force(self):
        # latest epoch is not None but a model has been given to function
        latest_epoch = (1, 1)
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # no model given = error
        target_model = None
        with self.assertRaises(ValueError):
            self.orga._load_model(target_model, force_model=True)

        # model given = ok
        target_model = "the model is simply handed through"

        model, epoch = self.orga._load_model(target_model, force_model=True)
        self.assertEqual(target_model, model)
        self.assertEqual(latest_epoch, epoch)


class TestZeroCenter(TestCase):
    def setUp(self):
        self.orga = Organizer("./")
        self.orga.cfg.zero_center_folder = "nope/"

    def test_get_xs_mean_existing(self):
        target = "asd"
        self.orga.xs_mean = target
        value = self.orga.get_xs_mean(logging=False)
        self.assertEqual(target, value)

    def test_get_xs_mean_no_folder(self):
        self.orga.cfg.zero_center_folder = None
        with self.assertRaises(ValueError):
            self.orga.get_xs_mean(logging=False)

    @patch('orcanet.core.load_zero_center_data')
    def test_get_xs_mean_new(self, load_mock):
        target = "blub"

        def mock_xs_mean():
            return target

        load_mock.return_value = mock_xs_mean()
        xs_mean = self.orga.get_xs_mean(logging=False)
        self.assertEqual(target, xs_mean)
        self.assertEqual(target, self.orga.xs_mean)


class TestConfiguration(TestCase):
    def setUp(self):
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.working_list = self.data_folder + "/in_out_test_list.toml"

    def test_output_folder_bracket(self):
        list_file = None
        config_file = None

        output_folder = "test"
        cfg = Configuration(output_folder, list_file, config_file)
        self.assertEqual(output_folder + "/", cfg.output_folder)

        output_folder = "test/"
        cfg = Configuration(output_folder, list_file, config_file)
        self.assertEqual(output_folder, cfg.output_folder)

    def test_listfile_already_given(self):
        config_file = None
        output_folder = "test"

        list_file = self.working_list
        cfg = Configuration(output_folder, list_file, config_file)

        with self.assertRaises(ValueError):
            cfg.import_list_file(list_file)


def build_test_model(compile=False):
    input_shape = (1,)
    inp = Input(input_shape, name="inp")
    outp = Dense(1, name="out")(inp)

    test_model = Model(inp, outp)
    if compile:
        test_model.compile("sgd", loss={"out": "mse"}, metrics={"out": "mae"})

    return test_model
