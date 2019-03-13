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

    def test_load_model_new_training(self):
        # latest epoch is None aka no model has been trained
        latest_epoch = None
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # no model given = error
        target_model = None
        with self.assertRaises(ValueError):
            self.orga._get_model(target_model, logging=False)

        # model given = ok
        target_model = "the model is simply handed through"
        model = self.orga._get_model(target_model, logging=False)
        self.assertEqual(target_model, model)

    def test_load_model_continue_training(self):
        # latest epoch is not None aka model has been trained before
        latest_epoch = (1, 1)
        self.orga.io.get_latest_epoch = MagicMock(return_value=latest_epoch)

        # no model given = load model
        target_model = None
        model_file = self.temp_dir + "test_model.h5"
        self.orga.io.get_model_path = MagicMock(return_value=model_file)

        try:
            os.mkdir(self.temp_dir)
            # no model given = ok
            saved_model = build_test_model()
            saved_model.save(model_file)
            loaded_model = self.orga._get_model(model=target_model, logging=False)
            # mssing: check if models are equal (they probably are)
        finally:
            os.remove(model_file)
            os.rmdir(self.temp_dir)

        # model given = ok
        target_model = "the model is simply handed through"

        model = self.orga._get_model(target_model, logging=False)
        self.assertEqual(target_model, model)

    def test_val_is_due(self):
        no_of_files = 3
        self.orga.cfg.validate_interval = 2
        targets = {
            (1, 1): False,
            (1, 2): True,
            (1, 3): True,

            (2, 1): False,
            (2, 2): True,
        }

        self.orga.io.get_no_of_files = MagicMock(return_value=no_of_files)
        for epoch, target in targets.items():
            value = self.orga._val_is_due(epoch)
            self.assertEqual(value, target)

    @patch('orcanet.core.make_model_prediction')
    def test_predict(self, mocked):
        self.orga.cfg._list_file = "test/test.toml"
        func_args = []

        def mock_make_model_prediction(orga, model, pred_filename, samples=None):
            func_args.extend([orga, model, pred_filename, samples])

        mocked.side_effect = mock_make_model_prediction

        self.orga.load_saved_model = MagicMock(return_value=build_test_model())

        value_filepath = self.orga.predict(epoch=1, fileno=1)
        target_filepath = "./predictions/pred_model_epoch_1_file_1_on_test_val_files.h5"

        self.assertEqual(value_filepath, target_filepath)
        self.assertEqual(func_args[2], value_filepath)
        self.assertEqual(func_args[3], None)


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
        self.temp_dir = os.path.join(os.path.dirname(__file__), ".temp", "core")
        os.mkdir(self.temp_dir)

    def tearDown(self):
        os.rmdir(self.temp_dir)

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

    def test_get_files_not_given(self):
        cfg = Configuration("./", None, None)
        with self.assertRaises(AttributeError):
            cfg.get_files("train")
        with self.assertRaises(AttributeError):
            cfg.get_files("val")
        with self.assertRaises(NameError):
            cfg.get_files("schmu")

    def test_update_config_bad_format(self):
        cfg = Configuration("./", None, None)

        content = [
            "[config]",
            "schmu=1",
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            with self.assertRaises(AttributeError):
                cfg.update_config(file_name)
        finally:
            os.remove(file_name)

    def test_load_list_file_ok(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'train_files = ["input_A_train_1.h5", "input_A_train_2.h5"]',
            'validation_files = ["input_A_val_1.h5"]',

            '[input_B]',
            'train_files = ["input_B_train_1.h5", "input_B_train_2.h5",]',
            'validation_files = ["input_B_val_1.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            cfg.import_list_file(file_name)

        finally:
            os.remove(file_name)

    def test_load_list_file_unknown_keyword(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'train_files = ["input_A_train_1.h5", "input_A_train_2.h5"]',
            'validation_files = ["input_A_val_1.h5"]',
            'schmu = 1',
            
            '[input_B]',
            'train_files = ["input_B_train_1.h5", "input_B_train_2.h5",]',
            'validation_files = ["input_B_val_1.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            with self.assertRaises(ValueError):
                cfg.import_list_file(file_name)
        finally:
            os.remove(file_name)

    def test_load_list_file_no_train_files(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'schmu=1',
            'validation_files = ["input_A_val_1.h5"]',

            '[input_B]',
            'train_files = ["input_B_train_1.h5", "input_B_train_2.h5",]',
            'validation_files = ["input_B_val_1.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            with self.assertRaises(NameError):
                cfg.import_list_file(file_name)
        finally:
            os.remove(file_name)

    def test_load_list_file_no_val_files(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'train_files = ["input_A_train_1.h5", "input_A_train_2.h5"]',
            'validation_files = ["input_A_val_1.h5"]',

            '[input_B]',
            'schmu=1',
            'train_files = ["input_B_train_1.h5", "input_B_train_2.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            with self.assertRaises(NameError):
                cfg.import_list_file(file_name)
        finally:
            os.remove(file_name)

    def test_load_list_file_different_no_of_train_files(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'train_files = ["input_A_train_1.h5", "input_A_train_2.h5"]',
            'validation_files = ["input_A_val_1.h5"]',

            '[input_B]',
            'train_files = ["input_B_train_1.h5", ]',
            'validation_files = ["input_B_val_1.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")
            with self.assertRaises(ValueError):
                cfg.import_list_file(file_name)
        finally:
            os.remove(file_name)

    def test_load_list_file_different_no_of_val_files(self):
        cfg = Configuration("./", None, None)

        content = [
            '[input_A]',
            'train_files = ["input_A_train_1.h5", "input_A_train_2.h5"]',
            'validation_files = ["input_A_val_1.h5", "input_A_val_2.h5"]',

            '[input_B]',
            'train_files = ["input_B_train_1.h5", "input_B_train_2.h5",]',
            'validation_files = ["input_B_val_1.h5",]',
        ]
        file_name = self.temp_dir + "/test_config.toml"

        try:
            with open(file_name, "w") as f:
                for line in content:
                    f.write(line + "\n")

            with self.assertRaises(ValueError):
                cfg.import_list_file(file_name)
        finally:
            os.remove(file_name)


def build_test_model(compile=False):
    input_shape = (1,)
    inp = Input(input_shape, name="inp")
    outp = Dense(1, name="out")(inp)

    test_model = Model(inp, outp)
    if compile:
        test_model.compile("sgd", loss={"out": "mse"}, metrics={"out": "mae"})

    return test_model
