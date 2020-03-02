#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch
import tensorflow.keras as ks
import tensorflow.keras.layers as layers

from orcanet.core import Organizer
from orcanet.model_builder import ModelBuilder
from orcanet_contrib.custom_objects import get_custom_objects


class TestModel(TestCase):
    def setUp(self):
        """
        Make a .temp directory in the current working directory, generate
        dummy data in it and set up the cfg object.

        """
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        self.model_file = os.path.join(self.data_folder, "model_builder_test_CNN_cat.toml")

        self.input_shapes = {
            "input_A": (3, 3, 3, 3),
        }

        orga = Organizer(".")
        orga.io.get_input_shapes = MagicMock(return_value=self.input_shapes)
        orga.cfg.custom_objects = get_custom_objects()

        self.orga = orga

    def test_model_setup_CNN_model(self):
        orga = self.orga

        builder = ModelBuilder(self.model_file)
        model = builder.build(orga)

        self.assertEqual(model.input_shape[1:], self.input_shapes["input_A"])
        self.assertEqual(model.output_shape[1:], (2, ))
        self.assertEqual(len(model.layers), 14)
        self.assertEqual(model.optimizer.epsilon, 0.2)

    def test_model_setup_CNN_model_custom_callback(self):
        builder = ModelBuilder(self.model_file)
        builder.optimizer = ks.optimizers.SGD()
        model = builder.build(self.orga)
        self.assertIsInstance(model.optimizer, ks.optimizers.SGD)

    @patch('orcanet.model_builder.toml.load')
    def test_load_optimizer(self, mock_toml_load):
        def toml_load(file):
            return file
        mock_toml_load.side_effect = toml_load

        file_cntn = {
            "model": {"blocks": None},
            "compile": {"optimizer": "keras:Adam", "losses": None, "lr": 1.0}
        }
        builder = ModelBuilder(file_cntn)
        opti = builder._get_optimizer()

        target = {
            'name': 'Adam',
            'learning_rate': 1.0,
            'beta_1': 0.8999999761581421,
            'beta_2': 0.9990000128746033,
            'decay': 0.0,
            'epsilon': 1e-07,
            'amsgrad': False
        }

        for k, v in opti.get_config().items():
            self.assertAlmostEqual(v, target[k])

    @patch('orcanet.model_builder.toml.load')
    def test_custom_blocks(self, mock_toml_load):
        def toml_load(file):
            return file
        mock_toml_load.side_effect = toml_load

        file_cntn = {
            "model": {"blocks": [{"type": "my_custom_block", "units": 10}, ]},
        }
        builder = ModelBuilder(file_cntn, my_custom_block=layers.Dense)
        model = builder.build_with_input({"a": (10, 1)}, compile_model=False)

        self.assertIsInstance(model.layers[-1], layers.Dense)
        self.assertEqual(model.layers[-1].get_config()["units"], 10)

    # def test_merge_models(self):
    #     def build_model(inp_layer_name, inp_shape):
    #         inp = layers.Input(inp_shape, name=inp_layer_name)
    #         x = layers.Convolution3D(3, 3)(inp)
    #         x = layers.Flatten()(x)
    #         out = layers.Dense(1, name="out_0")(x)
    #
    #         model = Model(inp, out)
    #         return model
    #
    #     model_file = os.path.join(self.data_folder, self.model_file)
    #     builder = ModelBuilder(model_file)
    #     model1 = build_model("inp_A", self.input_shapes["input_A"])
    #     model2 = build_model("inp_B", self.input_shapes["input_A"])
    #     merged_model = builder.merge_models([model1, model2])
    #
    #     for layer in model1.layers + model2.layers:
    #         if isinstance(layer, layers.Dense):
    #             continue
    #         merged_layer = merged_model.get_layer(layer.name)
    #         for i in range(len(layer.get_weights())):
    #             self.assertTrue(np.array_equal(layer.get_weights()[i],
    #                                            merged_layer.get_weights()[i]))
