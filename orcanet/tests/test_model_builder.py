#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Convolution3D
from unittest import TestCase
from unittest.mock import MagicMock

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

    def test_merge_models(self):
        def build_model(inp_layer_name, inp_shape):
            inp = Input(inp_shape, name=inp_layer_name)
            x = Convolution3D(3, 3)(inp)
            x = Flatten()(x)
            out = Dense(1, name="out_0")(x)

            model = Model(inp, out)
            return model

        model_file = os.path.join(self.data_folder, self.model_file)
        builder = ModelBuilder(model_file)
        model1 = build_model("inp_A", self.input_shapes["input_A"])
        model2 = build_model("inp_B", self.input_shapes["input_A"])
        merged_model = builder.merge_models([model1, model2])

        for layer in model1.layers + model2.layers:
            if isinstance(layer, Dense):
                continue
            merged_layer = merged_model.get_layer(layer.name)
            for i in range(len(layer.get_weights())):
                self.assertTrue(np.array_equal(layer.get_weights()[i],
                                               merged_layer.get_weights()[i]))
