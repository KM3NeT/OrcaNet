from unittest import TestCase
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

import orcanet.builder_util.layer_blocks as layer_blocks


class TestInceptionBlockV2(TestCase):
    def setUp(self):
        self.inp_2d = layers.Input(shape=(10, 10, 1))
        self.inp_3d = layers.Input(shape=(10, 10, 10, 1))

    def test_shape_2d(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
        )(self.inp_2d)
        self.assertSequenceEqual(
            Model(self.inp_2d, x).output_shape, (None, 10, 10, 21))

    def test_shape_2d_strides(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
            strides=2,
        )(self.inp_2d)
        self.assertSequenceEqual(
            Model(self.inp_2d, x).output_shape, (None, 5, 5, 15))

    def test_shape_3d(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=3,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
            strides=1,
        )(self.inp_3d)
        self.assertSequenceEqual(
            Model(self.inp_3d, x).output_shape, (None, 10, 10, 10, 21))

    def test_shape_3d_strides(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=3,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
            strides=2,
        )(self.inp_3d)
        self.assertSequenceEqual(
            Model(self.inp_3d, x).output_shape, (None, 5, 5, 5, 15))

    def test_2d_params(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
        )(self.inp_2d)
        self.assertEqual(
            Model(self.inp_2d, x).count_params(), 450)

    def test_2d_params_stride(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
            strides=2,
        )(self.inp_2d)
        self.assertEqual(
            Model(self.inp_2d, x).count_params(), 436)


class TestConvBlock(TestCase):
    def setUp(self):
        self.inp_2d = layers.Input(shape=(9, 10, 1))
        self.inp_3d = layers.Input(shape=(8, 9, 10, 1))

    def test_2d(self):
        x = layer_blocks.ConvBlock(
            conv_dim=2,
            filters=11,
            dropout=0.2,
            pool_size=2,
        )(self.inp_2d)
        model = Model(self.inp_2d, x)

        self.assertEqual(model.count_params(), 110)
        self.assertSequenceEqual(model.output_shape, (None, 4, 5, 11))
        target_layers = (
            layers.InputLayer,
            layers.Conv2D,
            layers.Activation,
            layers.MaxPool2D,
            layers.Dropout,
        )
        for layer, target_layer in zip(model.layers, target_layers):
            self.assertIsInstance(layer, target_layer)

    def test_2d_time_distributed(self):
        x = layer_blocks.ConvBlock(
            conv_dim=2,
            filters=11,
            dropout=0.2,
            pool_size=2,
            time_distributed=True,
            padding=[1, 0],
        )(self.inp_3d)
        model = Model(self.inp_3d, x)

        self.assertEqual(model.count_params(), 110)
        self.assertSequenceEqual(model.output_shape, (None, 8, 4, 4, 11))
        target_layers = [layers.InputLayer] + [layers.TimeDistributed] * 5

        for layer, target_layer in zip(model.layers, target_layers):
            self.assertIsInstance(layer, target_layer)
