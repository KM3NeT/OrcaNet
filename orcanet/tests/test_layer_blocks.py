from unittest import TestCase
import tensorflow as tf

import orcanet.builder_util.layer_blocks as layer_blocks


class TestInceptionBlockV2(TestCase):
    def setUp(self):
        self.inp_2d = tf.keras.layers.Input(shape=(10, 10, 1))
        self.inp_3d = tf.keras.layers.Input(shape=(10, 10, 10, 1))

    def test_shape_2d(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
        )(self.inp_2d)
        self.assertSequenceEqual(
            tf.keras.Model(self.inp_2d, x).output_shape, (None, 10, 10, 21))

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
            tf.keras.Model(self.inp_2d, x).output_shape, (None, 5, 5, 15))

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
            tf.keras.Model(self.inp_3d, x).output_shape, (None, 10, 10, 10, 21))

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
            tf.keras.Model(self.inp_3d, x).output_shape, (None, 5, 5, 5, 15))

    def test_2d_params(self):
        x = layer_blocks.InceptionBlockV2(
            conv_dim=2,
            filters_pool=3,
            filters_1x1=4,
            filters_3x3=(5, 6),
            filters_3x3dbl=(7, 8),
        )(self.inp_2d)
        self.assertEqual(
            tf.keras.Model(self.inp_2d, x).count_params(), 450)

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
            tf.keras.Model(self.inp_2d, x).count_params(), 436)


class TestConvBlock(TestCase):
    def setUp(self):
        self.inp_2d = tf.keras.layers.Input(shape=(9, 10, 1))
        self.inp_3d = tf.keras.layers.Input(shape=(8, 9, 10, 1))

    def test_2d(self):
        x = layer_blocks.ConvBlock(
            conv_dim=2,
            filters=11,
            dropout=0.2,
            pool_size=2,
        )(self.inp_2d)
        model = tf.keras.Model(self.inp_2d, x)

        self.assertEqual(model.count_params(), 110)
        self.assertSequenceEqual(model.output_shape, (None, 4, 5, 11))
        target_layers = (
            tf.keras.layers.InputLayer,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Activation,
            tf.keras.layers.MaxPool2D,
            tf.keras.layers.Dropout,
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
        model = tf.keras.Model(self.inp_3d, x)

        self.assertEqual(model.count_params(), 110)
        self.assertSequenceEqual(model.output_shape, (None, 8, 4, 4, 11))
        target_layers = [tf.keras.layers.InputLayer] + [tf.keras.layers.TimeDistributed] * 5

        for layer, target_layer in zip(model.layers, target_layers):
            self.assertIsInstance(layer, target_layer)


class TestOutputRegNormal(TestCase):
    # also using this class as template for other tests
    @classmethod
    def setUpClass(cls):
        inp = tf.keras.layers.Input(shape=(3, 4))
        output = layer_blocks.OutputRegNormal(
            output_neurons=3,
            output_name="test",
            unit_list=[5, 5],
            mu_activation="relu",
            transition="keras:Flatten",
        )(inp)
        cls.model = tf.keras.Model(inp, output)

    def setUp(self):
        self.targets = {
            "n_layers": 11,
            "output_shape": (None, 2, 3),
            "n_params": 131,
            "output_names": ["test"],
        }

    def test_unit_list_is_int(self):
        inp = tf.keras.layers.Input(shape=(3, 4))
        output = layer_blocks.OutputRegNormal(
            output_neurons=3,
            output_name="test",
            unit_list=5,
            mu_activation="relu",
            transition="keras:Flatten",
        )(inp)
        model = tf.keras.Model(inp, output)

    def test_n_layers(self):
        self.assertEqual(len(self.model.layers), self.targets["n_layers"])

    def test_output_shape(self):
        self.assertTupleEqual(self.model.output_shape, self.targets["output_shape"])

    def test_n_params(self):
        self.assertEqual(self.model.count_params(), self.targets["n_params"])

    def test_output_names(self):
        if "output_names" in self.targets:
            self.assertListEqual(self.model.output_names, self.targets["output_names"])


class TestOutputRegNormalSplit(TestOutputRegNormal):
    @classmethod
    def setUpClass(cls):
        inp = tf.keras.layers.Input(shape=(3, 4))
        output = layer_blocks.OutputRegNormalSplit(
            output_neurons=3,
            output_name="test",
            unit_list=[5, 5],
            sigma_unit_list=[3, ],
            mu_activation="relu",
            transition="keras:Flatten",
        )(inp)
        cls.model = tf.keras.Model(inp, output)

    def setUp(self):
        self.targets = {
            "n_layers": 15,
            "output_shape": [(None, 3), (None, 2, 3)],
            "n_params": 164,
            "output_names": ["test", "test_err"],
        }

    def test_output_shape(self):
        for i in range(len(self.targets["output_shape"])):
            self.assertTupleEqual(self.model.output_shape[i], self.targets["output_shape"][i])


class TestResnetBnetBlock(TestOutputRegNormal):
    @classmethod
    def setUpClass(cls):
        inp = tf.keras.layers.Input(shape=(5, 4, 3, 2))
        output = layer_blocks.ResnetBnetBlock(
            conv_dim=2,
            filters=[2, 3, 4],
            strides=2,
            batchnorm=True,
        )(inp)
        cls.model = tf.keras.Model(inp, output)

    def setUp(self):
        self.targets = {
            "n_layers": 13,
            "output_shape": (None, 5, 2, 2, 4),
            "n_params": 130,
        }
