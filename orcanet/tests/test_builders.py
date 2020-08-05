import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

from orcanet.builder_util.builders import BlockBuilder


class TestSequentialBuilder(tf.test.TestCase):

    def test_input_names_and_shapes_full_model(self):
        defaults = {"type": "conv_block", "conv_dim": 3}
        builder = BlockBuilder(defaults=defaults)

        input_name = "test_input"
        input_shape = (4, 4, 4, 1)
        input_shapes = {input_name: input_shape}

        conv_layer_config = (
            {"filters": 3, },
            {"filters": 3, "pool_size": (2, 2, 2)},
            {"filters": 5, },
            {"type": "OutputCateg", "output_name": "ts_output", "categories": 2, "transition": "keras:Flatten"}
        )

        model = builder.build(input_shapes, conv_layer_config)
        self.assertEqual([input_name, ], model.input_names)
        self.assertEqual(input_shape, model.input_shape[1:])

    def test_attach_layer_conv(self):
        inp = layers.Input((6, 6, 1))
        defaults = {
            "type": "conv_block",
            "conv_dim": 2,
        }
        layer_config = {
            "filters": 2,
            "pool_size": 2,
            "dropout": 0.2,
            "batchnorm": True,
            "kernel_l2_reg": 0.0001
        }

        builder = BlockBuilder(defaults)
        x = builder.attach_block(inp, layer_config)
        model = Model(inp, x)

        self.assertIsInstance(model.layers[1], layers.Convolution2D)
        kreg = model.layers[1].get_config()["kernel_regularizer"]["config"]
        self.assertTrue("l1" not in kreg)
        self.assertAlmostEqual(kreg["l2"], layer_config["kernel_l2_reg"])

        self.assertIsInstance(model.layers[2], layers.BatchNormalization)
        self.assertIsInstance(model.layers[3], layers.Activation)
        self.assertIsInstance(model.layers[4], layers.MaxPooling2D)
        self.assertIsInstance(model.layers[5], layers.Dropout)
        self.assertEqual(model.output_shape[1:], (3, 3, 2))

    def test_attach_output_layers_regression_output_shape_and_names(self):
        config = {"type": "OutputRegErr", "output_names": ['out_A', 'out_B'], "flatten": True}

        inp = layers.Input((5, 1))
        builder = BlockBuilder()
        x = builder.attach_block(inp, config)
        model = Model(inp, x)

        target_output_shapes = {
            'out_A': (1,),
            'out_B': (1,),
            'out_A_err': (2,),
            'out_B_err': (2,),
        }

        output_shapes = {}
        for i, output_name in enumerate(model.output_names):
            output_shapes[output_name] = model.output_shape[i][1:]

        self.assertDictEqual(output_shapes, target_output_shapes)

    def test_attach_layer_dense(self):
        inp = layers.Input((3, 3, 1))
        defaults = {"type": "conv_block", "conv_dim": 2}
        layer_config = {"type": "dense_block", "units": 5, "dropout": 0.2, "batchnorm": True}

        builder = BlockBuilder(defaults)
        x = builder.attach_block(inp, layer_config)
        model = Model(inp, x)

        self.assertIsInstance(model.layers[1], layers.Dense)
        self.assertIsInstance(model.layers[2], layers.BatchNormalization)
        self.assertIsInstance(model.layers[3], layers.Activation)
        self.assertIsInstance(model.layers[4], layers.Dropout)
        self.assertEqual(model.output_shape[1:], (3, 3, 5))

    def test_attach_layer_wrong_layer_config_conv_block(self):
        inp = layers.Input((3, 3, 1))
        layer_config = {"type": "conv_block", "filters": 2, "units": 5}
        builder = BlockBuilder(None)

        with self.assertRaises(TypeError):
            builder.attach_block(inp, layer_config)

    def test_attach_layer_wrong_layer_config_dense_block(self):
        inp = layers.Input((3, 3, 1))
        layer_config = {"type": "dense_block", "filters": 2, "units": 5}
        builder = BlockBuilder(None)

        with self.assertRaises(TypeError):
            builder.attach_block(inp, layer_config)

    def test_attach_layer_cudnn_lstm(self):
        defaults = {"type": "conv_block", "conv_dim": 3, "recurrent_activation": "relu"}
        layer_config = {"type": "keras:LSTM", "units": 5, "activation": "relu"}
        builder = BlockBuilder(defaults)

        inp = layers.Input((10, 1))
        x = builder.attach_block(inp, layer_config)
        model = Model(inp, x)

        self.assertSequenceEqual(model.output_shape, (None, 5))
        check_these_in_config = {
            "units": 5,
            "recurrent_activation": "relu",
            "activation": "relu",
        }
        cfg = model.layers[-1].get_config()
        for k, v in check_these_in_config.items():
            self.assertEqual(cfg[k], v)
