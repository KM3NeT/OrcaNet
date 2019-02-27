from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Lambda
from keras import backend as K
from keras.regularizers import l2
import inspect

from orcanet.builder_util.layer_blocks import (
    input_block, ConvBlock, DenseBlock)


class BlockBuilder:
    """
    Can build single-input block-wise sequential neural network.

    Attributes
    ----------
    body_defaults : dict or None
        Default values for all blocks in the body section of the model.
    head_defaults : dict or None
        Default values for all blocks in the head section of the model.

    """

    def __init__(self, body_defaults=None, head_defaults=None):
        """ Set dicts with default values for the layers of the model.
        """
        self._all_blocks = {
            "conv_block": ConvBlock,
            "dense_block": DenseBlock,
        }

        if body_defaults is not None and "kernel_reg" in body_defaults:
            if body_defaults["kernel_reg"] == "l2":
                body_defaults["kernel_reg"] = l2
            else:
                raise NameError(
                    "Unknown kernel_reg: " + str(body_defaults["kernel_reg"]))

        if head_defaults is not None and "kernel_reg" in head_defaults:
            if head_defaults["kernel_reg"] == "l2":
                head_defaults["kernel_reg"] = l2
            else:
                raise NameError(
                    "Unknown kernel_reg: " + str(head_defaults["kernel_reg"]))

        self._check_arguments(body_defaults)
        self._check_arguments(head_defaults)

        self.body_defaults = body_defaults
        self.head_defaults = head_defaults

    def build(self, input_shape, body_configs, head_arch, head_arch_args=None):
        """
        Build the whole model, using the default values when arguments
        are missing in the layer_configs.

        Parameters
        ----------
        input_shape : dict
            Name and shape of the input layer.
        body_configs : list
            List of configurations for the blocks in the body of the model.
            Each element in the list is a dict and will result in one block
            connected to the previous one. The dict has to contain the type
            of the block, as well as any arguments required by that
            specific block type.
        head_arch : str or None
            Specifies the architecture of the head.
        head_arch_args : dict or None
            Required arguments for the given head_arch.

        Returns
        -------
        model : keras model

        """
        if not isinstance(input_shape, dict):
            raise TypeError(
                "input_shapes must be a dict, not ", type(input_shape))
        if not len(input_shape) == 1:
            raise TypeError(
                "input_shapes must have length 1, not ", len(input_shape))

        input_layer = input_block(input_shape)[0]

        x = input_layer
        for layer_config in body_configs:
            x = self.attach_block(x, layer_config)
        conv_output_flat = Flatten()(x)

        if head_arch is None:
            outputs = conv_output_flat
        else:
            outputs = self.attach_output_layers(conv_output_flat,
                                                head_arch, head_arch_args)

        model = Model(inputs=input_layer, outputs=outputs)
        return model

    def attach_block(self, layer, layer_config, is_output=False):
        """
        Attach a block to the given layer based on the layer config.

        Will use the default values given during initialization if they are not
        present in the layer config.

        Parameters
        ----------
        layer : keras layer
            Layer to attach the block to.
        layer_config : dict
            Configuration of the block to attach. The dict has to contain
            the type of the block, as well as any arguments required by that
            specific block.
        is_output : bool
            Whether to use the default values for the head layers or
            the body layers.

        Returns
        -------
        x : keras layer

        """
        if is_output:
            defaults = self.head_defaults
        else:
            defaults = self.body_defaults
        filled = self._with_defaults(layer_config, defaults)
        block = self._get_blocks(filled.pop("type"))
        x = block(**filled)(layer)
        return x

    def attach_output_layers(self, layer, head_arch, head_arch_args):
        """
        Append the head dense layers to the network.

        Parameters
        ----------
        layer : keras layer
            The Keras layer instance to attach the head to.
        head_arch : str
            Specifies the architecture of the head layers.
        head_arch_args : dict
            Required arguments for the given output_arch.

        Returns
        -------
        outputs : list
            List of the keras output layers of the network.

        """
        if head_arch == "categorical":
            # categorical problem
            assert head_arch_args is not None, "No output_kwargs given"
            outputs = self.attach_output_cat(layer, **head_arch_args)

        elif head_arch == "regression_error":
            # regression with error estimation, two outputs for each label
            assert head_arch_args is not None, "No output_kwargs given"
            outputs = self.attach_output_reg_err(layer, **head_arch_args)

        else:
            raise ValueError("Unknown head_arch: " + str(head_arch))

        return outputs

    def attach_output_cat(self, layer, categories, output_name):
        """ Small dense network for multiple categories. """
        x = self.attach_block(
            layer, {"type": "dense_block", "units": 128},
            is_output=True)

        x = self.attach_block(
            x, {"type": "dense_block", "units": 32, "dropout": None},
            is_output=True)

        out = Dense(units=categories, activation='softmax',
                    kernel_initializer='he_normal', name=output_name)(x)
        return [out, ]

    def attach_output_reg_err(self, layer, output_names):
        """ Double network for regression + error estimation. """
        outputs = []

        # Network for the labels
        x = self.attach_block(
            layer, {"type": "dense_block", "units": 128},
            is_output=True)

        x = self.attach_block(
            x, {"type": "dense_block", "units": 32, "dropout": None},
            is_output=True)

        for name in output_names:
            output_label = Dense(units=1, name=name)(x)
            outputs.append(output_label)

        # Network for the errors of the labels
        x_err = Lambda(lambda a: K.stop_gradient(a))(layer)

        x_err = self.attach_block(
            x_err, {"type": "dense_block", "units": 128},
            is_output=True)

        x_err = self.attach_block(
            x_err, {"type": "dense_block", "units": 64},
            is_output=True)

        x_err = self.attach_block(
            x_err, {"type": "dense_block", "units": 32, "dropout": None},
            is_output=True)

        for i, name in enumerate(output_names):
            output_label_error = Dense(units=1, activation='linear',
                                       name=name + '_err_temp')(x_err)
            # Predicted label gets concatenated with its error (needed for loss function)
            output_label_merged = Concatenate(name=name + '_err')([outputs[i],
                                                                   output_label_error])
            outputs.append(output_label_merged)

        return outputs

    def _with_defaults(self, config, defaults):
        """ Make a copy of a layer config and complete it with default values
        for its block, if they are missing in the layer config.
        """
        conf = dict(config)

        if config is not None and "type" in config:
            block_name = config["type"]
        elif defaults is not None and "type" in defaults:
            block_name = defaults["type"]
            conf["type"] = defaults["type"]
        else:
            raise KeyError("No layer block type specified")

        block = self._get_blocks(block_name)
        args = inspect.getfullargspec(block).args
        if "self" in args:
            args.remove("self")

        if defaults is not None:
            for key, val in defaults.items():
                if key in args and key not in conf:
                    conf[key] = val

        return conf

    def _get_blocks(self, name=None):
        """ Get the block class/function depending on the name. """
        if name is None:
            return self._all_blocks
        elif name in self._all_blocks:
            return self._all_blocks[name]
        else:
            raise NameError("Unknown block type: " + str(name))

    def _check_arguments(self, defaults):
        """ Check if given defaults appear in at least one block. """
        if defaults is None:
            return
        # possible arguments for all blocks
        psb_args = ["type", ]
        for block in self._get_blocks().values():
            args = inspect.getfullargspec(block).args
            for arg in args:
                if arg not in psb_args and arg != "self":
                    psb_args.append(arg)

        for t_def in defaults.keys():
            if t_def not in psb_args:
                raise NameError("Unknown argument: " + str(t_def))
