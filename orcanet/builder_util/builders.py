import inspect
import warnings
import keras as ks
import keras.layers as layers

import orcanet.builder_util.layer_blocks as layer_blocks


class BlockBuilder:
    """
    Builds single-input block-wise sequential neural network.

    Attributes
    ----------
    defaults : dict or None
        Default values for all blocks in the model.
    verbose : bool
        Print info about the building process?

    """
    def __init__(self, defaults=None, verbose=False, **kwargs):
        """
        Set dict with default values for the layers of the model.
        Can also define custom block names as kwargs (key = toml name,
        value = block).
        """
        # dict with toml keyword vs block for all custom blocks
        self.all_blocks = dict(inspect.getmembers(layer_blocks, inspect.isclass))

        # legacy
        self.all_blocks = {
            **self.all_blocks,
            "conv_block": layer_blocks.ConvBlock,
            "dense_block": layer_blocks.DenseBlock,
            "resnet_block": layer_blocks.ResnetBlock,
            "resnet_bneck_block": layer_blocks.ResnetBnetBlock,
            "categorical": _attach_output_cat,
            "gpool": _attach_output_gpool_categ,
            "gpool_categ": _attach_output_gpool_categ,
            "gpool_reg": layer_blocks.OutputReg,
            "regression_error": layer_blocks.OutputRegErr,
        }

        if kwargs:
            self.all_blocks = {**self.all_blocks, **kwargs}

        self._check_arguments(defaults)
        self.defaults = defaults
        self.verbose = verbose

    def build(self, input_shape, configs):
        """
        Build the whole model, using the default values when arguments
        are missing in the layer_configs.

        Parameters
        ----------
        input_shape : dict
            Name and shape of the input layer.
        configs : list
            List of configurations for the blocks in the model.
            Each element in the list is a dict and will result in one block
            connected to the previous one. The dict has to contain the type
            of the block, as well as any arguments required by that
            specific block type.

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

        input_layer = get_input_block(input_shape)[0]

        x = input_layer
        for layer_config in configs:
            x = self.attach_block(x, layer_config)

        return ks.models.Model(inputs=input_layer, outputs=x)

    def attach_block(self, layer, layer_config):
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

        Returns
        -------
        keras layer

        """
        filled = self._with_defaults(layer_config, self.defaults)
        if self.verbose:
            print(f"Attaching layer {filled} to tensor {layer}")
        block = self._get_blocks(filled.pop("type"))
        return block(**filled)(layer)

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
        args = list(inspect.signature(block).parameters.keys())

        if defaults is not None:
            for key, val in defaults.items():
                if key in args and key not in conf:
                    conf[key] = val

        return conf

    def _get_blocks(self, name=None):
        """ Get the block class/function depending on the name. """
        if name is None:
            return self.all_blocks
        elif name.startswith("keras:"):
            return getattr(ks.layers, name.split("keras:")[1])
        elif name in self.all_blocks:
            return self.all_blocks[name]
        else:
            raise NameError(
                f"Unknown block type: {name}, must either start with "
                f"'keras:', or be one of {list(self.all_blocks.keys())}")

    def _check_arguments(self, defaults):
        """ Check if given defaults appear in at least one block. """
        if defaults is None:
            return
        # possible arguments for all blocks
        psb_args = ["type", ]
        for block in self._get_blocks().values():
            args = list(inspect.signature(block).parameters.keys())
            for arg in args:
                if arg not in psb_args and arg != "kwargs":
                    psb_args.append(arg)

        for t_def in defaults.keys():
            if t_def not in psb_args:
                warnings.warn(
                    f"Unknown default argument: {t_def} (has to appear in a block)")


def get_input_block(input_shapes):
    """
    Build input layers according to a dict mapping the layer names to shapes.

    Parameters
    ----------
    input_shapes : dict
        Keys: Input layer names.
        Values: Their shapes.

    Returns
    -------
    inputs : List
        A list of named keras input layers.

    """
    inputs = []
    for input_name, input_shape in input_shapes.items():
        inputs.append(layers.Input(
            shape=input_shape, name=input_name, dtype=ks.backend.floatx()))
    return inputs


class _attach_output_cat:
    def __init__(self, categories, output_name,
                 flatten=True):
        self.categories = categories
        self.output_name = output_name
        self.flatten = flatten

    def __call__(self, layer):
        if self.flatten:
            transition = "keras:Flatten"
        else:
            transition = None
        out = layer_blocks.OutputCateg(
            categories=self.categories,
            output_name=self.output_name,
            transition=transition,
            unit_list=(128, 32),
        )(layer)
        return out


class _attach_output_gpool_categ:
    def __init__(self, categories, output_name, dropout=None):
        self.categories = categories
        self.output_name = output_name
        self.dropout = dropout

    def __call__(self, layer):
        x = layers.GlobalAveragePooling2D()(layer)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        out = layer_blocks.OutputCateg(
            categories=self.categories,
            output_name=self.output_name,
            transition=None,
        )(x)
        return out
