#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import warnings
import toml
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as layers

from orcanet.builder_util.builders import BlockBuilder


class ModelBuilder:
    """
    Build and compile a keras model from a toml file, using OrcaNet building blocks.

    The input of the model can match the dimensions of the input
    data given to the Organizer taking into account the sample
    modifier.

    Attributes
    ----------
    configs : list
        List with keywords for building each layer block in the model.
    defaults : dict
        Default values for the layer blocks in the model.
    optimizer : str or Optimizer
        Optimizer for training the model. Can be a string like "adam" (or
        "keras:adam" for the default keras variant), or an object derived
        from ks.optimizers.Optimizer.
    compile_opt : dict
        Keys: Names of the output layers of the model.
        Values: Loss function, optionally weight and optionally metric of
        each output layer.
        Format: { layer_name : { loss_function:, weight:, metrics: } }
        The loss_function is a string or a function, the weight is a float
        and metrics is a list of functions/strings.
    optimizer_args : dict, optional
        Kwargs for the optimizer. Not used when an optimizer object is given.
    input_opts : dict
        Specify options for the input of the model.

    Methods
    -------
    build
        Build the network using an instance of Organizer.
    build_with_input
        Build the network without an Organizer, just using given input shapes.
    compile
        Compile a model with the optimizer settings given in the model_file.

    """
    def __init__(self, model_file, **custom_blocks):
        """
        Read out parameters for creating models with OrcaNet from a toml file.

        Parameters
        ----------
        model_file : str
            Path to the model toml file.
        custom_blocks
            For building models with custom blocks in the toml:
            Custom block names as kwargs ('toml name'='block').

        """
        file_content = toml.load(model_file)
        self.custom_blocks = custom_blocks

        try:
            if "model" in file_content:
                model_args = file_content["model"]
                self.configs = model_args.pop('blocks')
                self.input_opts = model_args.pop("input_opts", {})
                self.defaults = model_args

            elif "body" in file_content:
                # legacy
                self._compat_init(file_content)

            self.optimizer = None
            self.compile_opt = None
            self.optimizer_args = {}
            if "compile" in file_content:
                compile_sect = file_content["compile"]
                self.optimizer = compile_sect.pop("optimizer", None)
                self.compile_opt = compile_sect.pop("losses", None)
                self.optimizer_args = compile_sect

        except KeyError as e:
            if len(e.args) == 1:
                option = e.args[0]
            else:
                option = e.args
            raise KeyError("Missing parameter in toml model file: "
                           + str(option)) from None

    def _compat_init(self, file_content):
        warnings.warn(
            "The format of this model toml file is deprecated, consider "
            "updating it to the new format (see online docu)."
        )
        # legacy
        body = file_content["body"]
        if "architecture" in body:
            arch = body.pop("architecture")
            if arch != "single":
                raise ValueError("architecture keyword is deprecated")
        self.configs = body.pop('blocks')
        self.defaults = body

        if "head" in file_content:
            head = file_content["head"]
            head_arch = head.pop("architecture")
            head_arch_args = head.pop("architecture_args")
            head_args = head

            head_block_config = head_arch_args
            head_block_config["type"] = head_arch
            self.configs.append({**head_block_config, **head_args})

    def build(self, orga, log_comp_opts=False, verbose=False):
        """
        Build the network using an instance of Organizer.

        Input layers will be adapted to the input files in the organizer.
        Can also add the matching modifiers and custom objects to the orga.

        Parameters
        ----------
        orga : orcanet.core.Organizer
            Contains all the configurable options in the OrcaNet scripts.
        log_comp_opts : bool
            If the info used for the compilation of the model should be
            logged to the log.txt.
        verbose : bool
            Print info about the building process?

        Returns
        -------
        model : keras model
            The network.

        """
        if orga.cfg.fixed_batchsize:
            if "batchsize" in self.input_opts and \
                    self.input_opts["batchsize"] != orga.cfg.batchsize:
                raise ValueError(
                    f"Batchsize in input_opts is {self.input_opts['batchsize']}, "
                    f"but in cfg its {orga.cfg.batchsize}")
            self.input_opts["batchsize"] = orga.cfg.batchsize

        def get_model():
            return self.build_with_input(
                orga.io.get_input_shapes(),
                compile_model=True,
                custom_objects=orga.cfg.get_custom_objects(),
                verbose=verbose,
            )

        if orga.cfg.multi_gpu and len(
                tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f'Number of GPUs: {strategy.num_replicas_in_sync}')
            with strategy.scope():
                model = get_model()
        else:
            model = get_model()

        if log_comp_opts:
            self.log_model_properties(orga)
        model.summary()
        return model

    def build_with_input(self, input_shapes,
                         compile_model=True,
                         custom_objects=None,
                         verbose=False):
        """
        Build the network with given input shapes.

        Parameters
        ----------
        input_shapes : dict
            Keys: Name of the inputs of the model.
            Values: Their shape without the batchsize.
        compile_model : bool
            Compile the model?
        custom_objects : dict, optional
            Custom objects to use during compiling.
        verbose : bool
            Print info about the building process?

        Returns
        -------
        model : ks.Model
            The network.

        """
        builder = BlockBuilder(
            self.defaults, verbose=verbose, input_opts=self.input_opts,
            **self.custom_blocks)
        model = builder.build(input_shapes, self.configs)

        if compile_model:
            self.compile_model(model, custom_objects=custom_objects)

        return model

    # def merge_models(self, model_list, trainable=False, stateful=True,
    #                  no_drop=True):
    #     """
    #     Concatenate two or more single input cnns to a big one.
    #
    #     It will explicitly look for a Flatten layer and cut after it,
    #     Concatenate all models, and then add the head layers.
    #
    #     Parameters
    #     ----------
    #     model_list : list
    #         List of keras models to stitch together.
    #     trainable : bool
    #         Whether the layers of the loaded models will be trainable.
    #     stateful : bool
    #         Whether the batchnorms of the loaded models will be stateful.
    #     no_drop : bool
    #         If true, rate of dropout layers from loaded models will
    #         be set to zero.
    #
    #     Returns
    #     -------
    #     model : keras model
    #         The uncompiled merged keras model.
    #
    #     """
    #     # Get the input and Flatten layers in each of the given models
    #     input_layers, flattens = [], []
    #     for i, model in enumerate(model_list):
    #         if len(model.inputs) != 1:
    #             raise ValueError(
    #                 "model input is not length 1 {}".format(model.inputs))
    #         input_layers.append(model.input)
    #         flatten_found = 0
    #         for layer in model.layers:
    #             layer.trainable = trainable
    #             layer.name = layer.name + '_net_' + str(i)
    #             if isinstance(layer, layers.BatchNormalization):
    #                 layer.stateful = stateful
    #             elif isinstance(layer, layers.Flatten):
    #                 flattens.append(layer.output)
    #                 flatten_found += 1
    #         if flatten_found != 1:
    #             raise TypeError(
    #                 "Expected 1 Flatten layer but got " + str(flatten_found))
    #
    #     # attach new head
    #     x = layers.Concatenate()(flattens)
    #     builder = BlockBuilder(body_defaults=None,
    #                            head_defaults=self.head_args)
    #     output_layer = builder.attach_output_layers(x, self.head_arch,
    #                                                 flatten=False,
    #                                                 **self.head_arch_args)
    #
    #     model = ks.models.Model(input_layers, output_layer)
    #     if no_drop:
    #         model = change_dropout_rate(model, before_concat=0.)
    #
    #     return model

    def compile_model(self, model, custom_objects=None):
        """
        Compile a model with the optimizer settings given as the attributes.

        Parameters
        ----------
        model : ks.model
            A keras model.
        custom_objects : dict or None
            Maps names (strings) to custom loss functions.

        Returns
        -------
        model : keras model
            The compiled (or recompiled) keras model.

        """
        if any((self.optimizer is None, self.compile_opt is None)):
            raise ValueError("Can not compile, need optimizer name and losses")

        loss_functions, loss_weights, loss_metrics = {}, {}, {}
        for layer_name, layer_info in self.compile_opt.items():
            # Replace the str function name with actual function if it is custom
            loss_function = layer_info['function']
            if custom_objects is not None and loss_function in custom_objects:
                loss_function = custom_objects[loss_function]
            loss_functions[layer_name] = loss_function

            # Use given weight, else use default weight of 1
            if 'weight' in layer_info:
                weight = layer_info['weight']
            else:
                weight = 1.0
            loss_weights[layer_name] = weight

            # Use given metrics, else use no metrics
            if 'metrics' in layer_info:
                metrics = layer_info['metrics']
            else:
                metrics = []
            if custom_objects is not None:
                for i, metric in enumerate(metrics):
                    if metric in custom_objects:
                        metrics[i] = custom_objects[metric]

            loss_metrics[layer_name] = metrics

        optimizer = self._get_optimizer()
        model.compile(loss=loss_functions, optimizer=optimizer,
                      metrics=loss_metrics, loss_weights=loss_weights)
        return model

    def log_model_properties(self, orga):
        """
        Writes the compile_opt config to the full log file.
        """
        lines = list()
        lines.append('-' * 60)
        time = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        lines.append('-' * 19 + " {} ".format(time) + '-' * 19)
        lines.append("A model has been built using the model builder with the following configurations:\n")
        lines.append('Loss functions: ')
        for key in self.compile_opt:
            lines.append(key + ': ' + str(self.compile_opt[key]))
        lines.append('\n')
        orga.io.print_log(lines)

    def _get_optimizer(self):
        if not isinstance(self.optimizer, str):
            if self.optimizer_args:
                warnings.warn(
                    "Custom callback used, optimizer_args are ignored: " +
                    str(self.optimizer_args)
                )
            return self.optimizer
        if self.optimizer == 'adam':
            optimizer = get_adam(**self.optimizer_args)
        elif self.optimizer == 'sgd':
            optimizer = get_sgd(**self.optimizer_args)
        elif self.optimizer.startswith("keras:"):
            optimizer = getattr(
                ks.optimizers, self.optimizer.split("keras:")[-1]
            )(**self.optimizer_args)
        else:
            raise NameError('Unknown optimizer name ({})'.format(self.optimizer))
        return optimizer


def get_adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0, **kwargs):
    # epsilon=1 for deep networks
    return ks.optimizers.Adam(beta_1=beta_1,
                              beta_2=beta_2,
                              epsilon=epsilon,
                              decay=decay,
                              **kwargs)


def get_sgd(momentum=0.9, decay=0, nesterov=True, **kwargs):
    return ks.optimizers.SGD(momentum=momentum,
                             decay=decay,
                             nesterov=nesterov,
                             **kwargs)


def _change_dropout_rate(model, before_concat, after_concat=None):
    """
    Change the dropout rate in a model.

    # TODO untested for tf 2.x!

    Only for models with a concatenate layer, aka multiple
    single input models that were merged together.

    Parameters
    ----------
    model : keras model

    before_concat : float
        New dropout rate before the concatenate layer in the model.
    after_concat : float or None
        New dropout rate after the concatenate layer. None will leave the
        dropout rate there as it was.

    """
    ch_bef, ch_aft, concat_found = 0, 0, 0

    for layer in model.layers:
        if isinstance(layer, layers.Dropout):
            if concat_found == 0:
                layer.rate = before_concat
                ch_bef += 1
            else:
                layer.rate = after_concat
                ch_aft += 1

        elif isinstance(layer, layers.Concatenate):
            concat_found += 1
            if after_concat is None:
                break

    if concat_found != 1:
        raise TypeError("Expected 1 Flatten layer but got " + str(concat_found))
    clone = ks.models.clone_model(model)
    clone.set_weights(model.get_weights())
    print("Changed dropout rates of {} layers before and {} layers after "
          "Concatenate.".format(ch_bef, ch_aft))
    return clone
