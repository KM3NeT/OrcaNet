#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import keras as ks
from keras.layers import Concatenate, Flatten, BatchNormalization, Dropout
from keras.models import Model
import toml
from datetime import datetime

from orcanet.builder_util.builders import BlockBuilder


class ModelBuilder:
    """
    Can build models adapted to an Organizer instance from a toml file.

    The input of the model will match the dimensions of the input
    data given to the Organizer taking into account the sample
    modifier.

    Attributes
    ----------
    body_arch : str
        Name of the architecture of the model (see self.build).
        Default: single.
    body_configs : list
        List with keywords for building each layer block in the model.
    body_args : dict
        Default values for the layer blocks in the model.
    head_arch : str
        Determines the head architecture of the model.
    head_arch_args : dict
        Keyword arguments for the specific output_arch used.
    head_args : dict
        Default values for the head layer blocks in the model.
    optimizer : str
        Optimizer for training the model. Either "Adam" or "SGD".
    compile_opt : dict
        Keys: Names of the output layers of the model.
        Values: Loss function, optionally weight and optionally metric of
        each output layer.
        Format: { layer_name : { loss_function:, weight:, metrics: } }
        The loss_function is a string or a function, the weight is a float
        and metrics is a list of functions/strings.
    optimizer_args : dict
        Kwargs for the optimizer.

    """
    def __init__(self, model_file):
        """
        Read out parameters for creating models with OrcaNet from a toml file.

        Parameters
        ----------
        model_file : str
            Path to the model toml file.

        """
        file_content = toml.load(model_file)

        try:
            body = file_content["body"]
            if "architecture" in body:
                self.body_arch = body.pop("architecture")
            else:
                self.body_arch = "single"

            self.body_configs = body.pop('blocks')
            self.body_args = body

            head = file_content["head"]
            self.head_arch = head.pop("architecture")
            self.head_arch_args = head.pop("architecture_args")
            self.head_args = head

            compile_sect = file_content["compile"]
            self.optimizer = compile_sect.pop("optimizer")
            self.compile_opt = compile_sect.pop("losses")
            self.optimizer_args = compile_sect

        except KeyError as e:
            if len(e.args) == 1:
                option = e.args[0]
            else:
                option = e.args
            raise KeyError("Missing parameter in toml model file: "
                           + str(option)) from None

    def build(self, orga, log_comp_opts=False):
        """
        Build the network.

        Input layers will be adapted to the input files in the organizer.
        Can also add the matching modifiers and custom objects to the orga.

        Parameters
        ----------
        orga : object Organizer
            Contains all the configurable options in the OrcaNet scripts.
        log_comp_opts : bool
            If the info used for the compilation of the model should be
            logged to the log.txt.

        Returns
        -------
        model : keras model
            The network.

        """
        input_shapes = orga.io.get_input_shapes()
        custom_objects = orga.cfg.custom_objects

        if self.body_arch == "single":
            # Build a single input network
            if len(input_shapes) is not 1:
                raise ValueError("Invalid input_shape for architecture {}: "
                                 "Has length {}, but must be 1\n input_shapes "
                                 "= {}".format(self.body_arch, len(input_shapes),
                                               input_shapes))
            builder = BlockBuilder(self.body_args, self.head_args)
            model = builder.build(input_shapes, self.body_configs,
                                  self.head_arch, self.head_arch_args)

        elif self.body_arch == "multi":
            # TODO flatten layer has to be added to each individual network
            #  before concatenating. Maybe add a multi option in the attach head
            #  functions?
            """
            # Build networks with an identical body,
            # then concatenate and add head layers
            builder = BlockBuilder(self.body_args, self.head_args)
            mid_layers, input_layers = [], []
            for input_name, input_shape in input_shapes.items():
                model = builder.build({input_name: input_shape},
                                      self.body_configs, head_arch=None)
                assert len(model.inputs) == 1, "model input is not length 1 " \
                                               "{}".format(model.inputs)
                assert len(model.outputs) == 1, "model output is not length 1 " \
                                                "{}".format(model.outputs)
                input_layers.append(model.inputs[0])
                mid_layers.append(model.outputs[0])
            x = Concatenate()(mid_layers)
            output_layer = builder.attach_output_layers(x, self.head_arch,
                                                        **self.head_arch_args)
            model = Model(input_layers, output_layer)
            """
            raise NotImplementedError

        elif self.body_arch == "merge":
            # Concatenate multiple models with a flatten layer to a single big one.
            # block_config is a list of paths
            model_list = []
            for path in self.body_configs:
                model_list.append(ks.models.load_model(path))
            model = self.merge_models(model_list)

        else:
            raise NameError("Unknown architecture: ", self.body_arch)

        """
        if n_gpu is not None:
            model, orga.cfg.batchsize = parallelize_model(model, n_gpu,
                                                          orga.cfg.batchsize)
        """
        self.compile_model(model, custom_objects)
        if log_comp_opts:
            self.log_model_properties(orga)
        model.summary()
        return model

    def merge_models(self, model_list, trainable=False, stateful=True,
                     no_drop=True):
        """
        Concatenate two or more single input cnns to a big one.

        It will explicitly look for a Flatten layer and cut after it,
        Concatenate all models, and then add the head layers.

        Parameters
        ----------
        model_list : list
            List of keras models to stitch together.
        trainable : bool
            Whether the layers of the loaded models will be trainable.
        stateful : bool
            Whether the batchnorms of the loaded models will be stateful.
        no_drop : bool
            If true, rate of dropout layers from loaded models will
            be set to zero.

        Returns
        -------
        model : keras model
            The uncompiled merged keras model.

        """
        # Get the input and Flatten layers in each of the given models
        input_layers, flattens = [], []
        for i, model in enumerate(model_list):
            if len(model.inputs) != 1:
                raise ValueError(
                    "model input is not length 1 {}".format(model.inputs))
            input_layers.append(model.input)
            flatten_found = 0
            for layer in model.layers:
                layer.trainable = trainable
                layer.name = layer.name + '_net_' + str(i)
                if isinstance(layer, BatchNormalization):
                    layer.stateful = stateful
                elif isinstance(layer, Flatten):
                    flattens.append(layer.output)
                    flatten_found += 1
            if flatten_found != 1:
                raise TypeError(
                    "Expected 1 Flatten layer but got " + str(flatten_found))

        # attach new head
        x = Concatenate()(flattens)
        builder = BlockBuilder(body_defaults=None,
                               head_defaults=self.head_args)
        output_layer = builder.attach_output_layers(x, self.head_arch,
                                                    flatten=False,
                                                    **self.head_arch_args)

        model = Model(input_layers, output_layer)
        if no_drop:
            model = change_dropout_rate(model, before_concat=0.)

        return model

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
        if self.optimizer == 'adam':
            optimizer = get_adam(**self.optimizer_args)
        elif self.optimizer == 'sgd':
            optimizer = get_sgd(**self.optimizer_args)
        else:
            raise NameError('Unknown optimizer name ({})'.format(self.optimizer))

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
            loss_metrics[layer_name] = metrics

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


def change_dropout_rate(model, before_concat, after_concat=None):
    """
    Change the dropout rate in a model.

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
        if isinstance(layer, Dropout):
            if concat_found == 0:
                layer.rate = before_concat
                ch_bef += 1
            else:
                layer.rate = after_concat
                ch_aft += 1

        elif isinstance(layer, Concatenate):
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


# def parallelize_model(model, n_gpu, batchsize, mode="avolkov"):
#     """
#     Parallelizes the nn-model to multiple gpu's.
#
#     Currently, up to 4 GPU's at Tiny-GPU are supported.
#
#     Parameters
#     ----------
#     model : ks.model.Model
#         Keras model of a neural network.
#     n_gpu : int
#         Number of gpu's that the model should be parallelized to.
#     batchsize : int
#         Batchsize that is used for the training / inferencing of the cnn.
#     mode : str
#         Avolkov or keras.
#
#     Returns
#     -------
#     model : ks.models.Model
#         The parallelized Keras nn instance (multi_gpu_model).
#     batchsize : int
#         The new batchsize scaled by the number of used gpu's.
#
#     """
#     assert n_gpu > 1 and isinstance(n_gpu, int), 'n_gpu must be an int with n_gpu >= 1!'
#
#     if mode == "avolkov":
#         from orcanet.builder_util.multi_gpu.multi_gpu import (
#             get_available_gpus, make_parallel, print_mgpu_modelsummary)
#
#         gpus_list = get_available_gpus(n_gpu)
#         ngpus = len(gpus_list)
#         print('Using GPUs: {}'.format(', '.join(gpus_list)))
#
#         # Data-Parallelize the model via function
#         model = make_parallel(model, gpus_list, usenccl=False, initsync=True,
#                               syncopt=False, enqueue=False)
#         print_mgpu_modelsummary(model)
#         batchsize = batchsize * ngpus
#
#     elif mode == "keras":
#         # For keras, one has to save the original model, not the saved one...
#         model = ks.utils.multi_gpu_model(model, n_gpu)
#         batchsize *= n_gpu
#
#     else:
#         raise NameError("Unknown mode", mode)
#
#     return model, batchsize
