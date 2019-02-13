#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import os
import keras as ks
import toml

from orcanet.model_archs.short_cnn_models import create_vgg_like_model_multi_input_from_single_nns, create_vgg_like_model
from orcanet.model_archs.wide_resnet import create_wide_residual_network
from orcanet.model_archs.losses import get_all_loss_functions
from orcanet_contrib.contrib import orca_label_modifiers, orca_sample_modifiers, orca_dataset_modifiers


class OrcaModel:
    """
    Class for building models.

    Attributes
    ----------
    compile_opt : dict
        Dict of loss functions and optionally weights & metrics that should be used for each nn output.
        Format: { layer_name : { loss_function:, weight:, metrics: } }
        The loss_function is a string or a function, the weight is a float and metrics is a list of functions/strings.
        Typically read in from a .toml file.
    optimizer : str
        Specifies, if "Adam" or "SGD" should be used as optimizer.
    nn_arch : str
        Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    class_type : str
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower') # TODO outdated docs
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    swap_4d_channels : None or str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X
    kwargs : dict
        Keyword arguments for the model generation.

    """

    def __init__(self, model_file):
        """
        Read out parameters for creating models with OrcaNet from a toml file.

        Parameters
        ----------
        model_file : str
            Path to the model toml file.

        """
        file_content = toml.load(model_file)['model']

        self.nn_arch = file_content.pop('nn_arch')
        self.compile_opt = file_content.pop('compile_opt')
        self.class_type = ''
        self.str_ident = ''
        self.swap_4d_channels = None
        self.optimizer = "adam"
        self.custom_objects = get_all_loss_functions()

        if 'class_type' in file_content:
            self.class_type = file_content.pop('class_type')
        if 'str_ident' in file_content:
            self.str_ident = file_content.pop('str_ident')
        if 'swap_4d_channels' in file_content:
            self.swap_4d_channels = file_content.pop('swap_4d_channels')

        self.kwargs = file_content

    def build(self, orca, n_gpu=None):
        """
        Function that builds a Keras nn model with a specific type of architecture.

        Will adapt to the data given in the orca object, and load in the necessary modifiers.
        Can also parallelize to multiple GPUs.

        Parameters
        ----------
        orca : Object OrcaHandler
            Contains all the configurable options in the OrcaNet scripts.
        n_gpu : int or None
            Number of gpu's that the model should be parallelized to. None for no parallelization.

        Returns
        -------
        model : ks.models.Model
            A Keras nn instance.

        """
        n_bins = orca.io.get_n_bins()
        batchsize = orca.cfg.batchsize

        if self.nn_arch == 'WRN':
            model = create_wide_residual_network(n_bins[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=self.swap_4d_channels)

        elif self.nn_arch == 'VGG':
            if 'multi_input_single_train' in self.str_ident:
                dropout = (0, 0.1)
                model = create_vgg_like_model_multi_input_from_single_nns(n_bins, self.str_ident,
                                                                          dropout=dropout, swap_4d_channels=self.swap_4d_channels)
            else:
                dropout = self.kwargs["dropout"]
                n_filters = self.kwargs["n_filters"]
                model = create_vgg_like_model(n_bins, self.class_type, dropout=dropout,
                                              n_filters=n_filters, swap_col=self.swap_4d_channels)  # 2 more layers
        else:
            raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')

        if n_gpu is not None:
            model, orca.cfg.batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize)
        self._compile_model(model)

        return model

    def update_orca(self, orca):
        """
        Update the orca object for using the model.

        If sth is None, dont update it in the orca. Otherwise, make sure its None
        there before setting anything (to not screw with the orca).

        """
        if self.swap_4d_channels is not None:
            sample_modifier = orca_sample_modifiers(self.swap_4d_channels, self.str_ident)
        else:
            sample_modifier = None
        label_modifier = orca_label_modifiers(self.class_type)

        if sample_modifier is not None:
            assert orca.cfg.sample_modifier is None, "Can not set sample modifier: " \
                                                     "Has already been set: {}".format(orca.cfg.sample_modifier)
        if label_modifier is not None:
            assert orca.cfg.label_modifier is None, "Can not set label modifier: " \
                                                    "Has already been set: {}".format(orca.cfg.label_modifier)
        if self.custom_objects is not None:
            assert orca.cfg.custom_objects is None, "Can not set custom objects: " \
                                                    "Have already been set: {}".format(orca.cfg.custom_objects)

        if sample_modifier is not None:
            orca.cfg.sample_modifier = sample_modifier
        if label_modifier is not None:
            orca.cfg.label_modifier = label_modifier
        if self.custom_objects is not None:
            orca.cfg.custom_objects = self.custom_objects

    def _compile_model(self, model):
        """
        Compile a model with the loss optimizer settings given in the model toml file.

        Returns
        -------
        model : ks.model
            A compile keras model.

        """
        if self.optimizer == 'adam':
            optimizer = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1,
                                           decay=0.0)  # epsilon=1 for deep networks
        elif self.optimizer == 'sgd':
            optimizer = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
        else:
            raise NameError('Unknown optimizer name ({})'.format(self.optimizer))

        loss_functions, loss_weights, loss_metrics = {}, {}, {}
        for layer_name, layer_info in self.compile_opt.items():
            # Replace the str function name with the actual function if it is custom
            loss_function = layer_info['function']
            if loss_function in self.custom_objects:
                loss_function = self.custom_objects[loss_function]
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

        model.compile(loss=loss_functions, optimizer=optimizer, metrics=loss_metrics, loss_weights=loss_weights)
        return model

    def recompile_model(self, orcahandler_instance):
        """
        Compile a loaded keras model once again.

        Parameters
        ----------
        orcahandler_instance : orcanet.core.OrcaHandler
            An instance of the top-level OrcaHandler class.
        Returns
        -------
        recompiled_model : ks.models.Model
            The loaded and recompiled keras model.

        """
        if orcahandler_instance.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        epoch = orcahandler_instance.io.get_latest_epoch()
        path_of_model = orcahandler_instance.io.get_model_path(epoch[0], epoch[1])
        print("Loading saved model: " + path_of_model)
        model = ks.models.load_model(path_of_model, custom_objects=orcahandler_instance.cfg.custom_objects)

        print("Recompiling the saved model")
        recompiled_model = self._compile_model(model)

        return recompiled_model


def parallelize_model_to_n_gpus(model, n_gpu, batchsize, mode="avolkov"):
    """
    Parallelizes the nn-model to multiple gpu's.

    Currently, up to 4 GPU's at Tiny-GPU are supported.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    n_gpu : int
        Number of gpu's that the model should be parallelized to.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    mode : str
        Avolkov or keras.

    Returns
    -------
    model : ks.models.Model
        The parallelized Keras nn instance (multi_gpu_model).
    batchsize : int
        The new batchsize scaled by the number of used gpu's.

    """
    assert n_gpu > 1 and isinstance(n_gpu, int), 'You probably made a typo: n_gpu must be an int with n_gpu >= 1!'

    if mode == "avolkov":
        from orcanet.model_archs.multi_gpu.multi_gpu import get_available_gpus, make_parallel, print_mgpu_modelsummary

        gpus_list = get_available_gpus(n_gpu)
        ngpus = len(gpus_list)
        print('Using GPUs: {}'.format(', '.join(gpus_list)))

        # Data-Parallelize the model via function
        model = make_parallel(model, gpus_list, usenccl=False, initsync=True, syncopt=False, enqueue=False)
        print_mgpu_modelsummary(model)
        batchsize = batchsize * ngpus

    elif mode == "keras":
        # For keras, one has to save the original model, not the saved one...
        model = ks.utils.multi_gpu_model(model, n_gpu)
        batchsize *= n_gpu

    else:
        raise NameError("Unknown mode", mode)

    return model, batchsize

