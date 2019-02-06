#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import keras as ks

from orcanet.model_archs.short_cnn_models import create_vgg_like_model_multi_input_from_single_nns, create_vgg_like_model
from orcanet.model_archs.wide_resnet import create_wide_residual_network
from orcanet.utilities.losses import get_all_loss_functions


def parallelize_model_to_n_gpus(model, n_gpu, batchsize):
    """
    Parallelizes the nn-model to multiple gpu's.

    Currently, up to 4 GPU's at Tiny-GPU are supported.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.

    Returns
    -------
    model : ks.models.Model
        The parallelized Keras nn instance (multi_gpu_model).
    batchsize : int
        The new batchsize scaled by the number of used gpu's.

    """
    if n_gpu[1] == 'avolkov':
        if n_gpu[0] == 1:
            return model, batchsize
        else:
            assert n_gpu[0] > 1 and isinstance(n_gpu[0], int), 'You probably made a typo: n_gpu must be an int with n_gpu >= 1!'
            
            from utilities.multi_gpu.multi_gpu import get_available_gpus, make_parallel, print_mgpu_modelsummary

            gpus_list = get_available_gpus(n_gpu[0])
            ngpus = len(gpus_list)
            print('Using GPUs: {}'.format(', '.join(gpus_list)))
            batchsize = batchsize * ngpus

            # Data-Parallelize the model via function
            model = make_parallel(model, gpus_list, usenccl=False, initsync=True, syncopt=False, enqueue=False)
            print_mgpu_modelsummary(model)

            return model, batchsize

    else:
        raise ValueError('Currently, no multi_gpu mode other than "avolkov" is available.')


def get_optimizer_info(compile_opt, optimizer='adam'):
    """
    Returns optimizer information for the training procedure.

    Parameters
    ----------
    compile_opt : dict
        Dict of loss functions and optionally weights & metrics that should be used for each nn output.
        Format: { layer_name : { loss_function:, weight:, metrics: } }
        The loss_function is a string or a function, the weight is a float and metrics is a list of functions/strings.
        Typically read in from a .toml file.
    optimizer : str
        Specifies, if "Adam" or "SGD" should be used as optimizer.

    Returns
    -------
    loss_functions : dict
        Dict with a loss function for each output, that is used for the loss argument of the Keras compile method.
    loss_metrics : dict
        Dict with a metrics list for each output, that is used for the metrics argument of the Keras compile method.
    loss_weights : dict
        Dict with a weight for each output, that is used for the weights argument of the Keras compile method.
    optimizer : ks.optimizers
        Keras optimizer instance, currently either "Adam" or "SGD".

    """
    if optimizer == 'adam':
        optimizer = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)  # epsilon=1 for deep networks
    elif optimizer == 'sgd':
        optimizer = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
    else:
        raise NameError('Unknown optimizer name ({})'.format(optimizer))
    custom_objects = get_all_loss_functions()

    loss_functions, loss_weights, loss_metrics = {}, {}, {}
    for layer_name in compile_opt.keys():

        # Replace function string with the actual function if it is custom
        if compile_opt[layer_name]['function'] in custom_objects:
            loss_function = custom_objects[compile_opt[layer_name]['function']]
        else:
            loss_function = compile_opt[layer_name]['function']

        # Use given weight, else use default weight of 1
        if 'weight' in compile_opt[layer_name]:
            weight = compile_opt[layer_name]['weight']
        else:
            weight = 1.0

        # Use given metrics, else use no metrics
        if 'metrics' in compile_opt[layer_name]:
            metrics = compile_opt[layer_name]['metrics']
        else:
            metrics = []

        loss_functions[layer_name] = loss_function
        loss_weights[layer_name] = weight
        loss_metrics[layer_name] = metrics

    return loss_functions, loss_metrics, loss_weights, optimizer


def build_nn_model(cfg):
    """
    Function that builds a Keras nn model with a specific type of architecture. Can also parallelize to multiple GPUs.

    Parameters
    ----------
    cfg : Object Configuration
        Contains all the configurable options in the OrcaNet scripts.
    Returns
    -------
    model : ks.models.Model
        A Keras nn instance.

    """
    modeldata = cfg.get_modeldata()
    assert modeldata is not None, "You need to specify modeldata before building a model with OrcaNet!"

    nn_arch = modeldata.nn_arch
    compile_opt = modeldata.compile_opt
    args = modeldata.args
    class_type = modeldata.class_type
    str_ident = modeldata.str_ident
    swap_col = modeldata.swap_4d_channels

    n_bins = cfg.get_n_bins()

    if nn_arch == 'WRN':
        model = create_wide_residual_network(n_bins[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=swap_col)

    elif nn_arch == 'VGG':
        if 'multi_input_single_train' in str_ident:
            dropout = (0, 0.1)
            model = create_vgg_like_model_multi_input_from_single_nns(n_bins, str_ident,
                                                                      dropout=dropout, swap_4d_channels=swap_col)
        else:
            dropout = args["dropout"]
            n_filters = args["n_filters"]
            model = create_vgg_like_model(n_bins, class_type, dropout=dropout,
                                          n_filters=n_filters, swap_col=swap_col)  # 2 more layers
    else:
        raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')

    loss_functions, loss_metrics, loss_weights, optimizer = get_optimizer_info(compile_opt, optimizer='adam')

    model, batchsize = parallelize_model_to_n_gpus(model, cfg.n_gpu, cfg.batchsize)

    model.compile(loss=loss_functions, optimizer=optimizer, metrics=loss_metrics, loss_weights=loss_weights)

    return model
