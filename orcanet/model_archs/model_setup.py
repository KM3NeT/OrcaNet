#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import keras as ks

from orcanet.model_archs.short_cnn_models import create_vgg_like_model_multi_input_from_single_nns, create_vgg_like_model
from orcanet.model_archs.wide_resnet import create_wide_residual_network
from orcanet.utilities.losses import get_all_loss_functions
from orcanet_contrib.contrib import orca_label_modifiers, orca_sample_modifiers


def parallelize_model_to_n_gpus(model, n_gpu, batchsize, loss_functions, optimizer, metrics, loss_weight):
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
    loss_functions : dict/str
        Dict/str with loss functions that should be used for each nn output. # TODO fix, make single loss func also use dict
    optimizer : ks.optimizers
        Keras optimizer instance, currently either "Adam" or "SGD".
    metrics : dict/str/None
        Dict/str with metrics that should be used for each nn output.
    loss_weight : dict/None
        Dict with loss weights that should be used for each sub-loss.

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

            model.compile(loss=loss_functions, optimizer=optimizer, metrics=metrics, loss_weights=loss_weight)  # TODO check if necessary

            return model, batchsize

    else:
        raise ValueError('Currently, no multi_gpu mode other than "avolkov" is available.')


def get_optimizer_info(loss_opt, optimizer='adam'):
    """
    Returns optimizer information for the training procedure.

    Parameters
    ----------
    loss_opt : tuple
        loss_opt[0]: dict
            Dicts of loss functions and optionally weights that should be used for each nn output.
            Typically read in from a .toml file.
            Format: { layer_name : { loss_function, weight } }
        loss_opt[1]: dict or None
            Metrics that should be used for each nn output.
    optimizer : str
        Specifies, if "Adam" or "SGD" should be used as optimizer.

    Returns
    -------
    loss_functions : dict
        Cf. loss_opt[0].
    metrics : dict
        Cf. loss_opt[1].
    loss_weight : dict
        Cf. loss_opt[2].
    optimizer : ks.optimizers
        Keras optimizer instance, currently either "Adam" or "SGD".

    """
    if optimizer == 'adam':
        optimizer = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)  # epsilon=1 for deep networks
    elif optimizer == "sgd":
        optimizer = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
    else:
        raise NameError("Unknown optimizer name ({})".format(optimizer))
    custom_objects = get_all_loss_functions()

    loss_functions, loss_weight = {}, {}
    for layer_name in loss_opt[0].keys():
        # Replace function string with the actual function if it is custom
        if loss_opt[0][layer_name]["function"] in custom_objects:
            loss_function = custom_objects[loss_opt[0][layer_name]["function"]]
        else:
            loss_function = loss_opt[0][layer_name]["function"]
        # Use given weight, else use default weight of 1
        if "weight" in loss_opt[0][layer_name]:
            weight = loss_opt[0][layer_name]["weight"]
        else:
            weight = 1.0
        loss_functions[layer_name] = loss_function
        loss_weight[layer_name] = weight

    metrics = loss_opt[1] if not loss_opt[1] is None else []

    return loss_functions, metrics, loss_weight, optimizer


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
    loss_opt = modeldata.loss_opt
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

    loss_functions, metrics, loss_weight, optimizer = get_optimizer_info(loss_opt, optimizer='adam')
    model, batchsize = parallelize_model_to_n_gpus(model, cfg.n_gpu, cfg.batchsize, loss_functions, optimizer, metrics,
                                                   loss_weight)
    model.compile(loss=loss_functions, optimizer=optimizer, metrics=metrics, loss_weights=loss_weight)
    if swap_col is not None:
        cfg.sample_modifier = orca_sample_modifiers(swap_col, str_ident)
    cfg.label_modifier = orca_label_modifiers(class_type)

    return model
