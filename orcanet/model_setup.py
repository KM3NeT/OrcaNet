#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts for making specific models.
"""

import keras as ks

from model_archs.short_cnn_models import create_vgg_like_model_multi_input_from_single_nns, create_vgg_like_model
from model_archs.wide_resnet import create_wide_residual_network
from utilities.losses import *


def build_nn_model(nn_arch, n_bins, class_type, swap_4d_channels, str_ident):
    """
    Function that builds a Keras nn model of a specific type.

    Parameters
    ----------
    nn_arch : str
        Architecture of the neural network.
    n_bins : list(tuple(int))
        Declares the number of bins for each dimension (e.g. (x,y,z,t)) in the train- and testfiles.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!

    Returns
    -------
    model : ks.models.Model
        A Keras nn instance.

    """
    if nn_arch == 'WRN':
        model = create_wide_residual_network(n_bins[0], nb_classes=class_type[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=swap_4d_channels)

    elif nn_arch == 'VGG':
        if 'multi_input_single_train' in str_ident:
            model = create_vgg_like_model_multi_input_from_single_nns(n_bins, str_ident, nb_classes=class_type[0], dropout=(0,0.1), swap_4d_channels=swap_4d_channels)

        else:
            model = create_vgg_like_model(n_bins, class_type, dropout=0.0,
                                          n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), swap_4d_channels=swap_4d_channels) # 2 more layers

    else: raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')

    return model


def build_or_load_nn_model(epoch, folder_name, nn_arch, n_bins, class_type, swap_4d_channels, str_ident):
    """
    Function that either builds (epoch = 0) a Keras nn model, or loads an existing one from the folder structure.

    Parameters
    ----------
    epoch : tuple(int, int)
        Declares if a previously trained model or a new model (=0) should be loaded, more info in the execute_nn function.
    folder_name : str
        Name of the main folder.
    nn_arch : str
        Architecture of the neural network.
    n_bins : list(tuple(int))
        Declares the number of bins for each dimension (e.g. (x,y,z,t)) in the train- and testfiles.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!

    Returns
    -------
    model : ks.models.Model
        A Keras nn instance.

    """
    if epoch[0] == 0:
        model = build_nn_model(nn_arch, n_bins, class_type, swap_4d_channels, str_ident)
    else:
        path_of_model = folder_name + '/saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
        model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    return model