#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for creating VGG-like models (including VGG-LSTM)"""

import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D, Convolution2D, MaxPooling2D, TimeDistributed, CuDNNLSTM
from keras import backend as K
from keras.regularizers import l2

from utilities.cnn_utilities import get_dimensions_encoding

#------------- VGG-like model -------------#

def decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels, str_ident = ''):
    """
    Returns the general dimension (2D/3D), the input dimensions (i.e. bs x 11 x 13 x 18 x channels=1 for 3D)
    and appropriate max_pool_sizes depending on the projection type.
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    :param int batchsize: Batchsize of the fed data.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :param str str_ident: optional string identifier that specifies the projection.
    :return: int dim: Dimension of the network (2D/3D).
    :return: tuple input_dim: dimensions tuple for 2D, 3D or 4D data (ints).
    :return: dict/dict(dict) max_pool_sizes: Dict that specifies the Pooling locations and properties for the VGG net.
                                  Key: After which convolutional layer a pooling layer be inserted (counts from layer zero!)
                                  Item: Specifies the strides.
    """
    if n_bins[0].count(1) == 1: # 3d case
        dim = 3
        input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
        if n_bins[0][1] == 1:
            print 'Using a VGG-like 3D CNN with XZT projection'
            max_pool_sizes = {1: (1,1,2), 3: (2,2,2), 7: (2,2,2)}

        elif n_bins[0][3] == 1:
            print 'Using a VGG-like 3D CNN with XYZ projection'
            max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}

        elif n_bins[0][0] == 1:
            print 'Using a VGG-like 3D CNN with YZT projection'
            max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}

        else:
            raise IndexError('No suitable 3D projection found in decode_input_dimensions().'
                             'Please add the projection type with the pooling dict to the function.')

    elif n_bins[0].count(1) == 0: # 4d case, 3.5D
        dim = 3
        if len(n_bins) > 1:

            # if swap_4d_channels == 'yzt-x_all-t_and_yzt-x_tight-1-t':
            #     max_pool_sizes = {'net_1': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)},
            #                       'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
            #     input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
            #     input_dim = [(input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]), # yzt-x all
            #                  (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])] # yzt-x tight-1

            if swap_4d_channels == 'xyz-t_and_yzt-x' and 'tight-1_tight-2' not in str_ident:
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)}, # only used if training from scratch
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])] # yzt-x

            elif swap_4d_channels == 'xyz-t_and_yzt-x' and 'multi_input_single_train_tight-1_tight-2' in str_ident:
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)}, # only used if training from scratch
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_3': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_4': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]), # yzt-x
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])] # yzt-x

            elif swap_4d_channels == 'xyz-t_and_yzt-x_and_xyt-z' and str_ident == 'multi_input_single_train_tight-1_tight-2':
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)}, # only used if training from scratch
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_3': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_4': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_5': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]), # yzt-x
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]), # yzt-x
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[4], input_dim[3])] # xyt-z

            elif swap_4d_channels is None:
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_2': {3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t tight-1 w-geo-fix
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4])] # xyz-t tight-1

            else:
                raise IOError('3.5D projection types with len(n_bins) > 1 other than "yzt-x_all-t_and_yzt-x_tight-1-t" are not yet supported.')

        else:
            input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
            if swap_4d_channels is None:
                print 'Using a VGG-like 3.5D CNN with XYZ data and T/C channel information.'
                #max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}
                max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)} # 2 more layers

            elif swap_4d_channels == 'yzt-x':
                print 'Using a VGG-like 3.5D CNN with YZT data and X channel information.'
                #max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}
                max_pool_sizes = {2: (1, 1, 2), 5: (2, 2, 2), 9: (2, 2, 2)} # 2 more layers
                input_dim = (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]) # [bs,y,z,t,x]

            elif swap_4d_channels == 'xyt-z':
                print 'Using a VGG-like 3.5D CNN with XYT data and Z channel information.'
                max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}
                input_dim = (input_dim[0], input_dim[1], input_dim[2], input_dim[4], input_dim[3]) # [bs,y,z,t,x]

            elif swap_4d_channels == 'xyz-t_and_yzt-x':
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])] # yzt-x

            elif swap_4d_channels == 'conv_lstm': # TODO fix whole function to make it more general and not only 3.5D stuff
                max_pool_sizes = {1: (2, 2, 2), 5: (2, 2, 2)}
                input_dim = (input_dim[0], input_dim[4], input_dim[1], input_dim[2], input_dim[3], 1) # t-xyz

            else:
                raise IOError('3.5D projection types other than XYZ-T and YZT-X are not yet supported.'
                              'Please add the max_pool_sizes dict in the function by yourself.')

    else:
        raise IOError('Data types other than 2D, 3D or 4D (3.5D actually) are not yet supported. '
                      'Please specify a 2D, 3D or 4D n_bins tuple.')

    return dim, input_dim, max_pool_sizes


def create_vgg_like_model(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3, swap_4d_channels=None,
                          activation='relu', kernel_reg=None):
    """
    Returns a VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Should only contain one element for this single input net.
    :param int nb_classes: Number of output classes.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param tuple n_filters: Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all dimensions.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :param str activation: Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.
    :param None/str kernel_reg: if L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.
    :return: Model model: Keras VGG-like model.
    """
    if n_filters is None: n_filters = (64,64,64,64,64,128,128,128)
    if kernel_reg is 'l2': kernel_reg = l2(0.0001)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels)

    input_layer = Input(shape=input_dim[1:], dtype=K.floatx())  # input_layer
    x = conv_block(input_layer, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(0), activation=activation, kernel_reg=kernel_reg)

    for i in xrange(1, len(n_filters)):
        x = conv_block(x, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(i), activation=activation, kernel_reg=kernel_reg)

    x = Flatten()(x)
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
    #x = BatchNormalization(axis=-1)(x) # TODO
    x = Activation(activation)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    #x = BatchNormalization(axis=-1)(x)  # TODO
    x = Activation(activation)(x)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def conv_block(ip, dim, n_filters, k_size=3, dropout=0, max_pooling=None, activation='relu', kernel_reg = None):
    """
    2D/3D Convolutional block followed by BatchNorm and Activation with optional MaxPooling or Dropout.
    C-B-A-(MP)-(D)
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int dim: 2D or 3D block.
    :param int n_filters: Number of filters used for the convolution.
    :param int k_size: Kernel size which is used for all three dimensions.
    :param float dropout: Adds a dropout layer if value is greater than 0.
    :param None/tuple max_pooling: Specifies if a MaxPooling layer should be added. e.g. (1,1,2) for 3D.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :param None/str kernel_reg: if L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.
    :return: x: Resulting output tensor (model).
    """
    if dim not in (2,3): raise ValueError('dim must be equal to 2 or 3.')
    convolution_nd = Convolution2D if dim==2 else Convolution3D
    max_pooling_nd = MaxPooling2D if dim==2 else MaxPooling3D

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = convolution_nd(n_filters, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_reg)(ip)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(activation)(x)

    if max_pooling is not None: x = max_pooling_nd(strides=max_pooling, padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    return x


def create_vgg_like_model_double_input(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3, swap_4d_channels=None, activation='relu'):
    """
    Returns a double input, VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
    The two single VGG networks are concatenated after the last flatten layers.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    :param int nb_classes: Number of output classes.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param tuple n_filters: Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all dimensions.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :return: Model model: Keras VGG-like model.
    """
    if n_filters is None: n_filters = (64,64,64,64,64,128,128,128)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels)

    input_layer_1 = Input(shape=input_dim[0][1:], dtype=K.floatx())  # input_layer 1
    input_layer_2 = Input(shape=input_dim[1][1:], dtype=K.floatx())  # input_layer 2

    # Net 1 convs
    x_1 = conv_block(input_layer_1, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_1'].get(0), activation=activation)

    for i in xrange(1, len(n_filters)):
        x_1 = conv_block(x_1, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_1'].get(i), activation=activation)

    # Net 2 convs
    x_2 = conv_block(input_layer_2, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_2'].get(0), activation=activation)

    for i in xrange(1, len(n_filters)):
        x_2 = conv_block(x_2, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_2'].get(i), activation=activation)

    # flatten both nets
    x_1, x_2 = Flatten()(x_1), Flatten()(x_2)

    # concatenate both nets
    x = ks.layers.concatenate([x_1, x_2])

    x = Dense(128, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input_layer_1, input_layer_2], outputs=x)

    return model


def create_vgg_like_model_multi_input_from_single_nns(n_bins, batchsize, str_ident, nb_classes=2, dropout=(0, 0.2), swap_4d_channels=None, activation='relu'):
    """
    Returns a double input, VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
    The two single VGG networks are concatenated after the last flatten layers.
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param str str_ident: optional string identifier that specifies the input projection type.
    :param int nb_classes: Number of output classes.
    :param (float, float) dropout: Adds dropout if >0.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :return: Model model: Keras VGG-like model.
    """
    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels, str_ident=str_ident)
    trained_model_paths = {}

    #if swap_4d_channels + str_ident  == 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1':
    if 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1' in swap_4d_channels + str_ident and 'multi_input_single_train_tight-1_tight-2' not in str_ident:
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_2-more-layers_epoch_32_file_1.h5'  # xyz-t, timecut tight_1, with geo fix, 2 more layers
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-1_w-geo-fix_bs64_dp0.1_2-more-layers_epoch_30_file_1.h5'  # yzt-x, timecut tight-1, with geo fix, 2 more layers

    elif swap_4d_channels is None: # xyz-t tight-1 and xyz-t tight-2
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_epoch_3_file_1.h5'  # xyz-t, timecut tight_1, with geo fix, fully trained epoch 23
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-2_w-geo-fix_bs64_dp0.1_epoch_3_file_1.h5'  # xyz-t, timecut tight-2, with geo fix, fully trained epoch 36

    elif 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1_tight-2' in swap_4d_channels + str_ident:
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_2-more-layers_epoch_32_file_1.h5' # xyz-t, tight-1, w-geo-fix, 2 more layers
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-1_w-geo-fix_bs64_dp0.1_2-more-layers_epoch_31_file_1.h5' # yzt-x, tight-1, w-geo-fix, 2 more layers
        trained_model_paths[2] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-2_w-geo-fix_bs64_2-more-layers_epoch_24_file_1.h5' # xyz-t, tight-2
        trained_model_paths[3] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-2_w-geo-fix_bs64_dp0.1_2-more-layers_epoch_29_file_1.h5' # yzt-x, tight-2

    elif swap_4d_channels + str_ident == 'xyz-t_and_yzt-x_and_xyt-z' + 'multi_input_single_train_tight-1_tight-2':
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_2-more-layers_epoch_32_file_1.h5' # xyz-t, tight-1, w-geo-fix, 2 more layers
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-1_w-geo-fix_bs64_dp0.1_2-more-layers_epoch_17_file_1.h5' # yzt-x, tight-1, w-geo-fix, 2 more layers
        trained_model_paths[2] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-2_w-geo-fix_bs64_dp0.1_epoch_30_file_1.h5' # xyz-t, tight-2
        trained_model_paths[3] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-2_w-geo-fix_bs64_dp0.1_epoch_33_file_1.h5' # yzt-x, tight-2
        trained_model_paths[4] = 'models/trained/trained_model_VGG_4d_xyt-z_muon-CC_to_elec-CC_xyt-z_tight-1_w-geo-fix_bs64_dp0.1_epoch_33_file_1.h5' # xyt-z, tight-1

    else:
        raise ValueError('The double input combination specified in "swap_4d_channels" is not known, check the function for what is available.')

    n_inputs = len(trained_model_paths)
    trained_models = {}
    input_layers = {}
    layer_numbers = {}
    x = {}

    for i in xrange(n_inputs):
        trained_models[i] = ks.models.load_model(trained_model_paths[i])

        input_layers[i] = Input(shape=input_dim[i][1:], name='input_net_' + str(i+1), dtype=K.floatx())
        layer_numbers[i] = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}

        x[i] = create_layer_from_config(input_layers[i], trained_models[i].layers[1], layer_numbers[i], trainable=False, net=str(i+1))
        for trained_layer in trained_models[i].layers[2:]:
            if 'flatten' in trained_layer.name: break  # we don't want to get anything after the flatten layer
            x[i] = create_layer_from_config(x[i], trained_layer, layer_numbers[i], trainable=False, net=str(i+1), dropout=dropout[0])

        x[i] = Flatten()(x[i])

    x = ks.layers.concatenate([x[i] for i in x])

    x = Dense(128, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    x = Dropout(dropout[1])(x)
    x = Dense(32, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input_layers[i] for i in input_layers], outputs=x)
    set_layer_weights(model, trained_models) # set weights

    for layer in model.layers: # freeze trainable batch_norm weights, but not running mean and variance
        if 'batch_norm' in layer.name:
            layer.stateful = True

    return model


# def create_vgg_like_model_double_input_from_single_nns(n_bins, batchsize, str_ident, nb_classes=2, dropout=(0, 0.2), swap_4d_channels=None, activation='relu'):
#     """
#     Returns a double input, VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
#     The two single VGG networks are concatenated after the last flatten layers.
#     :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
#     :param int nb_classes: Number of output classes.
#     :param int batchsize: Batchsize of the data that will be used with the VGG net.
#     :param (float, float) dropout: Adds dropout if >0.
#     :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
#     :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
#     :return: Model model: Keras VGG-like model.
#     """
#     dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels)
#
#     if swap_4d_channels == 'yzt-x_all-t_and_yzt-x_tight-1-t':
#         trained_model_1_path = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_only_new_timecut_dp01_epoch_47_file_1.h5'  # yzt-x, timecut_all, old geo
#         trained_model_2_path = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_new_tight_timecut_250_500_dp01_epoch_34_file_1.h5'  # yzt-x, timecut tight-1, old geo
#
#     elif swap_4d_channels + str_ident  == 'xyz-t_and_yzt-x' + 'double_input_single_train_tight-1':
#         # trained_model_1_path = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_epoch_22_file_1.h5'  # xyz-t, timecut tight_1, with geo fix
#         # trained_model_2_path = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_new_tight_timecut_250_500_dp01_epoch_34_file_1.h5'  # yzt-x, timecut tight-1, old geo
#         # New
#         trained_model_1_path = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_2-more-layers_epoch_19_file_1.h5'  # xyz-t, timecut tight_1, with geo fix, 2 more layers
#         trained_model_2_path = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_new_tight_timecut_250_500_dp01_larger_bs_64_w_geo_fix_epoch_30_file_1.h5'  # yzt-x, timecut tight-1, new geo
#
#     elif swap_4d_channels is None: # xyz-t tight-1 and xyz-t tight-2
#         trained_model_1_path = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_epoch_3_file_1.h5'  # xyz-t, timecut tight_1, with geo fix, fully trained epoch 23
#         trained_model_2_path = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-2_w-geo-fix_bs64_dp0.1_epoch_3_file_1.h5'  # xyz-t, timecut tight-2, with geo fix, fully trained epoch 36
#
#     else:
#         raise ValueError('The double input combination specified in "swap_4d_channels" is not known, check the function for what is available.')
#
#     trained_model_1 = ks.models.load_model(trained_model_1_path)
#     trained_model_2 = ks.models.load_model(trained_model_2_path)
#
#     # model 1
#     input_layer_net_1 = Input(shape=input_dim[0][1:], name='input_net_1', dtype=K.floatx()) # have to do that manually
#     layer_numbers_net_1 = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}
#
#     x_1 = create_layer_from_config(input_layer_net_1, trained_model_1.layers[1], layer_numbers_net_1, trainable=False, net='1')
#
#     for trained_layer in trained_model_1.layers[2:]:
#         if 'flatten' in trained_layer.name: break  # we don't want to get anything after the flatten layer
#         x_1 = create_layer_from_config(x_1, trained_layer, layer_numbers_net_1, trainable=False, net='1', dropout=dropout[0])
#
#     # model 2
#     input_layer_net_2 = Input(shape=input_dim[1][1:], name='input_net_2', dtype=K.floatx()) # change input layer name
#     layer_numbers_net_2 = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}
#
#     x_2 = create_layer_from_config(input_layer_net_2, trained_model_2.layers[1], layer_numbers_net_2, trainable=False, net='2')
#
#     for trained_layer in trained_model_2.layers[2:]:
#         if 'flatten' in trained_layer.name: break # we don't want to get anything after the flatten layer
#         x_2 = create_layer_from_config(x_2, trained_layer, layer_numbers_net_2, trainable=False, net='2', dropout=dropout[0])
#
#     # flatten both nets
#     x_1, x_2 = Flatten()(x_1), Flatten()(x_2)
#
#     # concatenate both nets
#     x = ks.layers.concatenate([x_1, x_2])
#
#     x = Dense(128, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)
#     x = Dropout(dropout[1])(x)
#     x = Dense(16, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)
#
#     x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)
#
#     model = Model(inputs=[input_layer_net_1, input_layer_net_2], outputs=x)
#
#     set_layer_weights(model, trained_model_1, trained_model_2) # set weights
#
#     for layer in model.layers: # freeze trainable batch_norm weights, but not running mean and variance
#         if 'batch_norm' in layer.name:
#             layer.stateful = True
#
#     return model


def create_layer_from_config(x, trained_layer, layer_numbers, trainable=False, net='', dropout=0):
    """
    Creates a new Keras nn layer from the config of an already existing layer.
    Changes the 'trainable' flag of the new layer to false and optionally udates the dropout rate.
    Adds a layer name based on the layer_numbers dict.
    :param x: Keras functional model api instance. E.g. TF tensors.
    :param ks.layer trained_layer: Keras layer instance that is already trained.
    :param dict layer_numbers: dictionary for the different layer types to keep track of the layer_number in the layer names.
    :param bool trainable: flag to set the <trainable> attribute of the new layer.
    :param str net: additional string that is added to the layer name. E.g. 'net_2' if a double input model is used.
    :param float dropout: optional, dropout rate of the new layer
    :return: x: Keras functional model api instance. E.g. TF tensors. Contains a new layer now!
    """
    if 'conv' in trained_layer.name:
        layer, name = Convolution3D, 'conv'
    elif 'batch_norm' in trained_layer.name:
        layer, name = BatchNormalization, 'batch_norm'
    elif 'activation' in trained_layer.name:
        layer, name = Activation, 'activation'
    elif 'pooling' in trained_layer.name:
        layer, name = MaxPooling3D, 'max_pooling'
    elif 'dropout' in trained_layer.name:
        layer, name = Dropout, 'dropout'
    elif 'dense' in trained_layer.name:
        layer, name = Dense, 'dense'
    else:
        return x # if 'input' or 'flatten'

    config = trained_layer.get_config()
    config.update({'trainable': trainable}) # for freezing the layer if wanted
    if name == 'dropout': config.update({'rate': dropout})

    if net == '':
        new_layer_name = name + '_' + str(layer_numbers[name])
    else:
        new_layer_name = name + '_' + str(layer_numbers[name]) + '_net_' + net

    layer_numbers[name] = layer_numbers[name] + 1
    config.update({'name': new_layer_name})

    x = layer.from_config(config)(x)

    return x


def set_layer_weights(model, trained_models):
    """
    Sets the weights of a double input model (until the first flatten layer) based on two pretrained models.
    :param Model model: Keras model instance.
    :param dict trained_models: dict that contains references to the Keras model instances of the already pretrained models.
    """
    skip_layers = ['dropout', 'input', 'dense', 'flatten', 'max_pooling', 'activation', 'concatenate']
    n_models = len(trained_models)
    trained_layers_w_weights = {}

    for i in xrange(n_models):
        trained_layers_w_weights[i] = [layer for layer in trained_models[i].layers if 'conv' in layer.name or 'batch_normalization' in layer.name]  # still ordered

        j = -1
        for layer in model.layers:
            if 'net_' + str(i+1) not in layer.name: continue

            skip = False  # workaround of hell...
            for skip_layer_str in skip_layers:
                if skip_layer_str in layer.name:
                    skip = True
            if skip: continue

            j += 1
            layer.set_weights(trained_layers_w_weights[i][j].get_weights())


def change_dropout_rate_for_double_input_model(n_bins, batchsize, trained_model, dropout=(0.1, 0.1), trainable=(True, True), swap_4d_channels=None):
    """
    Function that rebuilds a keras model and modifies its dropout rate. Workaround, till layer.rate is fixed to work with Dropout layers.
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param ks.models.Model trained_model: Trained Keras model, upon which the dropout rate should be changed.
    :param (float, float) dropout: Adds dropout if > 0. First value for the conv block, second value for the dense.
    :param (bool, bool) trainable: Sets the trainable flag for the conv block layers and for the dense layers.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim. Only used to decode the input_dim.
    :return: Model model: Keras VGG-like model based on the trained_model with modified dropout layers.
    """
    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels)

    # rebuild trained_model based on it's layer config
    input_layer_net_1 = Input(shape=input_dim[0][1:], name='input_net_1', dtype=K.floatx())  # have to do that manually
    input_layer_net_2 = Input(shape=input_dim[1][1:], name='input_net_2', dtype=K.floatx())

    layer_numbers_net_1 = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}
    layer_numbers_net_2 = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}
    layer_names = [layer.name for layer in trained_model.layers]

    x_1, x_2 = input_layer_net_1, input_layer_net_2 # needed for creating the trained_model later on
    for trained_layer in trained_model.layers[2:]:
        if 'net_1' in trained_layer.name:
            x_1 = create_layer_from_config(x_1, trained_layer, layer_numbers_net_1, trainable=trainable[0], net='1', dropout=dropout[0])

        elif 'net_2' in trained_layer.name:
            x_2 = create_layer_from_config(x_2, trained_layer, layer_numbers_net_2, trainable=trainable[0], net='2', dropout=dropout[0])

        else: break

    # flatten both nets and concatenate them
    x_1, x_2 = Flatten()(x_1), Flatten()(x_2)
    x = ks.layers.concatenate([x_1, x_2])

    # rebuild dense layers
    layer_index_first_dense = layer_names.index('concatenate_1') + 1
    layer_numbers_fc = {'dense': 1, 'dropout': 1}

    for trained_layer in trained_model.layers[layer_index_first_dense:]:
        x = create_layer_from_config(x, trained_layer, layer_numbers_fc, trainable=trainable[1], dropout=dropout[1])

    model = Model(inputs=[input_layer_net_1, input_layer_net_2], outputs=x)

    set_layer_weights_from_single_trained_model(model, trained_model) # set weights

    return model


def set_layer_weights_from_single_trained_model(model, trained_model):
    """
    Sets the weights of a Keras model based on an already trained trained_model.
    :param Model model: Keras model instance withought weights.
    :param Model trained_model: Pretrained Keras model used to set the weights for the new model.
    """
    skip_layers = ['dropout', 'input', 'flatten', 'max_pooling', 'activation', 'concatenate']

    i = -1
    for layer in model.layers:
        i += 1

        skip = False # workaround of hell...
        for skip_layer_str in skip_layers:
            if skip_layer_str in layer.name:
                skip = True
        if skip: continue

        layer.set_weights(trained_model.layers[i].get_weights())


def create_convolutional_lstm(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3, activation='relu', kernel_reg=None):
    """
    Returns a VGG-like, convolutional LSTM model (stacked conv. layers + LSTM) with MaxPooling and Dropout if wished.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)
    :param list(tuple) n_bins: Number of bins (x,y,z,t) of the data.
    :param int nb_classes: Number of output classes.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param tuple/None n_filters: Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all dimensions.
    :param str activation: Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.
    :param None/str kernel_reg: if L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.
    :return: Model model: Keras VGG-like model.
    """
    #if n_filters is None: n_filters = (64,64,64,64,64,128,128,128)
    if n_filters is None: n_filters = (32, 32, 64, 64, 64, 64, 128)
    if kernel_reg is 'l2': kernel_reg = l2(0.0001)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, 'conv_lstm') # TODO fix input dim

    input_layer = Input(shape=input_dim[1:], dtype=K.floatx())  # input_layer
    x = conv_block_time_distributed(input_layer, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(0), activation=activation, kernel_reg=kernel_reg)

    for i in xrange(1, len(n_filters)):
        x = conv_block_time_distributed(x, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(i), activation=activation, kernel_reg=kernel_reg)

    x = TimeDistributed(Flatten())(x)

    x = CuDNNLSTM(768)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    x = Activation(activation)(x)
    # #x = BatchNormalization(axis=-1)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    x = Activation(activation)(x)
    # #x = BatchNormalization(axis=-1)(x)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def conv_block_time_distributed(ip, n_filters, k_size=3, dropout=0, max_pooling=None, activation='relu', kernel_reg = None):
    """
    2D/3D Convolutional block followed by BatchNorm and Activation with optional MaxPooling or Dropout.
    C-B-A-(MP)-(D)
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int n_filters: Number of filters used for the convolution.
    :param int k_size: Kernel size which is used for all three dimensions.
    :param float dropout: Adds a dropout layer if value is greater than 0.
    :param None/tuple max_pooling: Specifies if a MaxPooling layer should be added. e.g. (1,1,2) for 3D.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :param None/str kernel_reg: if L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.
    :return: x: Resulting output tensor (model).
    """
    x = TimeDistributed(Convolution3D(n_filters, (k_size,) * 3, padding='same', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_reg))(ip)

    x = TimeDistributed(BatchNormalization(axis=-1))(x)
    x = Activation(activation)(x)

    if max_pooling is not None: x = TimeDistributed(MaxPooling3D(pool_size=max_pooling, padding='valid'))(x)
    if dropout > 0.0: x = TimeDistributed(Dropout(dropout))(x)

    return x


#------------- VGG-like model -------------#









