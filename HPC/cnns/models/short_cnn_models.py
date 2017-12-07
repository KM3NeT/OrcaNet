#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D, Convolution2D, MaxPooling2D
from keras import backend as K
from keras.regularizers import l2

from utilities.cnn_utilities import get_dimensions_encoding

#------------- VGG-like model -------------#

def decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels):
    """
    Returns the general dimension (2D/3D), the input dimensions (i.e. bs x 11 x 13 x 18 x channels=1 for 3D)
    and appropriate max_pool_sizes depending on the projection type.
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
    :param int batchsize: Batchsize of the fed data.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :return: int dim: Dimension of the network (2D/3D).
    :return: tuple input_dim: dimensions tuple for 2D, 3D or 4D data (ints).
    :return: dict/dict(dict) max_pool_sizes: Dict that specifies the Pooling locations and properties for the VGG net.
                                  Key: After which convolutional layer a pooling layer be inserted (counts from layer zero!)
                                  Item: Specifies the strides.
    """
    input_dim = get_dimensions_encoding(n_bins, batchsize) # includes batchsize

    if n_bins.count(1) == 2: # 2d case
        dim = 2

        if n_bins[0] == 1 and n_bins[3] == 1:
            print 'Using a VGG-like 2D CNN with YZ projection'
            max_pool_sizes = {3: (2,2), 7: (2,2)}

        else:
            raise IndexError('No suitable 2D projection found in decode_input_dimensions().'
                             'Please add the projection type with the pooling dict to the function.')

    elif n_bins.count(1) == 1: # 3d case
        dim = 3

        if n_bins[1] == 1:
            print 'Using a VGG-like 3D CNN with XZT projection'
            max_pool_sizes = {1: (1,1,2), 3: (2,2,2), 7: (2,2,2)}

        elif n_bins[3] == 1:
            print 'Using a VGG-like 3D CNN with XYZ projection'
            max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}

        elif n_bins[0] == 1:
            print 'Using a VGG-like 3D CNN with YZT projection'
            max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}

        else:
            raise IndexError('No suitable 3D projection found in decode_input_dimensions().'
                             'Please add the projection type with the pooling dict to the function.')

    elif n_bins.count(1) == 0:
        dim = 3

        if swap_4d_channels is None:
            print 'Using a VGG-like 3.5D CNN with XYZ data and T channel information.'
            max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}

        elif swap_4d_channels == 'yzt-x':
            print 'Using a VGG-like 3.5D CNN with YZT data and X channel information.'
            max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}
            input_dim = (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]) # [bs,y,z,t,x]

        elif swap_4d_channels == 'xyz-t_and_yzt-x':
            max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},
                              'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
            input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                         (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])] # yzt-x

        elif swap_4d_channels == 'xyz-t_and_tyz-x':
            max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},
                              'net_2': {1: (2, 1, 1), 3: (2, 2, 2), 7: (6, 2, 2)}}
            input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]), # xyz-t
                         (input_dim[0], input_dim[4], input_dim[2], input_dim[3], input_dim[1])] # tyz-x

        else:
            raise IOError('3.5D projection types other than XYZ-T and YZT-X are not yet supported.'
                          'Please add the max_pool_sizes dict in the function by yourself.')

    else:
        raise IOError('Data types other than 2D, 3D or 4D (3.5D actually) are not yet supported. '
                      'Please specify a 2D, 3D or 4D n_bins tuple.')

    return dim, input_dim, max_pool_sizes


def create_vgg_like_model(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3, swap_4d_channels=None,
                          activation='relu', kernel_reg = None):
    """
    Returns a VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
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
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    x = Activation(activation)(x)
    #x = BatchNormalization(axis=-1)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    x = Activation(activation)(x)
    #x = BatchNormalization(axis=-1)(x)

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
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
    :param int nb_classes: Number of output classes.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param tuple n_filters: Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all dimensions.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :param str activation: Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    :return: Model model: Keras VGG-like model.
    """
    if n_filters is None: n_filters = (64,64,64,64,64,128,128,128) #TODO change n_filters maybe

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
    x_1 = Flatten()(x_1)
    x_2 = Flatten()(x_2)

    # concatenate both nets
    x = ks.layers.concatenate([x_1, x_2])

    x = Dense(256, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, activation=activation, kernel_initializer='he_normal')(x) #bias_initializer=ks.initializers.Constant(value=0.1)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input_layer_1, input_layer_2], outputs=x)

    return model


#------------- VGG-like model -------------#








