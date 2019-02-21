# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.
Loosely based on https://github.com/titu1994/Wide-Residual-Networks/blob/master/wide_residual_network.py

# Reference
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
"""
from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, Convolution3D, \
    AveragePooling2D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from scraps.old_model_scripts.file_dump import get_dimensions_encoding


def decode_input_dimensions(n_bins, batchsize, swap_4d_channels):
    """
    Returns the input dimensions (e.g. batchsize x 11 x 13 x 18 x Channels for 3D)
    and appropriate strides and avg_pooling sizes depending on the projection type.
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
    :param int batchsize: Batchsize of the fed data.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :return: tuple input_dim: dimensions tuple for 2D, 3D or 4D data (ints).
    :return: [tuple, tuple] strides: Strides of the WRN.
    :return: tuple average_pooling_size: Avg_pooling size of the WRN.
    """
    input_dim = get_dimensions_encoding(n_bins, batchsize) # includes batchsize

    if n_bins.count(1) == 2: # 2d case
        dim = 2

        if n_bins[0] == 1 and n_bins[3] == 1:
            print('Using a Wide ResNet with YZ projection')
            strides = [(1,1), (1,1), (2,2)]
            average_pooling_size = (7,9)

        elif n_bins[0] == 1 and n_bins[1] == 1:
            print('Using a Wide ResNet with ZT projection')
            strides = [(1,1), (1,2), (2,2)]
            average_pooling_size = (9,13)

        else:
            raise IndexError('No suitable 2D projection found in decode_input_dimensions().'
                             'Please add the projection type with the strides and the average_pooling_size to the function.')

    elif n_bins.count(1) == 1: # 3d case
        dim = 3

        if n_bins[1] == 1:
            print('Using a Wide ResNet with XZT projection')
            strides = [(1,1,1), (1,1,2), (2,2,2)]
            average_pooling_size = (6,9,13)

        elif n_bins[3] == 1:
            print('Using a Wide ResNet with XYZ projection')
            strides = [(1,1,1), (1,1,1), (2,2,2)]
            average_pooling_size = (6,7,9)

        elif n_bins[0] == 1:
            print('Using a Wide ResNet with YZT projection')
            strides = [(1,1,1), (1,1,2), (2,2,2)]
            average_pooling_size = (7,9,13)

        else:
            raise IndexError('No suitable 3D projection found in decode_input_dimensions().'
                             'Please add the projection type with the strides and the average_pooling_size to the function.')

    elif n_bins.count(1) == 0:
        dim = 3

        if swap_4d_channels is None:
            print('Using a WRN 3.5D CNN with XYZ data and T channel information.')
            #strides = [(1,1,1), (1,1,1), (2,2,2)]
            strides = [(1, 1, 1), (2, 2, 2)]
            average_pooling_size = (6,7,9)

        elif swap_4d_channels == 'yzt-x':
            print('Using a WRN 3.5D CNN with YZT data and X channel information.')
            #strides = [(1,1,1), (1,1,2), (2,2,2)]
            strides = [(1,1,2), (2,2,2)]
            average_pooling_size = (7,9,13)
            input_dim = (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]) # [bs,y,z,t,x]

        else:
            raise IOError('3.5D projection types other than XYZ-T and YZT-X are not yet supported.'
                          'Please add the max_pool_sizes dict in the function by yourself.')

    else:
        raise IOError('Data types other than 2D, 3D or 4D (3.5D actually) are not yet supported. '
                      'Please specify a 2D, 3D or 4D n_bins tuple.')

    return dim, input_dim, strides, average_pooling_size


def create_wide_residual_network(n_bins, n=1, k=8, dropout=0.0, k_size=3, verbose=True, swap_4d_channels=None):
    """
    Creates a 2D or 3D Wide Residual Network with specified parameters.
    The torch implementation from the paper differs slightly (change default arguments in BatchNorm):
    - BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    :param tuple n_bins: Number of bins (x,y,z,t) of the data that will be fed to the network.
    :param int nb_classes: Number of output classes.
    :param int n: Depth of the network. Compute n = (N - 4) / 6.
                  Example : For a depth of 16, N = 16, n = (16 - 4) / 6 = 2
                  Example2: For a depth of 28, N = 28, n = (28 - 4) / 6 = 4
                  Example3: For a depth of 40, N = 40, n = (40 - 4) / 6 = 6
    :param int k: Width of the network (gets multiplied by the number of filters for each convolution).
    :param float dropout: Adds dropout if value is greater than 0.0.
    :param int k_size: Kernel size that should be used, same for each dimension.
    :param bool verbose: Debug info to describe the created WRN.
    :param None/str swap_4d_channels: For 3.5D nets, specifies if the default channel (t) should be swapped with another dim.
    :return: Model model: Keras WRN model.
    """
    nb_classes = 2
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    # TODO Batchsize has to be given to decode_input_dimensions_vgg, but is not used for constructing the model.
    # For now: Just use some random value.
    batchsize = 64

    dim, input_dim, strides, avg_pool_size = decode_input_dimensions(n_bins, batchsize, swap_4d_channels)  # includes batchsize
    average_pooling_nd = AveragePooling2D if dim==2 else AveragePooling3D

    input_layer = Input(shape=input_dim[1:], dtype=K.floatx()) # batch_shape=input_dim

    #x = BatchNormalization(axis=channel_axis)(input_layer) # can be used after input in order to omit mean substraction. Care: Change initial conv param to x
    x = initial_conv(input_layer, 64, dim, dropout=dropout, k_size=k_size, channel_axis=channel_axis)
    x = expand_conv(x, 64, dim, k=k, dropout=dropout, k_size=k_size, strides=strides[0], channel_axis=channel_axis)
    nb_conv = 10 # 1x initial_conv + 3x expand_conv = 1x1 + 3*3 = 10

    for i in range(n):
        x = conv_block(x, 64, dim, k=k, dropout=dropout, k_size=k_size, channel_axis=channel_axis)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = expand_conv(x, 128, dim, k=k, dropout=dropout, strides=strides[1], channel_axis=channel_axis)

    for i in range(n):
        x = conv_block(x, 128, dim, k=k, dropout=dropout, k_size=k_size, channel_axis=channel_axis)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # x = expand_conv(x, 64, dim, k=k, dropout=dropout, strides=strides[2], channel_axis=channel_axis)
    #
    # for i in range(n):
    #     x = conv_block(x, 64, dim, k=k, dropout=dropout, k_size=k_size, channel_axis=channel_axis)
    #     nb_conv += 2
    #
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = average_pooling_nd(avg_pool_size)(x) # use global average pooling instead of fully connected
    x = Flatten()(x)

    # could also be transformed to one neuron -> binary_crossentropy + sigmoid instead of 2 neurons -> cat._crossentropy + softmax
    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x) # actually linear in the paper, could also be transformed to one neuron

    model = Model(input_layer, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


def initial_conv(input_layer,n_filters, dim, dropout=0.0, k_size=3, channel_axis=-1):
    """
    Initial convolution prior to the ResNet blocks (2D/3D).
    C-B-A
    :param ks.layers.Input input_layer: Keras Input layer (tensor) that specifies the shape of the input data.
    :param int n_filters: Number of filters used for the initial convolution.
    :param int dim: 2D or 3D block.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size that should be used.
    :param int channel_axis: the channel axis that the BatchNorm layer should be applied on.
    :return: x: Resulting output tensor (model).
    """
    if dim not in (2,3): raise ValueError('dim must be equal to 2 or 3.')
    convolution_nd = Convolution2D if dim==2 else Convolution3D

    x = convolution_nd(n_filters, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False)(input_layer) # TODO probably more filters, standard=16

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    return x


def expand_conv(ip, n_filters, dim, k=1, dropout=0.0, k_size=3, strides=None, channel_axis=-1):
    """
    Intermediate convolution block that expands the number of features (increase number of filters, i.e. 16->32).
    2D/3D.
    (C-B-A-C) + (C) -> M
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int n_filters: Number of filters used for each convolution.
    :param int dim: 2D or 3D block.
    :param int k: Width of the convolution, multiplicative factor to "n_filters".
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all three dimensions.
    :param tuple(int) strides: Strides of the convolutional layers.
    :param int channel_axis: the channel axis that the BatchNorm layer should be applied on.
    :return: m: Keras functional layer instance where the last layer is the merge.
    """
    if dim not in (2, 3): raise ValueError('dim must be equal to 2 or 3.')
    if strides is None: strides = (1,) * dim
    convolution_nd = Convolution2D if dim == 2 else Convolution3D

    x = convolution_nd(n_filters * k, (k_size,) * dim, padding='same', strides=strides, kernel_initializer='he_normal', use_bias=False)(ip)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = convolution_nd(n_filters * k, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    skip = convolution_nd(n_filters * k, (1,) * dim, padding='same', strides=strides, kernel_initializer='he_normal', use_bias=False)(ip)

    if dropout > 0.0: skip = Dropout(dropout)(skip) #

    m = Add()([x, skip])

    return m


def conv_block(ip, n_filters, dim, k=1, dropout=0.0, k_size=3, channel_axis=-1):
    """
    ResNet block (2D/3D).
    (B-A-C-B-A-C) + (ip) = M
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int n_filters: Number of filters used for each convolution.
    :param int dim: 2D or 3D block.
    :param int k: Width of the convolution, multiplicative factor to "n_filters".
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all three dimensions.
    :param int channel_axis: the channel axis that the BatchNorm layer should be applied on.
    :return: m: Keras functional layer instance where the last layer is the merge.
    """
    if dim not in (2, 3): raise ValueError('dim must be equal to 2 or 3.')
    convolution_nd = Convolution2D if dim == 2 else Convolution3D

    init = ip # TODO useless?

    x = BatchNormalization(axis=channel_axis)(ip)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x) ##

    x = convolution_nd(n_filters * k, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = convolution_nd(n_filters * k, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    if dropout > 0.0: init = Dropout(dropout)(init) #

    m = Add()([init, x])
    return m