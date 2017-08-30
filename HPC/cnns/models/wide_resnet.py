# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.
Forked from https://github.com/titu1994/Wide-Residual-Networks/blob/master/wide_residual_network.py

# Reference
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
"""
from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from utilities.cnn_utilities import get_dimensions_encoding


def model_wide_residual_network(n_bins, batchsize, **kwargs):

    if n_bins.count(1) == 2:
        dim = 2
        model = create_wide_residual_network(n_bins, batchsize, dim, **kwargs)

    elif n_bins.count(1) == 1:
        dim = 3
        model = create_wide_residual_network(n_bins, batchsize, dim, **kwargs)

    else:
        raise IOError('Data types other than 2D or 3D are not yet supported. '
                      'Please specify a 2D or 3D n_bins tuple.')

    return model


def decode_input_dimensions(batchsize, n_bins):

    input_dim = get_dimensions_encoding(batchsize, n_bins) # includes batchsize

    # 2d case
    if n_bins[0] == 1 and n_bins[3] == 1 and n_bins.count(1) == 2:
        print 'Using a Wide ResNet with YZ projection'
        strides = [(1,1), (2,2)]
        average_pooling_size = (6,7)

    # 3d case
    elif n_bins[1] == 1 and n_bins.count(1) == 1:
        print 'Using a Wide ResNet with XZT projection'
        strides = [(1, 1, 2), (2, 2, 2)]
        average_pooling_size = (6, 9, 13)

    elif n_bins[3] == 1 and n_bins.count(1) == 1:
        print 'Using a Wide ResNet with XYZ projection'
        strides = [(1,1,1), (2,2,2)]
        average_pooling_size = (6,7,9)

    else:
        raise IndexError('The projection type could not be decoded using the parameter n_bins. '
                         'Please check if your projection is available in the function.')

    return input_dim, strides, average_pooling_size


def create_wide_residual_network(n_bins, batchsize, dim, nb_classes=2, N=2, k=8, dropout=0.0, k_size=3, verbose=True):
    """
    Creates a 2D or 3D Wide Residual Network with specified parameters.
    The torch implementation from the paper differs slightly (change default arguments in Conv and BatchNorm):
    - BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    - Convolution2D(..., use_bias=False).
    :param tuple n_bins: Number of bins (x,y,z,t) of the data that will be fed to the network.
    :param int batchsize: Batchsize of the feeded data.
    :param int nb_classes: Number of output classes.
    :param int N: Depth of the network. Compute N = (n - 4) / 6.
                  Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
                  Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
                  Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param int k: Width of the network (gets multiplied by the number of filters for each convolution).
    :param float dropout: Adds dropout if value is greater than 0.0.
    :param int k_size: Kernel size that should be used, same for each dimension.
    :param bool verbose: Debug info to describe created WRN.
    :return:
    """
    if dim not in (2, 3): # Sanity check
        raise IOError('Data types other than 2D or 3D are not yet supported. '
                      'Please specify a 2D or 3D n_bins tuple.')

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    input_dim, strides, avg_pool_size = decode_input_dimensions(batchsize, n_bins) # includes batchsize
    input_layer = Input(shape=input_dim[1:], dtype=K.floatx()) # batch_shape=input_dim

    x = initial_conv(input_layer, dim, k_size=k_size)
    x = expand_conv(x, 16, dim, k=k, k_size=k_size)
    nb_conv = 10 # 1x initial_conv + 3x expand_conv = 1x1 + 3*3 = 10

    for i in range(N - 1):
        x = conv_block(x, 16, k=k, dropout=dropout, k_size=k_size)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, dim, k=k, strides=strides[0])

    for i in range(N - 1):
        x = conv_block(x, 32, k=k, dropout=dropout, k_size=k_size)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, dim, k=k, strides=strides[1])

    for i in range(N - 1):
        x = conv_block(x, 64, k=k, dropout=dropout, k_size=k_size)
        nb_conv += 2

    if dim == 3: x = AveragePooling3D(avg_pool_size)(x) # use global average pooling instead of fully connected
    if dim == 2: x = AveragePooling2D(avg_pool_size)(x)
    x = Flatten()(x)

    # could also be transformed to one neuron -> binary_crossentropy + sigmoid instead of 2 neurons -> cat._crossentropy + softmax
    x = Dense(nb_classes, activation='softmax')(x) # actually linear in the paper, could also be transformed to one neuron

    model = Model(input_layer, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


def initial_conv(input_layer, dim, k_size=3):
    """
    Initial convolution prior to the ResNet blocks. C-B-A
    :param ks.layers.Input input_layer: Keras Input layer (tensor) that specifies the shape of the input data.
    :param int k_size: Kernel size that should be used.
    :return: x: Resulting output tensor (model).
    """
    if dim not in (2,3): raise ValueError('dim must be equal to 2 or 3.')

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    if dim == 3: x = Convolution3D(16, (k_size, k_size, k_size), padding='same', kernel_initializer='he_normal')(input_layer)
    if dim == 2: x = Convolution2D(16, (k_size, k_size), padding='same', kernel_initializer='he_normal')(input_layer)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, n_filters, dim, k=1, k_size=3, strides=(1, 1, 1)):
    """
    Intermediate convolution block that expands the number of features (increase number of filters, i.e. 16->32).
    (C-B-A-C) + (C) -> M
    :param init: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int n_filters: Number of filters used for each convolution.
    :param int k: Width of the convolution, multiplicative factor to "n_filters".
    :param int k_size: Kernel size which is used for all three dimensions.
    :param tuple(int) strides: Strides of the convolutional layers.
    :return: m: Keras functional layer instance where the last layer is the merging.
    """
    if dim not in (2, 3): raise ValueError('dim must be equal to 2 or 3.')

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    if dim == 3: x = Convolution3D(n_filters * k, (k_size, k_size, k_size), padding='same', strides=strides, kernel_initializer='he_normal')(init)
    if dim == 2: x = Convolution2D(n_filters * k, (k_size, k_size), padding='same', strides=strides, kernel_initializer='he_normal')(init)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dim == 3:
        x = Convolution3D(n_filters * k, (k_size, k_size, k_size), padding='same', kernel_initializer='he_normal')(x)
        skip = Convolution3D(n_filters * k, (k_size, k_size, k_size), padding='same', strides=strides,
                             kernel_initializer='he_normal')(init)
    if dim == 2:
        x = Convolution2D(n_filters * k, (k_size, k_size), padding='same', kernel_initializer='he_normal')(x)
        skip = Convolution2D(n_filters * k, (k_size, k_size), padding='same', strides=strides, kernel_initializer='he_normal')(init)

    m = Add()([x, skip])

    return m


def conv_block(ip, n_filters, dim, k=1, dropout=0.0, k_size=3):
    """
    ResNet block. (B-A-C-B-A-C) + (ip) = M
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int n_filters: Number of filters used for each convolution.
    :param int k: Width of the convolution, multiplicative factor to "n_filters".
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all three dimensions.
    :return: m: Keras functional layer instance where the last layer is the merging.
    """
    if dim not in (2, 3): raise ValueError('dim must be equal to 2 or 3.')

    init = ip # TODO useless?

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(ip)
    x = Activation('relu')(x)
    if dim == 3: x = Convolution3D(n_filters * k, (k_size, k_size, k_size), padding='same', kernel_initializer='he_normal')(x)
    if dim == 2: x = Convolution2D(n_filters * k, (k_size, k_size), padding='same', kernel_initializer='he_normal')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    if dim == 3: x = Convolution3D(n_filters * k, (k_size, k_size, k_size), padding='same', kernel_initializer='he_normal')(x)
    if dim == 2: x = Convolution2D(n_filters * k, (k_size, k_size), padding='same', kernel_initializer='he_normal')(x)

    m = Add()([init, x])
    return m


#-------- only legacy code from here on --------
#old
def conv1_block(ip, k=1, dropout=0.0, k_size=3):
    init = ip # TODO useless?

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(ip)
    x = Activation('relu')(x)
    x = Convolution3D(16 * k, (k_size, k_size, k_size), padding='same')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution3D(16 * k, (k_size, k_size, k_size), padding='same')(x)

    m = Add()([init, x])
    return m

#old
def conv2_block(ip, k=1, dropout=0.0, k_size=3):
    init = ip
    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(ip)
    x = Activation('relu')(x)
    x = Convolution3D(32 * k, (k_size, k_size, k_size), padding='same')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution3D(32 * k, (k_size, k_size, k_size), padding='same')(x)

    m = Add()([init, x])
    return m

#old
def conv3_block(ip, k=1, dropout=0.0, k_size=3):
    init = ip
    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(ip)
    x = Activation('relu')(x)
    x = Convolution3D(64 * k, (k_size, k_size, k_size), padding='same')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution3D(64 * k, (k_size, k_size, k_size), padding='same')(x)

    m = Add()([init, x])
    return m