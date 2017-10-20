#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D, Convolution2D, MaxPooling2D
from keras import backend as K

from utilities.cnn_utilities import get_dimensions_encoding

#------------- VGG-like model -------------#

def decode_input_dimensions_vgg(n_bins, batchsize):
    """
    Returns the general dimension (2D/3D), the input dimensions (i.e. bs x 11 x 13 x 18 x channels=1 for 3D)
    and appropriate max_pool_sizes depending on the projection type.
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
    :param int batchsize: Batchsize of the fed data.
    :return: int dim: Dimension of the network (2D/3D).
    :return: tuple input_dim: dimensions tuple for 2D, 3D or 4D data (ints).
    :return: dict max_pool_sizes: Dict that specifies the Pooling locations and properties for the VGG net.
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

    else:
        raise IOError('Data types other than 2D or 3D are not yet supported. '
                      'Please specify a 2D or 3D n_bins tuple.')

    return dim, input_dim, max_pool_sizes


def create_vgg_like_model(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3):
    """
    Returns a VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)
    :param tuple n_bins: Number of bins (x,y,z,t) of the data.
    :param int nb_classes: Number of output classes.
    :param int batchsize: Batchsize of the data that will be used with the VGG net.
    :param tuple n_filters: Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    :param float dropout: Adds dropout if >0.
    :param int k_size: Kernel size which is used for all three dimensions.
    :return: Model model: Keras VGG-like model.
    """
    if n_filters is None: n_filters = (64,64,64,64,64,128,128,128)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize)

    input_layer = Input(shape=input_dim[1:], dtype=K.floatx())  # input_layer
    x = conv_block(input_layer, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(0))

    for i in xrange(1, len(n_filters)):
        x = conv_block(x, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(i))

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, activation='relu')(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def conv_block(ip, dim, n_filters, k_size=3, dropout=0, max_pooling=None):
    """
    2D/3D Convolutional block followed by BatchNorm and Activation with optional MaxPooling or Dropout.
    C-B-A-(MP)-(D)
    :param ip: Keras functional layer instance that is used as the starting point of this convolutional block.
    :param int dim: 2D or 3D block.
    :param int n_filters: Number of filters used for the convolution.
    :param int k_size: Kernel size which is used for all three dimensions.
    :param float dropout: Adds a dropout layer if value is greater than 0.
    :param None/tuple max_pooling: Specifies if a MaxPooling layer should be added. e.g. (1,1,2) for 3D.
    :return: x: Resulting output tensor (model).
    """
    if dim not in (2,3): raise ValueError('dim must be equal to 2 or 3.')
    convolution_nd = Convolution2D if dim==2 else Convolution3D
    max_pooling_nd = MaxPooling2D if dim==2 else MaxPooling3D

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = convolution_nd(n_filters, (k_size,) * dim, padding='same', kernel_initializer='he_normal', use_bias=False)(ip)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if max_pooling is not None: x = max_pooling_nd(strides=max_pooling, padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    return x

#------------- VGG-like model -------------#


#------------- Undocumented legacy tests -------------#

def define_3d_model_xyz_test(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    dropout_val = 0.1

    model = ks.models.Sequential()
    model.add(Dense(512, input_shape=(n_bins[0], n_bins[1], n_bins[2], 1) ,activation="relu"))
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[0], n_bins[1], n_bins[2], 1), padding="same"))
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(dropout_val))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    #model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Dropout(dropout_val))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    #model.add(Dense(128, activation="relu"))
    #model.add(Dense(64, activation="relu"))
    #model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='sigmoid'))

    return model

def define_3d_model_xyz(number_of_classes, n_bins, dropout=0):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    #dropout_val = 0.1

    model = ks.models.Sequential()
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[0], n_bins[1], n_bins[2], 1),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    #model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(dropout))
    #model.add(Dropout(dropout_val))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model

    #if number_parallel > 1:
    #	model = make_parallel(model, number_parallel)

    # model.add(layers.core.Permute((4,3,2,1)))
    # model.add(layers.core.Reshape((128, 8)))

def define_3d_model_xzt(number_of_classes, n_bins, dropout=0):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3


    model = ks.models.Sequential()
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[0], n_bins[2], n_bins[3], 1),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model


def define_2d_model_yz(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    dropout_val = 0

    model = ks.models.Sequential()
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", input_shape=(n_bins[1], n_bins[2], 1),
                            padding="same", kernel_initializer='he_normal'))
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(normal.BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model


def define_2d_model_yz_test(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    dropout_val = 0

    model = ks.models.Sequential()
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", input_shape=(n_bins[1], n_bins[2], 1),
                            padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization()) # set use_bias=False
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal'))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model

def define_2d_model_yz_test_batch_norm(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    dropout_val = 0.1

    model = ks.models.Sequential()
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", input_shape=(n_bins[1], n_bins[2], 1),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model


def define_2d_model_zt_test_batch_norm(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3
    dropout_val = 0.2

    model = ks.models.Sequential()
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", input_shape=(n_bins[2], n_bins[3], 1),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_1, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(1,2)))
    model.add(Convolution2D(n_filters_2, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_val))
    model.add(Convolution2D(n_filters_3, (kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model


def define_35d_model_xyz_t(number_of_classes, n_bins, dropout=0):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3


    model = ks.models.Sequential()
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[0], n_bins[1], n_bins[2], n_bins[3]),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    #model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model


def define_3d_model_yzt(number_of_classes, n_bins, dropout=0):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3


    model = ks.models.Sequential()
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[1], n_bins[2], n_bins[3], 1),
                            padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same", kernel_initializer='he_normal', use_bias=False))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes, activation='softmax')) #activation='sigmoid'

    return model

#------------- Undocumented legacy tests -------------#








