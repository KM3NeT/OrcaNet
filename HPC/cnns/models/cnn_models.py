#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder"""

import keras as ks
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D


def define_3d_model_xyz(number_of_classes, n_bins):
    n_filters_1 = 64
    n_filters_2 = 64
    n_filters_3 = 128
    kernel_size = 3

    model = ks.models.Sequential()
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", input_shape=(n_bins[0], n_bins[1], n_bins[2], 1), padding="same"))
    model.add(Convolution3D(n_filters_1, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(1,1,2)))
    model.add(Dropout(0.1))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(Convolution3D(n_filters_2, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    #model.add(normal.BatchNormalization())
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(Dropout(0.1))
    model.add(Convolution3D(n_filters_3, (kernel_size,kernel_size,kernel_size), activation="relu", padding="same"))
    model.add(MaxPooling3D(strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(number_of_classes) )

    return model

    #if number_parallel > 1:
    #	model = make_parallel(model, number_parallel)

    # model.add(layers.core.Permute((4,3,2,1)))
    # model.add(layers.core.Reshape((128, 8)))
