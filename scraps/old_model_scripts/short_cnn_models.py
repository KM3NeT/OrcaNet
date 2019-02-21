#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for creating VGG-like models (including VGG-LSTM)"""

import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D,\
                         Convolution2D, MaxPooling2D, TimeDistributed, CuDNNLSTM, Concatenate, Lambda
from keras import backend as K
from keras.regularizers import l2

from scraps.old_model_scripts.file_dump import get_dimensions_encoding

# ------------- VGG-like model -------------#


def decode_input_dimensions_vgg(n_bins, batchsize, swap_col, str_ident =''):
    """
    Returns the general dimension (2D/3D), the input dimensions (i.e. bs x 11 x 13 x 18 x channels=1 for 3D)
    and appropriate max_pool_sizes depending on the projection type.

    Parameters
    ----------
    n_bins : dict(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples for different inputs.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    swap_col : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.

    Returns
    -------
    dim : int
        Dimension of the network (2D/3D/4D).
    input_dim : tuple
        Dimensions tuple for 2D, 3D or 4D data (ints).
    max_pool_sizes : dict/dict(dict)
        Dict that specifies the Pooling locations and properties for a cnn
        Key: After which convolutional layer a pooling layer be inserted (counts from layer zero!)
        Item: Specifies the strides.

    """
    # TODO Change this to actually use the dict keywords!
    n_bins = list(n_bins.values())
    if n_bins[0].count(1) == 1:  # 3d case
        dim = 3
        input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize

        if n_bins[0][3] == 1:
            print('Using a VGG-like 3D CNN with XYZ projection')
            max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}

        elif n_bins[0][0] == 1:
            print('Using a VGG-like 3D CNN with YZT projection')
            max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}

        else:
            raise IndexError('No suitable 3D projection found in decode_input_dimensions().'
                             'Please add the projection type with the pooling dict to the function.')

    elif n_bins[0].count(1) == 0:  # 4d case, 3.5D
        dim = 3
        if len(n_bins) > 1:

            if swap_col == 'xyz-t_and_yzt-x' and 'tight-1_tight-2' not in str_ident:
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},  # only used if training from scratch
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]),  # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])]  # yzt-x

            elif swap_col == 'xyz-t_and_yzt-x' and 'multi_input_single_train_tight-1_tight-2' in str_ident:
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},  # only used if training from scratch
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_3': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_4': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]),  # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1]),  # yzt-x
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]),  # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])]  # yzt-x

            elif swap_col == 'xyz-t_and_xyz-c_single_input':
                max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}  # 2 more layers ; same as swap_col=None for len(n_bins)=1
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)
                input_dim = (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4] + 31)

            elif swap_col is None:
                max_pool_sizes = {'net_1': {5: (2, 2, 2), 9: (2, 2, 2)},
                                  'net_2': {5: (2, 2, 2), 9: (2, 2, 2)}}
                input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]),  # xyz-t tight-1
                             (input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4])]  # xyz-t tight-2

            else:
                raise IOError('3.5D projection types with len(n_bins) > 1 other than "yzt-x_all-t_and_yzt-x_tight-1-t" are not yet supported.')

        else:
            input_dim = get_dimensions_encoding(n_bins[0], batchsize)  # includes batchsize
            if swap_col is None:
                print('Using a VGG-like 3.5D CNN with XYZ data and T/C channel information.')
                # max_pool_sizes = {3: (2, 2, 2), 7: (2, 2, 2)}
                max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)} # 2 more layers
                # max_pool_sizes = {7: (2, 2, 2), 11: (2, 2, 2)}  # 4 more layers

            elif swap_col == 'yzt-x':
                print('Using a VGG-like 3.5D CNN with YZT data and X channel information.')
                # max_pool_sizes = {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}
                max_pool_sizes = {2: (1, 1, 2), 5: (2, 2, 2), 9: (2, 2, 2)} # 2 more layers
                input_dim = (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])  # [bs,y,z,t,x]

            elif swap_col == 'xyz-t_and_yzt-x':
                max_pool_sizes = {'net_1': {3: (2, 2, 2), 7: (2, 2, 2)},
                                  'net_2': {1: (1, 1, 2), 3: (2, 2, 2), 7: (2, 2, 2)}}
                input_dim = [(input_dim[0], input_dim[1], input_dim[2], input_dim[3], input_dim[4]),  # xyz-t
                             (input_dim[0], input_dim[2], input_dim[3], input_dim[4], input_dim[1])]  # yzt-x

            elif swap_col == 'conv_lstm':  # TODO fix whole function to make it more general and not only 3.5D stuff
                max_pool_sizes = {1: (2, 2, 2), 5: (2, 2, 2)}
                input_dim = (input_dim[0], input_dim[4], input_dim[1], input_dim[2], input_dim[3], 1)  # t-xyz

            else:
                raise IOError('3.5D projection types other than XYZ-T and YZT-X are not yet supported.'
                              'Please add the max_pool_sizes dict in the function by yourself.')

    else:
        raise IOError('Data types other than 3D or 4D (3.5D actually) are not yet supported. '
                      'Please specify a 3D or 4D n_bins tuple.')

    return dim, input_dim, max_pool_sizes


def create_vgg_like_model(n_bins, class_type, n_filters=None, dropout=0, k_size=3, swap_col=None,
                          activation='relu', kernel_reg=None):
    """
    Returns a VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.

    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)

    Parameters
    ----------
    n_bins : dict(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples for different inputs.
    class_type : str
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    n_filters : tuple
        Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    dropout : float
        Adds dropout if >0.
    k_size : int
        Kernel size which is used for all dimensions.
    swap_col : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    activation : str
        Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.
    kernel_reg : None/str
        If L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.

    Returns
    -------
    model : ks.models.Model
        A VGG-like Keras nn instance.

    """
    if n_filters is None: n_filters = (64, 64, 64, 64, 64, 128, 128, 128)
    if kernel_reg is 'l2': kernel_reg = l2(0.0001)

    # TODO Batchsize has to be given to decode_input_dimensions_vgg, but is not used for constructing the model.
    # For now: Just use some random value.
    batchsize = 64

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_col)

    input_layer = Input(shape=input_dim[1:], name=swap_col, dtype=K.floatx())  # input_layer
    x = conv_block(input_layer, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(0), activation=activation, kernel_reg=kernel_reg)

    for i in range(1, len(n_filters)):
        x = conv_block(x, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(i), activation=activation, kernel_reg=kernel_reg)

    conv_output_flat = Flatten()(x)
    outputs = add_dense_layers_to_cnn(conv_output_flat, class_type, dropout=dropout, activation=activation, kernel_reg=kernel_reg)

    model = Model(inputs=input_layer, outputs=outputs)

    return model


def conv_block(ip, dim, n_filters, k_size=3, dropout=0, max_pooling=None, activation='relu', kernel_reg = None):
    """
    2D/3D Convolutional block followed by BatchNorm and Activation with optional MaxPooling or Dropout.

    C-B-A-(MP)-(D)

    Parameters
    ----------
    ip : ? # TODO
        Keras functional layer instance that is used as the starting point of this convolutional block.
    dim : int
        Specifies the dimension of the convolutional block, 2D/3D.
    n_filters : int
        Number of filters used for the convolutional layer.
    k_size : int
        Kernel size which is used for all three dimensions.
    dropout : float
        Adds a dropout layer if the value is greater than 0.
    max_pooling : None/tuple
        Specifies if a MaxPooling layer should be added. e.g. (1,1,2) -> strides for a 3D conv block.
    activation : str
        Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    kernel_reg : None/str
        If L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.

    Returns
    -------
    x :
        Resulting output tensor (model).

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


def add_dense_layers_to_cnn(conv_output_flat, class_type, dropout=0, activation='relu', kernel_reg=None):
    """
    Appends dense layers to the convolutional layers of a cnn.

    Parameters
    ----------
    conv_output_flat : ? # TODO
        The Keras layer instance after the Flatten() layer.
    class_type : str
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    dropout : float
        Adds a dropout layer if the value is greater than 0.
    activation : str
        Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    kernel_reg : None/str
        If L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.

    Returns
    -------
    outputs : list
        List of outputs for the cnn.

    """

    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(conv_output_flat)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(32, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(x)

    outputs = []

    if class_type == 'ts_classifier':  # categorical problem
        x = Dense(2, activation='softmax', kernel_initializer='he_normal', name='ts_output')(x)
        outputs.append(x)

    elif class_type == 'bg_classifier':  # categorical problem
        x = Dense(3, activation='softmax', kernel_initializer='he_normal', name='bg_output')(x)
        outputs.append(x)

    elif class_type == 'bg_classifier_2_class':  # categorical problem
        x = Dense(2, activation='softmax', kernel_initializer='he_normal', name='bg_output')(x)
        outputs.append(x)

    else:  # regression case, one output for each regression label

        if class_type == 'energy_dir_bjorken-y_errors':
            label_names = ('e', 'dir_x', 'dir_y', 'dir_z', 'by')
            for name in label_names:
                output_label = Dense(1, name=name)(x)
                outputs.append(output_label)

            # build second dense network for errors
            conv_output_flat_stop_grad = Lambda(lambda a: K.stop_gradient(a))(conv_output_flat)
            x_err = Dense(128, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(
                conv_output_flat_stop_grad)
            if dropout > 0.0: x_err = Dropout(dropout)(x_err)
            x_err = Dense(64, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(
                x_err)
            if dropout > 0.0: x_err = Dropout(dropout)(x_err)
            x_err = Dense(32, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(
                x_err)

            for i, name in enumerate(label_names):
                output_label_error = Dense(1, activation='linear', name=name + '_err_temp')(x_err)
                # second stop_gradient through the concat layer into the dense_nn of the labels is in the err_loss function
                output_label_merged = Concatenate(name=name + '_err')([outputs[i], output_label_error])
                outputs.append(output_label_merged)

        elif class_type == 'energy_dir_bjorken-y_vtx_errors':
            label_names = ('e', 'dx', 'dy', 'dz', 'by', 'vx', 'vy', 'vz', 'vt')
            for name in label_names:
                output_label = Dense(1, name=name)(x)
                outputs.append(output_label)

            # build second dense network for errors
            conv_output_flat_stop_grad = Lambda(lambda a: K.stop_gradient(a))(conv_output_flat)
            x_err = Dense(128, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(conv_output_flat_stop_grad)
            if dropout > 0.0: x_err = Dropout(dropout)(x_err)
            x_err = Dense(64, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(x_err)
            if dropout > 0.0: x_err = Dropout(dropout)(x_err)
            x_err = Dense(32, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, activation=activation)(x_err)

            for i, name in enumerate(label_names):
                output_label_error = Dense(1, activation='linear', name=name + '_err_temp')(x_err)
                # second stop_gradient through the concat layer into the dense_nn of the labels is in the err_loss function
                output_label_merged = Concatenate(name=name + '_err')([outputs[i], output_label_error])
                outputs.append(output_label_merged)

        else:
            raise ValueError(class_type, "is not a known class_type!")

    return outputs


def create_vgg_like_model_double_input(n_bins, batchsize, nb_classes=2, n_filters=None, dropout=0, k_size=3, swap_4d_channels=None, activation='relu'):
    """
    Returns a double input, VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished for classification.

    The two single VGG networks are concatenated after the last flatten layers.
    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)

    Parameters
    ----------
    n_bins : list(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    nb_classes : int
        Number of output classes.
    n_filters : tuple
        Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    dropout : float
        Adds dropout if >0.
    k_size : int
        Kernel size which is used for all dimensions.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    activation : str
        Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.

    Returns
    -------
    model : ks.models.Model
        A VGG-like, double input Keras nn instance.

    """
    if n_filters is None: n_filters = (64,64,64,64,64,128,128,128)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels)

    input_layer_1 = Input(shape=input_dim[0][1:], dtype=K.floatx())  # input_layer 1
    input_layer_2 = Input(shape=input_dim[1][1:], dtype=K.floatx())  # input_layer 2

    # Net 1 convs
    x_1 = conv_block(input_layer_1, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_1'].get(0), activation=activation)

    for i in range(1, len(n_filters)):
        x_1 = conv_block(x_1, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_1'].get(i), activation=activation)

    # Net 2 convs
    x_2 = conv_block(input_layer_2, dim, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_2'].get(0), activation=activation)

    for i in range(1, len(n_filters)):
        x_2 = conv_block(x_2, dim, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes['net_2'].get(i), activation=activation)

    # flatten both nets
    x_1, x_2 = Flatten()(x_1), Flatten()(x_2)

    # concatenate both nets
    x = ks.layers.concatenate([x_1, x_2])

    x = Dense(128, activation=activation, kernel_initializer='he_normal')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, activation=activation, kernel_initializer='he_normal')(x)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input_layer_1, input_layer_2], outputs=x)

    return model


def create_vgg_like_model_multi_input_from_single_nns(n_bins, str_ident, dropout=(0, 0.2), swap_4d_channels=None, activation='relu'):
    """
    Returns a double input, VGG-like model (stacked conv. layers) with MaxPooling and Dropout if wished.

    The two single VGG networks are concatenated after the last flatten layers.

    Parameters
    ----------
    n_bins : list(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    nb_classes : int
        Number of output classes.
    dropout : tuple(float, float)
        Adds dropout if >0.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    activation : str
        Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.

    Returns
    -------
    model : ks.models.Model
        A VGG-like, double input Keras nn instance, with pretrained conv layers.

    """
    # TODO Batchsize has to be given to decode_input_dimensions_vgg, but is not used for constructing the model.
    # For now: Just use some random value.
    batchsize=64
    nb_classes = 2

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, swap_4d_channels, str_ident=str_ident)
    trained_model_paths = {}
    if swap_4d_channels is None: swap_4d_channels = ''

    if 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1' in swap_4d_channels + str_ident and 'multi_input_single_train_tight-1_tight-2' not in str_ident:
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_muon-CC_to_elec-CC_xyz-t_tight-1_w-geo-fix_bs64_2-more-layers_epoch_32_file_1.h5'  # xyz-t, timecut tight_1, with geo fix, 2 more layers
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_yzt-x_muon-CC_to_elec-CC_tight-1_w-geo-fix_bs64_dp0.1_2-more-layers_epoch_30_file_1.h5'  # yzt-x, timecut tight-1, with geo fix, 2 more layers

    elif swap_4d_channels is '': # xyz-t tight-1 and xyz-t tight-2
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_track-shower_lp_tight-1_bs64_dp0.1_pad_same_epoch_15_file_1.h5'  # lp, xyz-t, timecut tight_1, fully trained epoch 16, file 1
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_xyz-t_track-shower_lp_tight-2_bs64_dp0.1_padsame_epoch_12_file_1.h5'  # lp, xyz-t, timecut tight-2, fully trained epoch 13, file 1

    elif 'xyz-t_and_yzt-x' + 'multi_input_single_train_tight-1_tight-2' in swap_4d_channels + str_ident:
        trained_model_paths[0] = 'models/trained/trained_model_VGG_4d_xyz-t_track-shower_lp_tight-1_bs64_dp0.1_pad_same_epoch_16_file_1.h5'  # lp, xyz-t, timecut tight-1, fully trained epoch 16, file 1
        trained_model_paths[2] = 'models/trained/trained_model_VGG_4d_xyz-t_track-shower_lp_tight-2_bs64_dp0.1_padsame_epoch_13_file_1.h5'  # lp, xyz-t, timecut tight-2, fully trained epoch 13, file 1
        trained_model_paths[1] = 'models/trained/trained_model_VGG_4d_yzt-x_track-shower_lp_tight-1_bs64_dp0.1_padsame_epoch_10_file_3.h5' # lp, yzt-x, timecut tight-1, fully trained epoch 10, file 3
        trained_model_paths[3] = 'models/trained/trained_model_VGG_4d_yzt-x_track-shower_lp_tight-2_bs64_dp0.1_padsame_epoch_11_file_2.h5' # lp, yzt-x, timecut tight-2, fully trained epoch 11, file 2

    else:
        raise ValueError('The double input combination specified in "swap_4d_channels" is not known, check the function for what is available.')

    n_inputs = len(trained_model_paths)
    trained_models = {}
    input_layers = {}
    layer_numbers = {}
    x = {}

    for i in range(n_inputs):
        trained_models[i] = ks.models.load_model(trained_model_paths[i])

        input_layers[i] = Input(shape=input_dim[i][1:], name='input_net_' + str(i+1), dtype=K.floatx())
        layer_numbers[i] = {'conv': 1, 'batch_norm': 1, 'activation': 1, 'max_pooling': 1, 'dropout': 1}

        x[i] = create_layer_from_config(input_layers[i], trained_models[i].layers[1], layer_numbers[i], trainable=False, net=str(i+1))
        for trained_layer in trained_models[i].layers[2:]:
            if 'flatten' in trained_layer.name: break  # we don't want to get anything after the flatten layer
            x[i] = create_layer_from_config(x[i], trained_layer, layer_numbers[i], trainable=False, net=str(i+1), dropout=dropout[0])

        x[i] = Flatten()(x[i])

    x = ks.layers.concatenate([x[i] for i in x])

    x = Dense(128, activation=activation, kernel_initializer='he_normal')(x)
    x = Dropout(dropout[1])(x)
    x = Dense(32, activation=activation, kernel_initializer='he_normal')(x)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input_layers[i] for i in input_layers], outputs=x)
    set_layer_weights(model, trained_models) # set weights

    for layer in model.layers: # freeze trainable batch_norm weights, but not running mean and variance
        if 'batch_norm' in layer.name:
            layer.stateful = True

    return model


def create_layer_from_config(x, trained_layer, layer_numbers, trainable=False, net='', dropout=0):
    """
    Creates a new Keras nn layer from the config of an already existing layer.

    Changes the 'trainable' flag of the new layer to false and optionally udates the dropout rate.
    Adds a layer name based on the layer_numbers dict.

    Parameters
    ----------
    x : ? # TODO
        Keras functional model api instance. E.g. TF tensors.
    trained_layer : ks.layer
        Keras layer instance that is already trained.
    layer_numbers : dict
        Dictionary for the different layer types to keep track of the layer_number in the layer names.
    trainable : bool
        Flag to set the <trainable> attribute of the new layer.
    net : str
        Additional string that is added to the layer name. E.g. 'net_2' if a double input model is used.
    dropout : float
        optional, dropout rate of the new layer

    Returns
    -------
    x : ?
        Keras functional model api instance. E.g. TF tensors. Contains a new layer now!

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

    Parameters
    ----------
    model : ks.models.Model
        Keras model instance.
    trained_models : dict
        Dict that contains references to the Keras model instances of the already pretrained models.

    """
    skip_layers = ['dropout', 'input', 'dense', 'flatten', 'max_pooling', 'activation', 'concatenate']
    n_models = len(trained_models)
    trained_layers_w_weights = {}

    for i in range(n_models):
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


def change_dropout_rate_for_multi_input_model(n_bins, batchsize, trained_model, dropout=(0.1, 0.1), trainable=(True, True), swap_4d_channels=None):
    """
    Function that rebuilds a keras model and modifies its dropout rate. Workaround, till layer.rate is fixed to work with Dropout layers.

    Parameters
    ----------
    n_bins : list(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    trained_model : ks.models.Model
        Trained Keras model, upon which the dropout rate should be changed.
    dropout : (float, float)
        Adds dropout if > 0. First value for the conv block, second value for the dense.
    trainable : (bool, bool)
        Sets the trainable flag for the conv block layers and for the dense layers.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Only used to decode the input_dim.

    Returns
    -------
    model : ks.models.Model
        Keras VGG-like model based on the trained_model with modified dropout layers.

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

    Parameters
    ----------
    model : ks.models.Model
        Keras model instance withought weights.
    trained_model : ks.models.Model
        Pretrained Keras model used to set the weights for the new model.

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
    Only working for classification as of now!

    The number of convolutional layers can be controlled with the n_filters parameter:
    n_conv_layers = len(n_filters)

    Parameters
    ----------
    n_bins : list(tuple(int))
        Number of bins (x,y,z,t) of the data. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    nb_classes : int
        Number of output classes.
    n_filters : None/tuple
        Number of filters for each conv. layer. len(n_filters)=n_conv_layer.
    dropout : float
        Adds dropout if >0.
    k_size : int
        Kernel size which is used for all dimensions.
    activation : str
        Type of activation function that should be used for the net. E.g. 'linear', 'relu', 'elu', 'selu'.
    kernel_reg : None/str
        If L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.

    Returns
    -------
    model : ks.models.Model
        Keras conv lstm model.

    """
    if n_filters is None: n_filters = (32, 32, 64, 64, 64, 64, 128)
    if kernel_reg is 'l2': kernel_reg = l2(0.0001)

    dim, input_dim, max_pool_sizes = decode_input_dimensions_vgg(n_bins, batchsize, 'conv_lstm') # TODO fix input dim

    input_layer = Input(shape=input_dim[1:], dtype=K.floatx())  # input_layer
    x = conv_block_time_distributed(input_layer, n_filters[0], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(0), activation=activation, kernel_reg=kernel_reg)

    for i in range(1, len(n_filters)):
        x = conv_block_time_distributed(x, n_filters[i], k_size=k_size, dropout=dropout, max_pooling=max_pool_sizes.get(i), activation=activation, kernel_reg=kernel_reg)

    x = TimeDistributed(Flatten())(x)

    x = CuDNNLSTM(768)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
    x = Activation(activation)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(16, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
    x = Activation(activation)(x)

    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def conv_block_time_distributed(ip, n_filters, k_size=3, dropout=0, max_pooling=None, activation='relu', kernel_reg = None):
    """
    Time distributed 2D/3D Convolutional block followed by BatchNorm and Activation with optional MaxPooling or Dropout.

    C-B-A-(MP)-(D)

    Parameters
    ----------
    ip :
        Keras functional layer instance that is used as the starting point of this convolutional block.
    n_filters : int
        Number of filters used for the convolution.
    k_size : int
        Kernel size which is used for all three dimensions.
    dropout : float
        Adds a dropout layer if value is greater than 0.
    max_pooling : None/tuple
        Specifies if a MaxPooling layer should be added. e.g. (1,1,2) -> strides for a 3D conv block.
    activation : str
        Type of activation function that should be used. E.g. 'linear', 'relu', 'elu', 'selu'.
    kernel_reg : None/str
        If L2 regularization with 1e-4 should be employed. 'l2' to enable the regularization.

    Returns
    -------
    x :
        Resulting output tensor (model).

    """
    x = TimeDistributed(Convolution3D(n_filters, (k_size,) * 3, padding='same', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_reg))(ip)

    x = TimeDistributed(BatchNormalization(axis=-1))(x)
    x = Activation(activation)(x)

    if max_pooling is not None: x = TimeDistributed(MaxPooling3D(pool_size=max_pooling, padding='valid'))(x)
    if dropout > 0.0: x = TimeDistributed(Dropout(dropout))(x)

    return x


#------------- VGG-like model -------------#









