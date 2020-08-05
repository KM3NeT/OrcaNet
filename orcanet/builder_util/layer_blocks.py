import tensorflow.keras.backend as K
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
import medgeconv


blocks = {}


def register(block):
    blocks[block.__name__] = block
    return block


register(medgeconv.DisjointEdgeConvBlock)


@register
class ConvBlock:
    """
    1D/2D/3D Convolutional block followed by BatchNorm, Activation,
    MaxPooling and/or Dropout.

    Parameters
    ----------
    conv_dim : int
        Specifies the dimension of the convolutional block, 1D/2D/3D.
    filters : int
        Number of filters used for the convolutional layer.
    strides : int or tuple
        The stride length of the convolution.
    padding : str or int or list
        If str: Padding of the conv block.
        If int or list: Padding argument of a ZeroPaddingND layer that
        gets added before the convolution.
    kernel_size : int or tuple
        Kernel size which is used for all three dimensions.
    pool_size : None or int or tuple
        Specifies pool size for the pooling layer, e.g. (1,1,2)
        -> sizes for a 3D conv block. If its None, no pooling will be added,
        except for when global average pooling is used.
    pool_type : str, optional
        The type of pooling layer to add. Ignored if pool_size is None.
        Can be max_pooling (default), average_pooling, or
        global_average_pooling.
    pool_padding : str
        Padding option of the pooling layer.
    dropout : float or None
        Adds a dropout layer if the value is not None.
        Can not be used together with sdropout.
        Hint: 0 will add a dropout layer, but with a rate of 0 (=no dropout).
    sdropout : float or None
        Adds a spatial dropout layer if the value is not None.
        Can not be used together with dropout.
    activation : str or None
        Type of activation function that should be used. E.g. 'linear',
        'relu', 'elu', 'selu'.
    kernel_l2_reg : float, optional
        Regularization factor of l2 regularizer for the weights.
    batchnorm : bool
        Adds a batch normalization layer.
    kernel_initializer : string
        Initializer for the kernel weights.
    time_distributed : bool
        If True, apply the TimeDistributed Wrapper around all layers.
    dilation_rate : int
        An integer or tuple/list of a single integer, specifying the
        dilation rate to use for dilated convolution. Currently,
        specifying any dilation_rate value != 1 is incompatible
        with specifying any strides value != 1.

    """
    def __init__(self, conv_dim,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding="same",
                 pool_type="max_pooling",
                 pool_size=None,
                 pool_padding="valid",
                 dropout=None,
                 sdropout=None,
                 activation='relu',
                 kernel_l2_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal",
                 time_distributed=False,
                 dilation_rate=1):
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.dropout = dropout
        self.sdropout = sdropout
        self.activation = activation
        self.kernel_l2_reg = kernel_l2_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer
        self.time_distributed = time_distributed
        self.dilation_rate = dilation_rate

    def __call__(self, inputs):
        if self.dropout is not None and self.sdropout is not None:
            raise ValueError("Can only use either dropout or spatial "
                             "dropout, not both")

        dim_layers = _get_dimensional_layers(self.conv_dim)
        convolution_nd = dim_layers["convolution"]
        s_dropout_nd = dim_layers["s_dropout"]

        if self.kernel_l2_reg is not None:
            kernel_reg = ks.regularizers.l2(self.kernel_l2_reg)
        else:
            kernel_reg = None

        if self.batchnorm:
            use_bias = False
        else:
            use_bias = True

        block_layers = list()

        if isinstance(self.padding, str):
            padding = self.padding
        else:
            block_layers.append(dim_layers["zero_padding"](self.padding))
            padding = "valid"

        block_layers.append(convolution_nd(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=use_bias,
            kernel_regularizer=kernel_reg,
            dilation_rate=self.dilation_rate)
        )
        if self.batchnorm:
            channel_axis = 1 if ks.backend.image_data_format() == "channels_first" else -1
            block_layers.append(layers.BatchNormalization(axis=channel_axis))
        if self.activation is not None:
            block_layers.append(layers.Activation(self.activation))

        if self.pool_type == "global_average_pooling":
            pooling_nd = dim_layers[self.pool_type]
            block_layers.append(pooling_nd())
        elif self.pool_size is not None:
            pooling_nd = dim_layers[self.pool_type]
            block_layers.append(pooling_nd(
                pool_size=self.pool_size, padding=self.pool_padding))

        if self.dropout is not None:
            block_layers.append(layers.Dropout(self.dropout))
        elif self.sdropout is not None:
            block_layers.append(s_dropout_nd(self.sdropout))

        x = inputs
        for block_layer in block_layers:
            if self.time_distributed:
                x = layers.TimeDistributed(block_layer)(x)
            else:
                x = block_layer(x)
        return x


@register
class DenseBlock:
    """
    Dense layer followed by BatchNorm, Activation and/or Dropout.

    Parameters
    ----------
    units : int
        Number of neurons of the dense layer.
    dropout : float or None
        Adds a dropout layer if the value is not None.
    activation : str or None
        Type of activation function that should be used. E.g. 'linear',
        'relu', 'elu', 'selu'.
    kernel_l2_reg : float, optional
        Regularization factor of l2 regularizer for the weights.
    batchnorm : bool
        Adds a batch normalization layer.

    """
    def __init__(self, units,
                 dropout=None,
                 activation='relu',
                 kernel_l2_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.kernel_l2_reg = kernel_l2_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        if self.kernel_l2_reg is not None:
            kernel_reg = ks.regularizers.l2(self.kernel_l2_reg)
        else:
            kernel_reg = None

        if self.batchnorm:
            use_bias = False
        else:
            use_bias = True

        x = layers.Dense(
            units=self.units,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=kernel_reg)(inputs)

        if self.batchnorm:
            channel_axis = 1 if ks.backend.image_data_format() == "channels_first" else -1
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = layers.Activation(self.activation)(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        return x


@register
class MEdgeConvBlock:
    """ EdgeConv as defined in ParticleNet, see github.com/StefReck/MEdgeConv """
    def __init__(self, units,
                 next_neighbors=16,
                 shortcut=True,
                 batchnorm_for_nodes=False,
                 pooling=False,
                 kernel_initializer="glorot_uniform",
                 activation="relu"):
        self.units = units
        self.next_neighbors = next_neighbors
        self.batchnorm_for_nodes = batchnorm_for_nodes
        self.shortcut = shortcut
        self.pooling = pooling
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def __call__(self, x):
        nodes, is_valid, coordinates = x

        if self.batchnorm_for_nodes:
            nodes = layers.BatchNormalization()(nodes)

        nodes = medgeconv.EdgeConv(
            units=self.units,
            next_neighbors=self.next_neighbors,
            kernel_initializer=self.kernel_initializer,
            activation=self.activation,
            shortcut=self.shortcut,
        )((nodes, is_valid, coordinates))

        if self.pooling:
            return medgeconv.GlobalAvgValidPooling()((nodes, is_valid))
        else:
            return nodes, is_valid, nodes


@register
class ResnetBlock:
    """
    A residual building block for resnets. 2 c layers with a shortcut.
    https://arxiv.org/pdf/1605.07146.pdf

    Parameters
    ----------
    conv_dim : int
        Specifies the dimension of the convolutional block, 2D/3D.
    filters : int
        Number of filters used for the convolutional layers.
    strides : int or tuple
        The stride length of the convolution. If strides is 1, this is
        the identity block. If not, it has a conv block
        at the shortcut.
    kernel_size : int or tuple
        Kernel size which is used for all three dimensions.
    activation : str or None
        Type of activation function that should be used. E.g. 'linear',
        'relu', 'elu', 'selu'.
    batchnorm : bool
        Adds a batch normalization layer.
    kernel_initializer : string
        Initializer for the kernel weights.
    time_distributed : bool
        If True, apply the TimeDistributed Wrapper around all layers.

    """
    def __init__(self, conv_dim,
                 filters,
                 strides=1,
                 kernel_size=3,
                 activation='relu',
                 batchnorm=False,
                 kernel_initializer="he_normal",
                 time_distributed=False):
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer
        self.time_distributed = time_distributed

    def __call__(self, inputs):
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=self.filters,
                      kernel_size=self.kernel_size,
                      strides=self.strides,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=self.activation,
                      time_distributed=self.time_distributed)(inputs)
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=self.filters,
                      kernel_size=self.kernel_size,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=None,
                      time_distributed=self.time_distributed)(x)

        if self.strides != 1:
            shortcut = ConvBlock(conv_dim=self.conv_dim,
                                 filters=self.filters,
                                 kernel_size=1,
                                 strides=self.strides,
                                 kernel_initializer=self.kernel_initializer,
                                 activation=None,
                                 batchnorm=self.batchnorm,
                                 time_distributed=self.time_distributed,
                                 )(inputs)
        else:
            shortcut = inputs

        x = layers.add([x, shortcut])
        acti_layer = layers.Activation(self.activation)
        if self.time_distributed:
            return layers.TimeDistributed(acti_layer)(x)
        else:
            return acti_layer(x)


@register
class ResnetBnetBlock:
    """
    A residual bottleneck building block for resnets.
    https://arxiv.org/pdf/1605.07146.pdf

    Parameters
    ----------
    conv_dim : int
        Specifies the dimension of the convolutional block, 2D/3D.
    filters : List
        Number of filters used for the convolutional layers.
        Has to be length 3. First and third is for the 1x1 convolutions.
    strides : int or tuple
        The stride length of the convolution. If strides is 1, this is
        the identity block. If not, it has a conv block
        at the shortcut.
    kernel_size : int or tuple
        Kernel size which is used for all three dimensions.
    activation : str or None
        Type of activation function that should be used. E.g. 'linear',
        'relu', 'elu', 'selu'.
    batchnorm : bool
        Adds a batch normalization layer.
    kernel_initializer : string
        Initializer for the kernel weights.

    """
    def __init__(self, conv_dim,
                 filters,
                 strides=1,
                 kernel_size=3,
                 activation='relu',
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        filters1, filters2, filters3 = self.filters

        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=filters1,
                      kernel_size=1,
                      strides=self.strides,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=self.activation)(inputs)
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=filters2,
                      kernel_size=self.kernel_size,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=self.activation)(x)
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=filters3,
                      kernel_size=1,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=None)(x)

        if self.strides != 1:
            shortcut = ConvBlock(conv_dim=self.conv_dim,
                                 filters=filters3,
                                 kernel_size=1,
                                 strides=self.strides,
                                 kernel_initializer=self.kernel_initializer,
                                 activation=None,
                                 batchnorm=self.batchnorm,
                                 )(inputs)
        else:
            shortcut = inputs

        x = layers.add([x, shortcut])
        x = layers.Activation(self.activation)(x)
        return x


@register
class InceptionBlockV2:
    """
    A GoogleNet Inception block (v2).
    https://arxiv.org/pdf/1512.00567v3.pdf, see fig. 5.
    Keras implementation, e.g.:
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py

    Parameters
    ----------
    conv_dim : int
        Specifies the dimension of the convolutional block, 1D/2D/3D.
    filters_1x1 : int or None
        No. of filters for the 1x1 convolutional branch.
        If None, dont make this branch.
    filters_pool : int or None
        No. of filters for the pooling branch.
        If None, dont make this branch.
    filters_3x3 : tuple or None
        No. of filters for the 3x3 convolutional branch. First int
        is the filters in the 1x1 conv, second int for the 3x3 conv.
        First should be chosen smaller for computational efficiency.
        If None, dont make this branch.
    filters_3x3dbl : tuple or None
        No. of filters for the 3x3 convolutional branch. First int
        is the filters in the 1x1 conv, second int for the two 3x3 convs.
        First should be chosen smaller for computational efficiency.
        If None, dont make this branch.
    strides : int or tuple
        Stride length of this block.
        Like in the keras implementation, no 1x1 convs with stride > 1
        will be used, instead they will be skipped.

    """
    def __init__(self,
                 conv_dim,
                 filters_1x1,
                 filters_pool,
                 filters_3x3,
                 filters_3x3dbl,
                 strides=1,
                 activation="relu",
                 batchnorm=False,
                 dropout=None):
        self.filters_1x1 = filters_1x1  # 64
        self.filters_pool = filters_pool  # 64
        self.filters_3x3 = filters_3x3  # 48, 64
        self.filters_3x3dbl = filters_3x3dbl  # 64, 96
        self.strides = strides
        self.conv_options = {
            "conv_dim": conv_dim,
            "dropout": dropout,
            "batchnorm": batchnorm,
            "activation": activation,
        }

    def __call__(self, inputs):
        branches = []
        # 1x1 convolution
        if self.filters_1x1 and self.strides == 1:
            branch1x1 = ConvBlock(
                filters=self.filters_1x1, kernel_size=1,
                strides=self.strides, **self.conv_options)(inputs)
            branches.append(branch1x1)

        # pooling
        if self.filters_pool:
            max_pooling_nd = _get_dimensional_layers(
                self.conv_options["conv_dim"])["max_pooling"]
            branch_pool = max_pooling_nd(
                pool_size=3, strides=self.strides,  padding='same')(inputs)
            if self.strides == 1:
                branch_pool = ConvBlock(
                    filters=self.filters_pool, kernel_size=1,
                    **self.conv_options)(branch_pool)
            branches.append(branch_pool)

        # 3x3 convolution
        if self.filters_3x3:
            branch3x3 = ConvBlock(
                filters=self.filters_3x3[0], kernel_size=1,
                **self.conv_options)(inputs)
            branch3x3 = ConvBlock(
                filters=self.filters_3x3[1], kernel_size=3,
                strides=self.strides, **self.conv_options)(branch3x3)
            branches.append(branch3x3)

        # double 3x3 convolution
        if self.filters_3x3dbl:
            branch3x3dbl = ConvBlock(
                filters=self.filters_3x3dbl[0], kernel_size=1,
                **self.conv_options)(inputs)
            branch3x3dbl = ConvBlock(
                filters=self.filters_3x3dbl[1], kernel_size=1,
                **self.conv_options)(branch3x3dbl)
            branch3x3dbl = ConvBlock(
                filters=self.filters_3x3dbl[1], kernel_size=1,
                strides=self.strides, **self.conv_options)(branch3x3dbl)
            branches.append(branch3x3dbl)

        # concatenate all branches
        channel_axis = 1 if ks.backend.image_data_format() == "channels_first" else -1
        x = layers.concatenate(branches, axis=channel_axis)
        return x


@register
class OutputReg:
    """
    Dense layer(s) for regression.

    Parameters
    ----------
    output_neurons : int
        Number of neurons in the last layer.
    output_name : str or None
        Name that will be given to the output layer of the network.
    unit_list : List, optional
        A list of ints. Add additional Dense layers after the gpool
        with this many units in them. E.g., [64, 32] would add
        two Dense layers, the first with 64 neurons, the secound with
        32 neurons.
    transition : str or None
        Name of a layer that will be used as the first layer of this block.
        Example: 'keras:GlobalAveragePooling2D', 'keras:Flatten'
    kwargs
        Keywords for the dense blocks that get added if unit_list is
        not None.

    """
    def __init__(self, output_neurons,
                 output_name,
                 unit_list=None,
                 transition='keras:Flatten',
                 **kwargs):
        self.output_neurons = output_neurons
        self.output_name = output_name
        self.unit_list = unit_list
        self.transition = transition
        self.kwargs = kwargs

    def __call__(self, layer):
        if self.transition:
            x = getattr(layers, self.transition.split("keras:")[-1])()(layer)
        else:
            x = layer

        if self.unit_list is not None:
            for units in self.unit_list:
                x = DenseBlock(units=units, **self.kwargs)(x)

        out = layers.Dense(
            units=self.output_neurons,
            activation=None,
            name=self.output_name)(x)

        return out


@register
class OutputCateg:
    """
    Dense layer(s) for categorization.

    Parameters
    ----------
    categories : int
        Number of categories (= neurons in the last layer).
    output_name : str
        Name that will be given to the output layer of the network.
    unit_list : List, optional
        A list of ints. Add additional Dense layers after the gpool
        with this many units in them. E.g., [64, 32] would add
        two Dense layers, the first with 64 neurons, the secound with
        32 neurons.
    transition : str or None
        Name of a layer that will be used as the first layer of this block.
        Example: 'keras:GlobalAveragePooling2D', 'keras:Flatten'
    kwargs
        Keywords for the dense blocks that get added if unit_list is
        not None.

    """
    def __init__(self, categories,
                 output_name,
                 unit_list=None,
                 transition='keras:Flatten',
                 **kwargs):
        self.categories = categories
        self.output_name = output_name
        self.unit_list = unit_list
        self.transition = transition
        self.kwargs = kwargs

    def __call__(self, layer):
        if self.transition:
            x = getattr(layers, self.transition.split("keras:")[-1])()(layer)
        else:
            x = layer

        if self.unit_list is not None:
            for units in self.unit_list:
                x = DenseBlock(units=units, **self.kwargs)(x)

        out = layers.Dense(
            units=self.categories,
            activation='softmax',
            kernel_initializer='he_normal',
            name=self.output_name)(x)

        return out


@register
class OutputRegErr:
    """
    Double network for regression + error estimation.

    It has 3 dense layer blocks, followed by one dense layer
    for each output_name, as well as dense layer blocks, followed by one dense layer
    for the respective error of each output_name.

    Parameters
    ----------
    output_names : List
        List of strs, the output names, each with one neuron + one err neuron.
    flatten : bool
        If True, start with a flatten layer.
    kwargs
        Keywords for the dense blocks.

    """
    def __init__(self, output_names, flatten=True, **kwargs):
        self.flatten = flatten
        self.output_names = output_names
        self.kwargs = kwargs

    def __call__(self, layer):
        if self.flatten:
            flatten = layers.Flatten()(layer)
        else:
            flatten = layer
        outputs = []

        x = DenseBlock(units=128, **self.kwargs)(flatten)
        x = DenseBlock(units=32, **self.kwargs)(x)

        for name in self.output_names:
            output_label = layers.Dense(units=1, name=name)(x)
            outputs.append(output_label)

        # Network for the errors of the labels
        x_err = layers.Lambda(lambda a: K.stop_gradient(a))(flatten)

        x_err = DenseBlock(units=128, **self.kwargs)(x_err)
        x_err = DenseBlock(units=64, **self.kwargs)(x_err)
        x_err = DenseBlock(units=32, **self.kwargs)(x_err)

        for i, name in enumerate(self.output_names):
            output_label_error = layers.Dense(
                units=1,
                activation='linear',
                name=name + '_err_temp')(x_err)
            # Predicted label gets concatenated with its error (needed for loss function)
            output_label_merged = layers.Concatenate(name=name + '_err')(
                [outputs[i], output_label_error])
            outputs.append(output_label_merged)
        return outputs


def _get_dimensional_layers(dim):
    if dim not in (1, 2, 3):
        raise ValueError(f'Dimension must be 1, 2 or 3, not {dim}')
    dim_layers = {
        "convolution": {
            1: layers.Convolution1D,
            2: layers.Convolution2D,
            3: layers.Convolution3D,
        },
        "max_pooling": {
            1: layers.MaxPooling1D,
            2: layers.MaxPooling2D,
            3: layers.MaxPooling3D,
        },
        "average_pooling": {
            1: layers.AveragePooling1D,
            2: layers.AveragePooling2D,
            3: layers.AveragePooling3D,
        },
        "global_average_pooling": {
            1: layers.GlobalAveragePooling1D,
            2: layers.GlobalAveragePooling2D,
            3: layers.GlobalAveragePooling3D,
        },
        "s_dropout": {
            1: layers.SpatialDropout1D,
            2: layers.SpatialDropout2D,
            3: layers.SpatialDropout3D,
        },
        "zero_padding": {
            1: layers.ZeroPadding1D,
            2: layers.ZeroPadding2D,
            3: layers.ZeroPadding3D,
        },
    }
    return {layer_type: dim_layers[layer_type][dim] for layer_type in dim_layers.keys()}
