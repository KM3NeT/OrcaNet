import keras.layers as layers
from keras import backend
from keras.regularizers import l2


class ConvBlock:
    def __init__(self, conv_dim,
                 filters,
                 kernel_size=3,
                 strides=1,
                 pool_size=None,
                 pool_padding="valid",
                 dropout=None,
                 sdropout=None,
                 activation='relu',
                 kernel_l2_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        """
        2D/3D Convolutional block followed by BatchNorm, Activation,
        MaxPooling and/or Dropout.

        Parameters
        ----------
        conv_dim : int
            Specifies the dimension of the convolutional block, 2D/3D.
        filters : int
            Number of filters used for the convolutional layer.
        strides : int or tuple
            The stride length of the convolution
        kernel_size : int or tuple
            Kernel size which is used for all three dimensions.
        pool_size : None or tuple
            Specifies if a MaxPooling layer should be added. e.g. (1,1,2)
            -> sizes for a 3D conv block.
        dropout : float or None
            Adds a dropout layer if the value is not None.
            Can not be used together with sdropout.
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

        """
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.dropout = dropout
        self.sdropout = sdropout
        self.activation = activation
        self.kernel_l2_reg = kernel_l2_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        if self.dropout is not None and self.sdropout is not None:
            raise ValueError("Can only use either dropout or spatial "
                             "dropout, not both")
        if self.conv_dim == 2:
            convolution_nd = layers.Convolution2D
            max_pooling_nd = layers.MaxPooling2D
            s_dropout_nd = layers.SpatialDropout2D
        elif self.conv_dim == 3:
            convolution_nd = layers.Convolution3D
            max_pooling_nd = layers.MaxPooling3D
            s_dropout_nd = layers.SpatialDropout3D
        else:
            raise ValueError('dim must be equal to 2 or 3.')

        if self.kernel_l2_reg is not None:
            kernel_reg = l2(self.kernel_l2_reg)
        else:
            kernel_reg = None

        if self.batchnorm:
            use_bias = False
        else:
            use_bias = True

        x = convolution_nd(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding='same',
                           kernel_initializer=self.kernel_initializer,
                           use_bias=use_bias,
                           kernel_regularizer=kernel_reg)(inputs)

        if self.batchnorm:
            channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = layers.Activation(self.activation)(x)
        if self.pool_size is not None:
            x = max_pooling_nd(pool_size=self.pool_size,
                               padding=self.pool_padding)(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        elif self.sdropout is not None:
            x = s_dropout_nd(self.sdropout)(x)

        return x


class DenseBlock:
    def __init__(self, units,
                 dropout=None,
                 activation='relu',
                 kernel_l2_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
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
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.kernel_l2_reg = kernel_l2_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        if self.kernel_l2_reg is not None:
            kernel_reg = l2(self.kernel_l2_reg)
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
            channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = layers.Activation(self.activation)(x)
        if self.dropout is not None:
            x = layers.Dropout(self.dropout)(x)
        return x


class ResnetBlock:
    def __init__(self, conv_dim,
                 filters,
                 strides=1,
                 kernel_size=3,
                 activation='relu',
                 batchnorm=False,
                 kernel_initializer="he_normal"):
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

        """
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=self.filters,
                      kernel_size=self.kernel_size,
                      strides=self.strides,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=self.activation)(inputs)
        x = ConvBlock(conv_dim=self.conv_dim,
                      filters=self.filters,
                      kernel_size=self.kernel_size,
                      kernel_initializer=self.kernel_initializer,
                      batchnorm=self.batchnorm,
                      activation=None)(x)

        if self.strides != 1:
            shortcut = ConvBlock(conv_dim=self.conv_dim,
                                 filters=self.filters,
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


class ResnetBnetBlock:
    def __init__(self, conv_dim,
                 filters,
                 strides=1,
                 kernel_size=3,
                 activation='relu',
                 batchnorm=False,
                 kernel_initializer="he_normal"):
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


def input_block(input_shapes):
    """
    Build input layers according to a dict mapping the layer names to shapes.

    Parameters
    ----------
    input_shapes : dict
        Keys: Input layer names.
        Values: Their shapes.

    Returns
    -------
    inputs : List
        A list of named keras input layers.

    """
    inputs = []
    for input_name, input_shape in input_shapes.items():
        inputs.append(layers.Input(shape=input_shape, name=input_name, dtype=backend.floatx()))
    return inputs
