from keras.layers import Input, Dense, Dropout, Activation, Convolution3D, \
    BatchNormalization, MaxPooling3D, Convolution2D, MaxPooling2D
from keras import backend as K
from keras.regularizers import l2


class ConvBlock:
    """
    2D/3D Convolutional block followed by BatchNorm, Activation,
    MaxPooling and/or Dropout.

    Attributes
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
    def __init__(self, conv_dim,
                 filters,
                 kernel_size=3,
                 strides=1,
                 pool_size=None,
                 pool_padding="valid",
                 dropout=None,
                 activation='relu',
                 kernel_l2_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.dropout = dropout
        self.activation = activation
        self.kernel_l2_reg = kernel_l2_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        if self.conv_dim == 2:
            convolution_nd = Convolution2D
            max_pooling_nd = MaxPooling2D
        elif self.conv_dim == 3:
            convolution_nd = Convolution3D
            max_pooling_nd = MaxPooling3D
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
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = Activation(self.activation)(x)
        if self.pool_size is not None:
            x = max_pooling_nd(pool_size=self.pool_size,
                               padding=self.pool_padding)(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)

        return x


class DenseBlock:
    """
    Dense layer followed by BatchNorm, Activation and/or Dropout.

    Attributes
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
            kernel_reg = l2(self.kernel_l2_reg)
        else:
            kernel_reg = None

        if self.batchnorm:
            use_bias = False
        else:
            use_bias = True

        x = Dense(units=self.units,
                  use_bias=use_bias,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=kernel_reg)(inputs)

        if self.batchnorm:
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = Activation(self.activation)(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)
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
        inputs.append(Input(shape=input_shape, name=input_name, dtype=K.floatx()))
    return inputs
