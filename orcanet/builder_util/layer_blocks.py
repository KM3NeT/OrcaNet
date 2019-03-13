from keras.layers import Input, Dense, Dropout, Activation, Convolution3D, \
    BatchNormalization, MaxPooling3D, Convolution2D, MaxPooling2D
from keras import backend as K


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
    kernel_size : int
        Kernel size which is used for all three dimensions.
    pool_size : None or tuple
        Specifies if a MaxPooling layer should be added. e.g. (1,1,2)
        -> sizes for a 3D conv block.
    dropout : float or None
        Adds a dropout layer if the value is not None.
    activation : str or None
        Type of activation function that should be used. E.g. 'linear',
        'relu', 'elu', 'selu'.
    kernel_reg : str or None
        If L2 regularization with 1e-4 should be employed. 'l2' to enable
        the regularization.
    batchnorm : bool
        Adds a batch normalization layer.
    kernel_initializer : string
        Initializer for the kernel weights.

    """
    def __init__(self, conv_dim,
                 filters,
                 kernel_size=3,
                 pool_size=None,
                 dropout=None,
                 activation='relu',
                 kernel_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        self.conv_dim = conv_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.activation = activation
        self.kernel_reg = kernel_reg
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

        x = convolution_nd(self.filters,
                           self.kernel_size,
                           padding='same',
                           kernel_initializer=self.kernel_initializer,
                           use_bias=False,
                           kernel_regularizer=self.kernel_reg)(inputs)

        if self.batchnorm:
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            x = BatchNormalization(axis=channel_axis)(x)
        if self.activation is not None:
            x = Activation(self.activation)(x)
        if self.pool_size is not None:
            x = max_pooling_nd(pool_size=self.pool_size, padding='valid')(x)
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
    kernel_reg : str or None
        If L2 regularization with 1e-4 should be employed. 'l2' to enable
        the regularization.
    batchnorm : bool
        Adds a batch normalization layer.

    """
    def __init__(self, units,
                 dropout=None,
                 activation='relu',
                 kernel_reg=None,
                 batchnorm=False,
                 kernel_initializer="he_normal"):
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.kernel_reg = kernel_reg
        self.batchnorm = batchnorm
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        x = Dense(self.units, kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_reg)(inputs)
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
