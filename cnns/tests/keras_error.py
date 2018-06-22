from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution3D, BatchNormalization
from keras import backend as K
import numpy as np


def create_cnn(nb_classes):
    """
    Returns a cnn model.
    """
    input_layer = Input(shape=(11,13,18,60), dtype=K.floatx())  # input_layer

    x = Convolution3D(64, (3,) * 3, use_bias=False)(input_layer)

    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    label_names = ['neuron_' + str(i) for i in range(nb_classes)]
    x = [Dense(1, name=name)(x) for name in label_names]

    model = Model(inputs=input_layer, outputs=x)

    return model


batchsize = 32
model_input = np.random.randint(0, 255, (32,11,13,18,60))

nn_model = create_cnn(5)
inp = nn_model.input
inp = [inp]  # only one input! let's wrap it in a list.
lp = 0.

layer_name = None
outputs = [layer.output for layer in nn_model.layers if
           layer.name == layer_name or layer_name is None]  # all layer outputs -> empty tf.tensors

funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

list_inputs = [model_input, lp]
layer_outputs = [func(list_inputs)[0] for func in funcs]
