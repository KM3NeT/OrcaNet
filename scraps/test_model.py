from keras.models import Model, clone_model
from keras.layers import Input, Dense, Concatenate, Dropout
import numpy as np
from keras import backend as K

from orcanet.utilities.nn_utilities import get_inputs


def build_test_model():
    inp_1 = Input((1,), name="inp_0")
    inp_2 = Input((1,), name="inp_1")

    x = Concatenate()([inp_1, inp_2])
    x = Dense(3)(x)

    output_1 = Dense(1, name="out_0")(x)
    output_2 = Dense(2, name="out_1")(x)

    test_model = Model((inp_1, inp_2), (output_1, output_2))
    return test_model


def dropout_test():
    def dropout_model(rate=0.):
        inp = Input((10,))
        out = Dropout(rate)(inp)
        model = Model(inp, out)
        return model

    def get_layer_output(model, xs, which=-1):
        l_out = K.function([model.layers[0].input, K.learning_phase()], [model.layers[which].output])
        # output in train mode = 1
        layer_output = l_out([xs, 1])[0]
        return layer_output

    model0 = dropout_model(0.)
    model1 = dropout_model(0.99)
    xs = np.ones((3, 10))

    print("no drop\n", get_layer_output(model0, xs))
    print("\nmax drop\n", get_layer_output(model1, xs))
    model1.layers[-1].rate = 0.
    print("\nchanged max drop to zero\n", model1.layers[-1].get_config())
    print(get_layer_output(model1, xs))
    model1_clone = clone_model(model1)
    print("\n clone changed model\n", get_layer_output(model1_clone, xs))


def get_structured_array():
    x = np.array([(9, 81.0), (3, 18),  (3, 18),  (3, 18)],
                 dtype=[('inp_0', 'f4'), ('inp_1', 'f4')])
    return x


def get_dict():
    x = {"inp_0": np.array([9, 3]), "inp_1": np.array([81, 18])}
    return x


def transf_arr(x):
    xd = {name: x[name] for name in x.dtype.names}
    return xd


def build_single_inp():
    inp = Input((2,), name="inp")

    x = Dense(3)(inp)

    output_1 = Dense(1, name="out_0")(x)
    output_2 = Dense(2, name="out_1")(x)

    test_model = Model(inp, (output_1, output_2))
    return test_model


def get_xs(model, batchsize=1):
    shapes = model.input_shape
    if len(model.input_names) == 1:
        shapes = [shapes, ]
    xs = {model.input_names[i]: np.ones([batchsize, ] + list(shapes[i][1:])) for i in range(len(model.input_names))}
    return xs


def get_activations_and_weights(xs, model, layer_name=None, learning_phase='test'):
    from keras.layers import InputLayer
    layer_names = [layer.name for layer in model.layers if
                   layer.name == layer_name or layer_name is None]

    weights = [layer.get_weights() for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    layer_nos = []
    for layer_no, layer in enumerate(model.layers):
        if layer_name is None or layer.name == layer_name:
            if not isinstance(layer, InputLayer):
                layer_nos.append(layer_no)

    # get input layers and input files
    inputs = get_inputs(model)
    # doesnt work with dicts for whatever reason so transform into lists instead
    model_inputs = [xs[key] for key in inputs.keys()]

    activations = [get_layer_output(model, model_inputs, layer_no, learning_phase) for layer_no in layer_nos]

    return layer_names, activations, weights


def get_layer_output(model, samples, layer_no, mode="test"):
    # samples : list
    if mode == "test":
        phase = 0
    elif mode == "train":
        phase = 1
    else:
        raise NameError("Unknown mode: ", mode)

    inp_tensors = model.input  # either a tensor or a list of tensors
    if not isinstance(inp_tensors, list):
        inp_tensors = [inp_tensors, ]
    if not isinstance(samples, list):
        samples = [samples, ]
    get_output = K.function(
        inp_tensors + [K.learning_phase(), ],
        [model.layers[layer_no].output])
    layer_output = get_output(samples + [phase, ])[0]
    return layer_output


model = build_single_inp()
model2 = build_test_model()

xs = get_xs(model)
get_layer_output(model, 3, get_xs(model))
get_layer_output(model2, 3, get_xs(model2))
"""


get_activations_and_weights(get_xs(model), model)
get_activations_and_weights(get_xs(model2), model2)
"""
"""
layer_name=None
learning_phase='test'


outputs = [layer.output for layer in model.layers if
           layer.name == layer_name or layer_name is None]  # all layer outputs -> empty tf.tensors
layer_names = [layer.name for layer in model.layers if
               layer.name == layer_name or layer_name is None]
weights = [layer.get_weights() for layer in model.layers if
           layer.name == layer_name or layer_name is None]
outputs = outputs[1:]  # remove the first input_layer from fetch

# get input layers and input files
inputs = get_inputs(model)

# doesnt work with dicts for whatever reason so transform into lists instead
keys = list(inputs.keys())
inp = [inputs[key] for key in keys]
model_inputs = [xs[key] for key in keys]
if len(inp) == 1:
    inp = inp[0]
    model_inputs = model_inputs[0]

funcs = [K.function([inp, K.learning_phase()], [out]) for out in
         outputs]  # evaluation functions

lp = 0. if learning_phase == 'test' else 1.
list_inputs = [model_inputs, lp]

layer_outputs = [func(list_inputs)[0] for func in funcs]
activations = []
for layer_activations in layer_outputs:
    activations.append(layer_activations)
"""