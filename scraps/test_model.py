from tensorflow.keras.models import Model, clone_model
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras import backend as K


def build_double_inp(compile=False):
    inp_1 = layers.Input((1,), name="inp_0")
    inp_2 = layers.Input((1,), name="inp_1")

    x = layers.Concatenate()([inp_1, inp_2])
    x = layers.Dense(3)(x)

    output_1 = layers.Dense(1, name="out_0")(x)
    output_2 = layers.Dense(2, name="out_1")(x)

    test_model = Model((inp_1, inp_2), (output_1, output_2))

    if compile:
        test_model.compile("adam", loss={"out_0": "mse", "out_1": "mse"},
                           metrics={"out_0": "mae", "out_1": "mae"})

    return test_model


def get_xys():
    x = {"inp_0": np.array([9, 3]), "inp_1": np.array([81, 18])}
    y = {"out_0": np.array([1, 1]), "out_1": np.array([[0,0], [0,0]])}
    return x, y


def build_single_inp():
    inp = layers.Input((2,), name="inp")

    x = layers.Dense(3)(inp)

    output_1 = layers.Dense(1, name="out_0")(x)
    output_2 = layers.Dense(2, name="out_1")(x)

    test_model = Model(inp, (output_1, output_2))
    return test_model


def get_xs(model, batchsize=1):
    """ Get dummy data fitting a model. """
    shapes = model.input_shape
    if len(model.input_names) == 1:
        shapes = [shapes, ]
    xs = {model.input_names[i]: np.ones([batchsize, ] + list(shapes[i][1:]))
          for i in range(len(model.input_names))}
    return xs


def dropout_test():
    def dropout_model(rate=0.):
        inp = layers.Input((10,))
        out = layers.Dropout(rate)(inp)
        model = Model(inp, out)
        return model

    def get_layer_output(model, xs, which=-1):
        l_out = K.function([model.layers[0].input, K.learning_phase()],
                           [model.layers[which].output])
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
