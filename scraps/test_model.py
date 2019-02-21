from keras.models import Model, clone_model
from keras.layers import Input, Dense, Concatenate, Dropout
import numpy as np
from keras import backend as K


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


x = get_structured_array()
xd = get_dict()
model = build_test_model()
model.summary()
y_pred = model.predict_on_batch(x)
y_pred[0] = np.reshape(y_pred[0], y_pred[0].shape[:-1])
# y_pred = {out.name.split(':')[0]: y_pred[i] for i, out in enumerate(model.outputs)}
dtypes = np.dtype([("out_0", "float32"), ("out_1", "float32", (2,2))])

result = np.array(y_pred, dtypes)

a = np.array( [[1,2], [3,4], [5,6]] )
aty = np.dtype([("asd", "float32", (2,1))])
b = np.array(a, aty)
