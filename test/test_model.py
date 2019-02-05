from keras.models import Model
from keras.layers import Input, Dense, Concatenate
import numpy as np


def build_test_model():
    inp_1 = Input((1,), name="inp_0")
    inp_2 = Input((1,), name="inp_1")

    x = Concatenate()([inp_1, inp_2])
    x = Dense(3)(x)

    output_1 = Dense(1, name="out_0")(x)
    output_2 = Dense(2, name="out_1")(x)

    test_model = Model((inp_1, inp_2), (output_1, output_2))
    return test_model


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
