from keras.models import Model
from keras.layers import Input, Dense, Concatenate


def build_test_model():
    inp_1 = Input((1,), name="inp_0")
    inp_2 = Input((2,), name="inp_1")

    x = Concatenate()([inp_1, inp_2])
    x = Dense(3)(x)

    output_1 = Dense(1, name="out_0")(x)
    output_2 = Dense(2, name="out_1")(x)

    test_model = Model((inp_1, inp_2), (output_1, output_2))
    return test_model


model = build_test_model()
