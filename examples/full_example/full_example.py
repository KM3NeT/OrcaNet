import numpy as np
import h5py
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

from orcanet.core import Organizer


def make_dummy_data(path):
    """
    Save a train and a val h5 file with random numbers as samples,
    and the sum as labels.

    """
    def get_dummy_data(samples):
        xs = np.random.rand(samples, 3)

        dtypes = [('sum', '<f8'), ]
        labels = np.sum(xs, axis=-1)
        ys = labels.ravel().view(dtype=dtypes)

        return xs, ys

    def save_dummy_data(path, samples):
        xs, ys = get_dummy_data(samples)

        with h5py.File(path, 'w') as h5f:
            h5f.create_dataset('x', data=xs, dtype='<f8')
            h5f.create_dataset('y', data=ys, dtype=ys.dtype)
    save_dummy_data(path + "example_train.h5", 40000)
    save_dummy_data(path + "example_val.h5", 5000)


def make_dummy_model():
    """
    Build and compile a small dummy model.
    """

    input_shape = (3,)

    inp = Input(input_shape, name="random_numbers")
    x = Dense(10)(inp)
    outp = Dense(1, name="sum")(x)

    model = Model(inp, outp)
    model.compile("sgd", loss="mae")

    return model


def use_orcanet():
    temp_folder = "output/"
    os.mkdir(temp_folder)

    make_dummy_data(temp_folder)
    list_file = "example_list.toml"

    organizer = Organizer(temp_folder + "sum_model", list_file)
    organizer.cfg.train_logger_display = 10

    model = make_dummy_model()
    organizer.train_and_validate(model, epochs=3)

    organizer.predict()


use_orcanet()

