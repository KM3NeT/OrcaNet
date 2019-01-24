"""
For testing how long functions take.
"""
import os
import time
import numpy as np
import h5py
from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file, load_zero_center_data, generate_batches_from_hdf5_file_shuffle
from orcanet.core import Configuration
from orcanet.model_archs.model_setup import build_nn_model


def main():
    batches = 100  # int(train_files[0][1]/cfg.batchsize)
    model_batches = 50
    functions = (generate_batches_from_hdf5_file_shuffle, generate_batches_from_hdf5_file_shuffle)
    args = [{"shuffle": True}, {"shuffle": False}]

    list_file = "/home/woody/capn/mppi013h/Code/OrcaNet/test/time_test_list.toml"
    zero_center_folder = "/home/woody/capn/mppi013h/Code/work/zero_center_folder/"
    modelfile = "/home/woody/capn/mppi013h/Code/OrcaNet/examples/settings_files/example_model.toml"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg = Configuration("./test/", list_file)
    cfg.zero_center_folder = zero_center_folder
    train_files = cfg.get_train_files()
    file = train_files[0][0]
    # cfg.use_local_node()
    print(batches, "batches from file", cfg.get_train_files())
    print("Batchsize:", cfg.batchsize)
    f = h5py.File(file[0])
    print(f["x"].shape)
    xs_mean = np.ones(f["x"].shape[1:])  # load_zero_center_data(cfg)
    f.close()

    cfg.set_from_model_file(modelfile)
    model = build_nn_model(cfg)
    model.summary()

    print("\n")
    for f_no, func in enumerate(functions):
        generator = func(cfg, file, zero_center_image=xs_mean, **args[f_no])
        print("Shape of batches:", next(generator)[0][0].shape)

        print("Reading out", model_batches, "batches...")
        batches_for_model = []
        start = time.time()
        for i in range(model_batches):
            batches_for_model.append(next(generator)[0][0])
        print(((time.time() - start) / model_batches), "per batch.\n")

        print("Model predicting...")
        start = time.time()
        for batch in batches_for_model:
            print(batch)
            model.predict_on_batch(batch)
        print((time.time()-start)/model_batches, "per batch.\n")

        print("Data is being read out from h5 file...")
        start = time.time()
        for i in range(batches):
            if i == 0:
                temp_start = time.time()
            elif i % 200 == 0:
                temp_time = time.time()
                print("At", i, "batches: ", (temp_time - temp_start)/(200*cfg.batchsize), "per batch")
                temp_start = temp_time
            _ = next(generator)
        end = time.time()
        print("\nTotal time:", end - start)
        print("Average per batch:", (end - start)/(batches * cfg.batchsize), "\n")
        print("\n----------------------------------------\n")



main()

