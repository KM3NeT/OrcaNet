"""
For testing how long functions take.
"""

import time
import numpy as np
import h5py
from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file, load_zero_center_data, generate_batches_from_hdf5_file_shuffle
from orcanet.core import Configuration


def main():
    cfg = Configuration("./test/", "/home/woody/capn/mppi013h/Code/OrcaNet/test/time_test_list.toml")
    cfg.zero_center_folder = "/home/woody/capn/mppi013h/Code/work/zero_center_folder/"
    train_files = cfg.get_train_files()
    batches = 1000  # int(train_files[0][1]/cfg.batchsize)
    file = train_files[0][0]
    print(batches, "batches from file", cfg.get_train_files())
    print("Batchsize:", cfg.batchsize)
    f = h5py.File(file[0])
    print(f["x"].shape)
    xs_mean = np.ones(f["x"].shape[1:])  # load_zero_center_data(cfg)
    f.close()

    functions = (generate_batches_from_hdf5_file, generate_batches_from_hdf5_file_shuffle, generate_batches_from_hdf5_file_shuffle)
    args = [{}, {"shuffle": False}, {"shuffle": True}]

    print("\n")
    for f_no, func in enumerate(functions):
        start = time.time()
        # ------------------------------------------------------
        generator = func(cfg, file, zero_center_image=xs_mean, **args[f_no])
        for i in range(batches):
            if i == 0:
                temp_start = time.time()
            elif i % 200 == 0:
                temp_time = time.time()
                print("At", i, "batches: ", (temp_time - temp_start)/(200*cfg.batchsize), "per batch")
                temp_start = temp_time
            next(generator)
        # ------------------------------------------------------
        end = time.time()
        print("\nTotal time:", end - start)
        print("Average per batch:", (end - start)/(batches * cfg.batchsize), "\n")
        print("\n----------------------------------------\n")

main()

