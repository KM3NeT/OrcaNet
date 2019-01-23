"""
For testing how long functions take.
"""

import time
import numpy as np
import h5py
from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file, load_zero_center_data
from orcanet.core import Configuration


def main():
    loops = 3

    cfg = Configuration("./test/", "/home/woody/capn/mppi013h/Code/OrcaNet/examples/settings_files/example_list_new_dataformat.toml")
    cfg.zero_center_folder = "/home/woody/capn/mppi013h/Code/work/zero_center_folder/"
    train_files = cfg.get_train_files()
    xs_mean = load_zero_center_data(cfg)
    steps = int(train_files[0][1]/cfg.batchsize)
    file = train_files[0][0]
    print(steps, "steps")
    f = h5py.File(file[0])
    print(f["x"].shape)
    f.close()

    start = time.time()
    # ------------------------------------------------------
    for n in range(loops):
        print("Loop no.", n+1)
        generator = generate_batches_from_hdf5_file(cfg, file, zero_center_image=xs_mean)
        for i in range(steps):
            next(generator)
    # ------------------------------------------------------
    end = time.time()
    print(end - start)


main()
