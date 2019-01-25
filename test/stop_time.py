"""
For testing how long generators take.

Will show the average time per batch a generator takes, as well as the average training time per batch of a model.
"""
import os
import time
import numpy as np
import h5py
from orcanet.utilities.nn_utilities import generate_batches_from_hdf5_file, load_zero_center_data, get_labels
from orcanet.core import Configuration
from orcanet.model_archs.model_setup import build_nn_model


def test_generators(batches, intermediate_log, functions, func_kwargs):
    """
    Test the time of generators.

    Parameters
    ----------
    batches : int
        how many batches to read out
    intermediate_log : int
        show current speeds after this many batches
    functions : list
        functions to test
    func_kwargs : list
        kwargs for these functions

    """

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
    # model.summary()

    # No large dataset in the new ys format exists, so use dummy data to train on instead
    ys = np.ones((cfg.batchsize, 16))
    dtypes = [('event_id', '<f8'), ('particle_type', '<f8'), ('energy', '<f8'), ('is_cc', '<f8'), ('bjorkeny', '<f8'),
              ('dir_x', '<f8'), ('dir_y', '<f8'), ('dir_z', '<f8'), ('time_interaction', '<f8'), ('run_id', '<f8'),
              ('vertex_pos_x', '<f8'), ('vertex_pos_y', '<f8'), ('vertex_pos_z', '<f8'),
              ('time_residual_vertex', '<f8'),
              ('prod_ident', '<f8'), ('group_id', '<i8')]
    ys = ys.ravel().view(dtype=dtypes)
    ys = get_labels(ys, [0, 'energy_dir_bjorken-y_vtx_errors'])

    print("\n")
    for f_no, func in enumerate(functions):
        print("------------- Function", f_no, " -------------")
        generator = func(cfg, file, zero_center_image=xs_mean, **func_kwargs[f_no])
        print("Shape of batches:", next(generator)[0][0].shape, "\n")
        average_read_time, average_model_time = [], []
        for i in range(batches):
            read_start = time.time()
            x = next(generator)[0][0]
            average_read_time.append(time.time() - read_start)

            model_start = time.time()
            model.train_on_batch(x, ys)
            average_model_time.append((time.time() - model_start))

            if i % intermediate_log == 0 and i != 0:
                print("At", i, "batches:  ", end="")
                print("Last {} batches:\tRead: {:.4g}\tTrain: {:.4g}".format(intermediate_log, np.average(average_read_time[i-200:i]), np.average(average_model_time[i-200:i])))
                # print("Total:\tRead:", np.average(average_read_time), "predict:", np.average(average_model_time))

        print("\nAverage read time per batch:\t{:.4g}".format(np.average(average_read_time)))
        print("Average train time per batch:\t{:.4g}".format(np.average(average_model_time)))


if __name__ == "__main__":
    batches = 5000
    intermediate_log = 200
    functions = (generate_batches_from_hdf5_file, generate_batches_from_hdf5_file)
    func_kwargs = [{}, {"shuffle": True}]

    test_generators(batches, intermediate_log, functions, func_kwargs)
