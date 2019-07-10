"""
For testing how long generators take.

Will show the average time per batch a generator takes, as well as the average training time per batch of a model.

Shuffle test:
-------------
Generator shuffles the order in which data gets read out.
Tested on 5000 batches each of (64, 11, 13, 18, 60) xyzt data with the vgg model with 3,000,000 params.
Results:

            Read-time per batch (s)    Train time per batch (s)
No Shuffle        0.07755                    0.2017
Shuffle           0.1276                     0.198

+65% due to shuffle! Still lower than train time though.

"""
import os
import time
import numpy as np
import h5py
from orcanet.core import Organizer
from model_builder import build_nn_model


def generate_batches_from_hdf5_file_tweak(cfg, files_dict, f_size=None, zero_center_image=None, yield_mc_info=False, shuffle=False, batches_at_once=10):
    """
    Yields batches of input data from h5 files.

    This will go through one file, or multiple files in parallel, and yield one batch of data, which can then
    be used as an input to a model. Since multiple filepaths can be given to read out in parallel,
    this can also be used for models with multiple inputs.
    # TODO Is reading n batches at once and yielding them one at a time faster then what we have currently?

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    files_dict : dict
        The name of every input as a key (can be multiple), the filepath of a h5py file to read samples from as values.
    f_size : int or None
        Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
        should be used for yielding the xs/ys arrays. This is important if you run fit_generator(epochs>1) with
        a filesize (and hence # of steps) that is smaller than the .h5 file.
    zero_center_image : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.
    yield_mc_info : bool
        Specifies if mc-infos (y_values) should be yielded as well.
        The mc-infos are used for evaluation after training and testing is finished.
    shuffle : bool
        Randomize the order in which batches are read from the file. Significantly reduces read out speed.

    Yields
    ------
    xs : dict
        Data for the model train on.
    ys : dict
        Labels for the model to train on.
    y_values : ndarray, optional
        y_values from the file. Only yielded if yield_mc_info is True.

    """
    batchsize = cfg.batchsize
    # name of the datagroups in the file
    samples_key = cfg.key_samples
    mc_key = cfg.key_y_values

    # If the batchsize is larger than the f_size, make batchsize smaller or nothing would be yielded
    if f_size is not None:
        if f_size < batchsize:
            batchsize = f_size

    while 1:
        # a dict with the names of list inputs as keys, and the opened h5 files as values.
        files = {}
        file_lengths = []
        # open the files and make sure they have the same length
        for input_key in files_dict:
            files[input_key] = h5py.File(files_dict[input_key], 'r')
            file_lengths.append(len(files[input_key][samples_key]))
        if not file_lengths.count(file_lengths[0]) == len(file_lengths):
            raise AssertionError("All data files must have the same length! Yours have:\n " + str(file_lengths))

        if f_size is None:
            f_size = file_lengths[0]
        # number of full batches available
        total_no_of_batches = int(f_size/batchsize)
        # positions of the samples in the file
        sample_pos = np.arange(total_no_of_batches, step=batches_at_once) * batchsize
        if shuffle:
            np.random.shuffle(sample_pos)

        for sample_n in sample_pos:
            # Read one batch of samples from the files and zero center
            # A dict with every input name as key, and a batch of data as values
            xs = {}
            for input_key in files:
                xs[input_key] = files[input_key][samples_key][sample_n: sample_n + batches_at_once*batchsize]
                if zero_center_image is not None:
                    xs[input_key] = np.subtract(xs[input_key], zero_center_image[input_key])
            # Get labels for the nn. Since the labels are hopefully the same for all the files, use the ones from the first
            y_values = list(files.values())[0][mc_key][sample_n:sample_n + batches_at_once*batchsize]

            for bsample_no in np.arange(batches_at_once)*batchsize:
                bxs = {}
                for input_key in xs:
                    bxs[input_key] = xs[input_key][bsample_no: bsample_no + batchsize]
                by_values = y_values[bsample_no: bsample_no + batchsize]

                # Modify the samples and the labels batchwise
                if cfg.sample_modifier is not None:
                    bxs = cfg.sample_modifier(bxs)

                # if swap_col is not None:
                #     xs = get_input_images(xs, swap_col, str_ident)
                bys = by_values  # get_labels(by_values, class_type)

                if not yield_mc_info:
                    yield bxs, bys
                else:
                    yield bxs, bys, by_values

        # for i in range(n_files):
        #     files[i].close()


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
    modelfile = "/home/woody/capn/mppi013h/Code/OrcaNet/examples/settings_files/explanation.toml"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    orca = Organizer("./test/", list_file)
    orca.cfg.zero_center_folder = zero_center_folder
    train_files = next(orca.io.yield_files("train"))
    # cfg.use_local_node()
    print(batches, "batches from file", train_files)
    print("Batchsize:", orca.cfg.batchsize)
    f = h5py.File(list(train_files.values())[0], "r")
    print(f["x"].shape)
    xs_mean = None  # np.ones(f["x"].shape[1:])  # load_zero_center_data(cfg)
    f.close()

    orca.cfg.import_model_file(modelfile)
    model = build_nn_model(orca)
    # model.summary()

    # No large dataset in the new ys format exists, so use dummy data to train on instead
    ys = np.ones((orca.cfg.batchsize, 16))
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
        generator = func(orca.cfg, train_files, zero_center_image=xs_mean, **func_kwargs[f_no])
        print("Shape of batches:", list(next(generator)[0].values())[0].shape, "\n")
        average_read_time, average_model_time = [], []
        for i in range(batches):
            read_start = time.time()
            x = list(next(generator)[0].values())[0]
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
    batches = 500
    intermediate_log = 100
    functions = (generate_batches_from_hdf5_file_tweak, generate_batches_from_hdf5_file_tweak)
    func_kwargs = [{"batches_at_once": 10, "shuffle": True}, {"batches_at_once": 10, }]

    test_generators(batches, intermediate_log, functions, func_kwargs)
