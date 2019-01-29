#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import warnings
import re
import numpy as np
import h5py
import os
import keras as ks
from functools import reduce

# ------------- Functions used for supplying images to the GPU -------------#


def generate_batches_from_hdf5_file(cfg, files_dict, f_size=None, zero_center_image=None, yield_mc_info=False, shuffle=False):
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
    mc_info : ndarray, optional
        Mc info from the file. Only yielded if yield_mc_info is True.

    """
    batchsize = cfg.batchsize
    class_type = cfg.class_type
    str_ident = cfg.str_ident
    swap_col = cfg.swap_4d_channels
    # name of the datagroups in the file
    samples_key = cfg.key_samples
    mc_key = cfg.key_labels

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
        sample_pos = np.arange(total_no_of_batches) * batchsize
        if shuffle:
            np.random.shuffle(sample_pos)

        for sample_n in sample_pos:
            # Read one batch of samples from the files and zero center
            # A dict with every input name as key, and a batch of data as values
            xs = {}
            for input_key in files:
                xs[input_key] = files[input_key][samples_key][sample_n: sample_n + batchsize]
                if zero_center_image is not None:
                    xs[input_key] = np.subtract(xs[input_key], zero_center_image[input_key])
            # Get labels for the nn. Since the labels are hopefully the same for all the files, use the ones from the first
            y_values = list(files.values())[0][mc_key][sample_n:sample_n + batchsize]

            # Modify the samples and the labels batchwise
            if cfg.sample_modifier is not None:
                xs = cfg.sample_modifier(xs)

            # if swap_col is not None:
            #     xs = get_input_images(xs, swap_col, str_ident)
            ys = y_values  # get_labels(y_values, class_type)

            if not yield_mc_info:
                yield xs, ys
            else:
                yield xs, ys, y_values

        # for i in range(n_files):
        #     files[i].close()


def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    if n_bins_x == 1:
        if n_bins_y == 1:
            print('Using 2D projected data without dimensions x and y')
            dimensions = (batchsize, n_bins_z, n_bins_t, 1)
        elif n_bins_z == 1:
            print('Using 2D projected data without dimensions x and z')
            dimensions = (batchsize, n_bins_y, n_bins_t, 1)
        elif n_bins_t == 1:
            print('Using 2D projected data without dimensions x and t')
            dimensions = (batchsize, n_bins_y, n_bins_z, 1)
        else:
            print('Using 3D projected data without dimension x')
            dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)

    elif n_bins_y == 1:
        if n_bins_z == 1:
            print('Using 2D projected data without dimensions y and z')
            dimensions = (batchsize, n_bins_x, n_bins_t, 1)
        elif n_bins_t == 1:
            print('Using 2D projected data without dimensions y and t')
            dimensions = (batchsize, n_bins_x, n_bins_z, 1)
        else:
            print('Using 3D projected data without dimension y')
            dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)

    elif n_bins_z == 1:
        if n_bins_t == 1:
            print('Using 2D projected data without dimensions z and t')
            dimensions = (batchsize, n_bins_x, n_bins_y, 1)
        else:
            print('Using 3D projected data without dimension z')
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)

    elif n_bins_t == 1:
        print('Using 3D projected data without dimension t')
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    else:
        # print 'Using full 4D data'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions


def get_labels(y_values, class_type):
    """
    TODO add docs

    CAREFUL: y_values is a structured numpy array! if you use advanced numpy indexing, this may lead to errors.
    Let's suppose you want to assign a particular value to one or multiple elements of the y_values array.

    E.g.
    y_values[1]['bjorkeny'] = 5
    This works, since it is basic indexing.

    Likewise,
    y_values[1:3]['bjorkeny'] = 5
    works as well, because basic indexing gives you a view (!).

    Advanced indexing though, gives you a copy.
    So this
    y_values[[1,2,4]]['bjorkeny'] = 5
    will NOT work! Same with boolean indexing, like

    bool_idx = np.array([True,False,False,True,False]) # if len(y_values) = 5
    y_values[bool_idx]['bjorkeny'] = 10
    This will NOT work as well!!

    Instead, use
    np.place(y_values['bjorkeny'], bool_idx, 10)
    This works.

    Parameters
    ----------
    y_values : ndarray
        TODO
    class_type : tuple(int, str) TODO is the int still needed anywhere?

    Returns
    -------
    ys : dict
        TODO

    """
    ys = dict()

    if class_type[1] == 'energy_dir_bjorken-y_vtx_errors':
        particle_type, is_cc = y_values['particle_type'], y_values['is_cc']
        elec_nc_bool_idx = np.logical_and(np.abs(particle_type) == 12, is_cc == 0)

        # correct energy to visible energy
        visible_energy = y_values[elec_nc_bool_idx]['energy'] * y_values[elec_nc_bool_idx]['bjorkeny']
        # fix energy to visible energy
        np.place(y_values['energy'], elec_nc_bool_idx, visible_energy)  # TODO fix numpy warning
        # set bjorkeny label of nc events to 1
        np.place(y_values['bjorkeny'], elec_nc_bool_idx, 1)

        ys['dx'], ys['dx_err'] = y_values['dir_x'], y_values['dir_x']
        ys['dy'], ys['dy_err'] = y_values['dir_y'], y_values['dir_y']
        ys['dz'], ys['dz_err'] = y_values['dir_z'], y_values['dir_z']
        ys['e'], ys['e_err'] = y_values['energy'], y_values['energy']
        ys['by'], ys['by_err'] = y_values['bjorkeny'], y_values['bjorkeny']

        ys['vx'], ys['vx_err'] = y_values['vertex_pos_x'], y_values['vertex_pos_x']
        ys['vy'], ys['vy_err'] = y_values['vertex_pos_y'], y_values['vertex_pos_y']
        ys['vz'], ys['vz_err'] = y_values['vertex_pos_z'], y_values['vertex_pos_z']
        ys['vt'], ys['vt_err'] = y_values['time_residual_vertex'], y_values['time_residual_vertex']

        for key_label in ys:
            ys[key_label] = ys[key_label].astype(np.float32)

    elif class_type[1] == 'track-shower':
        # for every sample, [0,1] for shower, or [1,0] for track

        # {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}
        # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
        # label is always shower, except if muon-CC
        particle_type, is_cc = y_values['particle_type'], y_values['is_cc']
        is_muon_cc = np.logical_and(np.abs(particle_type) == 14, is_cc == 1)
        is_not_muon_cc = np.invert(is_muon_cc)

        batchsize = y_values.shape[0]
        categorical_ts = np.zeros((batchsize, 2), dtype='bool')  # categorical [shower, track] -> [1,0] = shower, [0,1] = track
        categorical_ts[is_not_muon_cc][:, 0] = 1
        categorical_ts[is_muon_cc][:, 1] = 1

        ys['ts_output'] = categorical_ts.astype(np.float32)

    else:
        raise ValueError('The label ' + str(class_type[1]) + ' in class_type[1] is not available.')

    return ys


def get_inputs(model):
    """
    Get the names and the layers of the inputs of the model.

    Parameters
    ----------
    model : ks.model
        A keras model.

    Returns
    -------
    layers :dict
        The input layers and names.

    """
    from keras.layers import InputLayer
    layers = {}
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            layers[layer.name] = layer
    return layers


# ------------- Functions used for supplying images to the GPU -------------#


# ------------- Functions for preprocessing -------------#

def load_zero_center_data(cfg):
    """
    Gets the xs_mean array(s) that can be used for zero-centering.
    TODO Test!

    The arrays are either loaded from a previously saved .npz file or they are calculated on the fly by
    calculating the mean value per bin for the given training files. The name of the saved image is derived from the
    name of the list file which was given to the cfg.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    Returns
    -------
    xs_mean : dict
        Dict of ndarray(s) that contains the mean_image of the x dataset (1 array per list input).
        Can be used for zero-centering later on.
        Example format:
        { "input_A" : ndarray, "input_B" : ndarray }

    """
    all_train_files = cfg.get_train_files()
    zero_center_folder = cfg.zero_center_folder
    train_files_list_name = os.path.basename(cfg.get_list_file())

    xs_mean = {}
    # loop over multiple input data files for a single event, each input needs its own xs_mean
    for input_key in all_train_files:
        # Collect all filepaths of the train_files for this projection in an array
        all_train_files_for_ip_i = all_train_files[input_key]
        # load the filepaths of all precalculated zero_center .npz files, which contain the xs_mean
        zero_center_files = load_fpaths_of_existing_zero_center_files(zero_center_folder)
        # get the xs_mean path for this input number i, if it exists in any of the files in the zero_center_folder
        xs_mean_for_ip_i_path = get_precalculated_xs_mean_if_exists(zero_center_files, all_train_files_for_ip_i)

        if xs_mean_for_ip_i_path is not None:
            print('Loading an existing zero center image for list input ' + str(input_key) +
                  ':\n   ' + xs_mean_for_ip_i_path)
            xs_mean_for_ip_i = np.load(xs_mean_for_ip_i_path)["xs_mean"]

        else:
            print('Calculating the xs_mean_array for list input ' + str(input_key) + ' in order to zero_center the data!')
            # if the train dataset is split over multiple files, we need to average over the single xs_mean_for_ip arrays.
            xs_mean_for_ip_i = get_mean_image(all_train_files_for_ip_i, cfg.key_samples, cfg.n_gpu[0])

            filename = zero_center_folder + train_files_list_name + '_input_' + str(input_key) + '.npz'
            np.savez(filename, xs_mean=xs_mean_for_ip_i, zero_center_used_ip_files=all_train_files_for_ip_i)
            print('Saved the xs_mean array for input ' + str(input_key) + ' with shape', xs_mean_for_ip_i.shape, ' to ', filename)

        xs_mean[input_key] = xs_mean_for_ip_i

    return xs_mean


def load_fpaths_of_existing_zero_center_files(zero_center_folder):
    """
    Loads the filepaths of all precalculated zero_center_files (.npz) in the zero_center_folder if they exist.

    Parameters
    ----------
    zero_center_folder : str
        Full path to the folder where the zero_centering arrays are / should be stored.

    Returns
    -------
    zero_center_files : list
        List that contains all filepaths of precalculated zero_center files.
        Can be empty, if no zero_center_files exist in that directory.

    """
    zero_center_files = []
    if os.path.isdir(zero_center_folder):
        for file in os.listdir(zero_center_folder):
            if file.endswith('.npz'):
                zero_center_files.append(zero_center_folder + file)
    else:
        os.mkdir(zero_center_folder)

    return zero_center_files


def get_precalculated_xs_mean_if_exists(zero_center_files, all_train_files_for_ip_i):
    """
    Function that searches for precalculated xs_mean arrays in the already existing zero_center_files.

    Specifically, the function opens every zero_center_file (.npz) and checks if the 'zero_center_used_ip_files' array
    is the same as the 'all_train_files_for_ip_i' array.

    Parameters
    ----------
    zero_center_files : list
        List that contains all filepaths of precalculated zero_center files.
    all_train_files_for_ip_i : list
        Contains the filepaths of all train_files for the i-th input.

    Returns
    -------
    xs_mean_for_ip_i : None/ndarray
        Returns the filepath to the xs_mean_for_ip_i array if it exists somewhere in the zero_center_files. If not, returns None.

    """
    xs_mean_for_ip_i = None
    for file in zero_center_files:
        zero_center_used_ip_files = np.load(file)['zero_center_used_ip_files']
        if np.array_equal(zero_center_used_ip_files, all_train_files_for_ip_i):
            xs_mean_for_ip_i = file
            break

    return xs_mean_for_ip_i


def get_mean_image(filepaths, key_samples, n_gpu):
    """
    Returns the mean_image of a xs dataset.
    Calculating still works if xs is larger than the available memory and also if the file is compressed!
    :param list filepaths: Filepaths of the data upon which the mean_image should be calculated.
    :param str key_samples: The name of the datagroup in your h5 input files which contains the samples to the network.
    :param int n_gpu: Number of used gpu's that is related to how much RAM is available (16G per GPU).
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    # check available memory and divide the mean calculation in steps
    total_memory = n_gpu * 8e9  # In bytes. Take 1/2 of what is available per GPU (16G), just to make sure.

    xs_means = []
    file_sizes = []
    for filepath in filepaths:
        file = h5py.File(filepath, "r")

        filesize = get_array_memsize(file['x'])
        steps = int(np.ceil(filesize/total_memory))
        n_rows = file[key_samples].shape[0]
        stepsize = int(n_rows / float(steps))

        # create xs_mean_arr that stores intermediate mean_temp results
        xs_mean_arr = np.zeros((steps, ) + file['x'].shape[1:], dtype=np.float64)
        print("Calculating the mean_image of the xs dataset for file: " + filepath)
        for i in range(steps):
            if i % 5 == 0:
                print('   Step ' + str(i) + " of " + str(steps))

            # for the last step, calculate mean till the end of the file
            if i == steps-1 or steps == 1:
                xs_mean_temp = np.mean(file[key_samples][i * stepsize: n_rows], axis=0, dtype=np.float64)
            else:
                xs_mean_temp = np.mean(file[key_samples][i*stepsize: (i+1) * stepsize], axis=0, dtype=np.float64)

            xs_mean_arr[i] = xs_mean_temp

        print("Done!")
        # The mean for this file
        xs_means.append(np.mean(xs_mean_arr, axis=0, dtype=np.float64).astype(np.float32))
        # the number of samples in this file
        file_sizes.append(n_rows)
    # calculate weighted average depending on no of samples in the files
    file_sizes = [size / np.sum(file_sizes) for size in file_sizes]
    xs_mean = np.average(xs_means, weights=file_sizes, axis=0)
    return xs_mean


def get_array_memsize(array):
    """
    Calculates the approximate memory size of an array.
    :param ndarray array: an array.
    :return: float memsize: size of the array in bytes.
    """
    shape = array.shape
    n_numbers = reduce(lambda x, y: x*y, shape)  # number of entries in an array
    precision = 8  # Precision of each entry, typically uint8 for xs datasets
    memsize = (n_numbers * precision) / float(8)  # in bytes

    return memsize


# ------------- Functions for preprocessing -------------#


# ------------- Various other functions -------------#

def get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident=''):
    """
    Derives the name of a model based on its number of bins and the class_type tuple.
    The final modelname is defined as 'model_Nd_proj_class_type[1]'.
    E.g. 'model_3d_xyz_muon-CC_to_elec-CC'.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) of the training images. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple that declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: String that declares which neural network model architecture is used.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies the projection type.
    :param str str_ident: Optional str identifier that gets appended to the modelname.
    :return: str modelname: Derived modelname.
    """
    modelname = 'model_' + nn_arch + '_'

    projection = ''
    for i, bins in enumerate(n_bins):

        dim = 4- bins.count(1)
        if i > 0: projection += '_and_'
        projection += str(dim) + 'd_'

        if bins.count(1) == 0 and i == 0: # for 4D input # TODO FIX BUG XYZT AFTER NAME
            if swap_4d_channels is not None:
                projection += swap_4d_channels
            else:
                projection += 'xyz-c' if bins[3] == 31 else 'xyz-t'

        else: # 2D/3D input
            if bins[0] > 1: projection += 'x'
            if bins[1] > 1: projection += 'y'
            if bins[2] > 1: projection += 'z'
            if bins[3] > 1: projection += 't'

    str_ident = '_' + str_ident if str_ident is not '' else str_ident
    modelname += projection + '_' + class_type[1] + str_ident

    return modelname

# ------------- Various other functions -------------#


# ------------- Classes -------------#

class TensorBoardWrapper(ks.callbacks.TensorBoard):
    """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
     Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
     Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.

     This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
     that the whole validation data is put into a single array by using the generator.
     Then, the single array is used in the validation steps. This workaround is experimental!"""
    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class BatchLevelPerformanceLogger(ks.callbacks.Callback):
    """
    Gibt loss aus über alle :display batches, gemittelt über die letzten :display batches
    TODO
    """

    def __init__(self, cfg, model, epoch):
        """

        Parameters
        ----------
        cfg : object Configuration
            Configuration object containing all the configurable options in the OrcaNet scripts.
        model
        epoch
        """
        ks.callbacks.Callback.__init__(self)
        self.display = cfg.train_logger_display
        self.epoch_number = epoch[0]
        self.f_number = epoch[1]
        self.model = model
        self.flush = cfg.train_logger_flush

        self.seen = 0
        self.logfile_train_fname = cfg.main_folder + 'log_train/log_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.txt'
        self.loglist = []

        self.cum_metrics = {}
        for metric in self.model.metrics_names:  # set up dict with all model metrics
            self.cum_metrics[metric] = 0

        self.steps_per_total_epoch, self.steps_cum = 0, [0]
        for f_size in cfg.get_train_file_sizes():
            steps_per_file = int(f_size / cfg.batchsize)
            self.steps_per_total_epoch += steps_per_file
            self.steps_cum.append(self.steps_cum[-1] + steps_per_file)

        with open(self.logfile_train_fname, 'w') as logfile_train:
            logfile_train.write('Batch\tBatch_float\t')
            for i, metric in enumerate(self.model.metrics_names):
                # write columns for all losses / metrics
                logfile_train.write(metric)
                if i + 1 < len(self.model.metrics_names): logfile_train.write('\t')  # newline \n is already written in the batch_statistics

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        for metric in self.model.metrics_names:
            self.cum_metrics[metric] += logs.get(metric)

        if self.seen % self.display == 0:
            batchnumber_float = (self.seen - self.display / 2.) / float(self.steps_per_total_epoch) + self.epoch_number - 1 \
                                + (self.steps_cum[self.f_number-1] / float(self.steps_per_total_epoch))
            line = '\n{0}\t{1}'.format(self.seen, batchnumber_float)
            for metric in self.model.metrics_names:
                line = line + '\t' + str(self.cum_metrics[metric] / self.display)
                self.cum_metrics[metric] = 0
            self.loglist.append(line)

            if self.flush != -1 and self.display % self.flush == 0:
                with open(self.logfile_train_fname, 'a') as logfile_train:
                    for batch_statistics in self.loglist:
                        logfile_train.write(batch_statistics)
                    self.loglist = []
                    logfile_train.flush()
                    os.fsync(logfile_train.fileno())

    def on_epoch_end(self, batch, logs={}):
        # on epoch end here means that this is called after one fit_generator loop in Keras is finished.
        with open(self.logfile_train_fname, 'a') as logfile_train:
            for batch_statistics in self.loglist:
                logfile_train.write(batch_statistics)
            logfile_train.flush()
            os.fsync(logfile_train.fileno())
