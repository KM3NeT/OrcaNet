#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a NN."""

import numpy as np
import h5py
import os
import keras as ks
from functools import reduce
from contextlib import ExitStack


def generate_batches_from_hdf5_file(orca, files_dict, f_size=None, zero_center_image=None, yield_mc_info=False, shuffle=False):
    """
    Yields batches of input data from h5 files.

    This will go through one file, or multiple files in parallel, and yield one batch of data, which can then
    be used as an input to a model. Since multiple filepaths can be given to read out in parallel,
    this can also be used for models with multiple inputs.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    files_dict : dict or None
        Pathes of the files to train on.
        Keys: The name of every input (given in the toml list file, can be multiple).
        Values: The filepath of a single h5py file to read samples from.
    f_size : int or None
        Specifies the number of samples to be read from the .h5 file.
        If none, the whole .h5 file will be used.
    zero_center_image : dict or None
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
            Keys : str  The name(s) of the input layer(s) of the model.
            Values : ndarray    A batch of samples for the corresponding input.
    ys : dict
        Labels for the model to train on.
            Keys : str  The name(s) of the output layer(s) of the model.
            Values : ndarray    A batch of labels for the corresponding output.
    mc_info : ndarray, optional
        Mc info from the file. Only yielded if yield_mc_info is True.

    """
    batchsize = orca.cfg.batchsize
    # name of the datagroups in the file
    samples_key = orca.cfg.key_samples
    mc_key = orca.cfg.key_labels
    # If the batchsize is larger than the f_size, make batchsize smaller or nothing would be yielded
    if f_size is not None:
        if f_size < batchsize:
            batchsize = f_size

    if orca.cfg.label_modifier is not None:
        label_modifier = orca.cfg.label_modifier
    else:
        assert orca.cfg._auto_label_modifier is not None, "Auto label modifier has not been set up (can be done with nn_utilities.get_auto_label_modifier)"
        label_modifier = orca.cfg._auto_label_modifier

    with ExitStack() as stack:
        # a dict with the names of list inputs as keys, and the opened h5 files as values.
        files = {}
        file_lengths = []
        # open the files and make sure they have the same length
        for input_key in files_dict:
            files[input_key] = stack.enter_context(h5py.File(files_dict[input_key], 'r'))
            file_lengths.append(len(files[input_key][samples_key]))

        if not file_lengths.count(file_lengths[0]) == len(file_lengths):
            raise AssertionError("All data files must have the same length! Yours have:\n " + str(file_lengths))

        if f_size is None:
            f_size = file_lengths[0]
        # number of batches available
        total_no_of_batches = int(np.ceil(f_size/batchsize))
        # positions of the samples in the file
        sample_pos = np.arange(total_no_of_batches) * batchsize
        if shuffle:
            np.random.shuffle(sample_pos)
        # append some samples due to preloading by the fit_generator method
        sample_pos = np.append(sample_pos, sample_pos[:orca.cfg.max_queue_size])

        for sample_n in sample_pos:
            # A dict with every input name as key, and a batch of data as values
            xs = {}
            # Read one batch of samples from the files and zero center
            for input_key in files:
                xs[input_key] = files[input_key][samples_key][sample_n: sample_n + batchsize]
                if zero_center_image is not None:
                    xs[input_key] = np.subtract(xs[input_key], zero_center_image[input_key])
            # Get labels for the nn. Since the labels are hopefully the same for all the files, use the ones from the first
            y_values = list(files.values())[0][mc_key][sample_n:sample_n + batchsize]

            # Modify the samples and labels before feeding them into the network
            if orca.cfg.sample_modifier is not None:
                xs = orca.cfg.sample_modifier(xs)

            ys = label_modifier(y_values)

            if not yield_mc_info:
                yield xs, ys
            else:
                yield xs, ys, y_values


def get_auto_label_modifier(model):
    """
    Get a label_modifier that reads out the labels from the files which are required by the model.

    If the model has more than one output layer, it has to be compiled with a dict of losses, not a list.

    Parameters
    ----------
    model : ks.Model
        A keras model.

    Returns
    -------
    label_modifier : function

    """
    assert isinstance(model.loss, dict), "You did not compile your model with a dict of losses. " \
                                         "Use a dict, so that it is clear which label from your data files " \
                                         "belongs to which output layer."
    names = tuple(model.loss.keys())

    def label_modifier(y_values):
        ys = {name: y_values[name] for name in names}
        return ys
    return label_modifier


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

# ------------- Zero center functions -------------#


def load_zero_center_data(orca):
    """
    Gets the xs_mean array(s) that can be used for zero-centering.

    The arrays are either loaded from a previously saved .npz file or they are calculated on the fly by
    calculating the mean value per bin for the given training files. The name of the saved image is derived from the
    name of the list file which was given to the cfg.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.

    Returns
    -------
    xs_mean : dict
        Dict of ndarray(s) that contains the mean_image of the x dataset (1 array per list input).
        Can be used for zero-centering later on.
        Example format:
        { "input_A" : ndarray, "input_B" : ndarray }

    """
    all_train_files = orca.cfg.get_files("train")
    zero_center_folder = orca.cfg.zero_center_folder
    train_files_list_name = os.path.basename(orca.cfg.get_list_file())

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
            xs_mean_for_ip_i = get_mean_image(all_train_files_for_ip_i, orca.cfg.key_samples)

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


def get_mean_image(filepaths, key_samples):
    """
    Returns the mean_image of a xs dataset.
    Calculating still works if xs is larger than the available memory and also if the file is compressed!
    :param list filepaths: Filepaths of the data upon which the mean_image should be calculated.
    :param str key_samples: The name of the datagroup in your h5 input files which contains the samples to the network.
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    # check available memory and divide the mean calculation in steps
    total_memory = 4e9  # * n_gpu # In bytes. Take max. 1/2 of what is available per GPU (16G), just to make sure.

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
    Write logfiles during training.

    Averages the losses of the model over some number of batches, and then writes that in a line in the logfile.

    """
    def __init__(self, orca, model, epoch):
        """

        Parameters
        ----------
        orca : object OrcaHandler
            Contains all the configurable options in the OrcaNet scripts.
        model : ks.Modle
            The keras model.
        epoch : tuple
            Epoch and file number.

        """
        ks.callbacks.Callback.__init__(self)
        self.display = orca.cfg.train_logger_display
        self.epoch_number = epoch[0]
        self.f_number = epoch[1]
        self.model = model
        self.flush = orca.cfg.train_logger_flush

        self.seen = 0
        self.logfile_train_fname = orca.io.get_subfolder("log_train", create=True) + '/log_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.txt'
        self.loglist = []

        self.cum_metrics = {}
        for metric in self.model.metrics_names:  # set up dict with all model metrics
            self.cum_metrics[metric] = 0

        self.steps_per_total_epoch, self.steps_cum = 0, [0]
        for f_size in orca.io.get_file_sizes("train"):
            steps_per_file = int(f_size / orca.cfg.batchsize)
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
