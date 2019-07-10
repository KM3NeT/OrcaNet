#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a NN."""

import numpy as np
import h5py
import os
import keras as ks
from functools import reduce
import keras.backend as K


def get_auto_label_modifier(model):
    """
    Get a label_modifier for when none is specified by the user.

    Will simply assume that for every output of the model,
    there is a column in the y_values with the same name.

    Parameters
    ----------
    model : ks.Model
        A keras model.

    Returns
    -------
    label_modifier : function

    """
    names = model.output_names

    def label_modifier(info_blob):
        y_values = info_blob["y_values"]
        ys = {name: y_values[name] for name in names}
        return ys
    return label_modifier


def get_layer_output(model, samples, layer_name, mode="test"):
    """
    Get the output of an intermediate layer from a model.

    Parameters
    ----------
    model : keras model
        The model.
    samples : List or ndarray
        The data to apply the model to.
    layer_name : str
        Name of the layer.
    mode : str
        Mode of the layers during the forward pass. Either train or test.
        Important for batchnorm, dropout, ...

    Returns
    -------
    layer_output : ndarray
        The output from the layer.

    """
    if mode == "test":
        phase = 0
    elif mode == "train":
        phase = 1
    else:
        raise NameError("Unknown mode: ", mode)

    # either a tensor or a list of tensors
    inp_tensors = model.input
    # Make input to always be a list
    if not isinstance(inp_tensors, list):
        inp_tensors = [inp_tensors, ]

    if isinstance(samples, dict):
        # doesnt work with dicts so transform into lists instead
        xs = [samples[key] for key in model.input_names]
    elif not isinstance(samples, list):
        # Make input to always be a list
        xs = [samples, ]
    else:
        xs = samples

    get_output = K.function(
        inp_tensors + [K.learning_phase(), ],
        [model.get_layer(layer_name).output])
    layer_output = get_output(xs + [phase, ])[0]
    return layer_output


# ------------- Zero center functions -------------#


def load_zero_center_data(orga, logging=False):
    """
    Gets the xs_mean array(s) that can be used for zero-centering.

    The arrays are either loaded from a previously saved .npz file or they
    are calculated on the fly by calculating the mean value per bin for the
    given training files. The name of the saved image is derived from the
    name of the list file which was given to the cfg.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    logging : bool
        If true, will log the execution of this function into the
        full summary in the output folder.

    Returns
    -------
    xs_mean : dict
        Dict of ndarray(s) that contains the mean_image of the x dataset
        (1 array per list input). Can be used for zero-centering later on.
        Example format:
        { "input_A" : ndarray, "input_B" : ndarray }

    """
    all_train_files = orga.cfg.get_files("train")
    zero_center_folder = orga.cfg.zero_center_folder
    if not zero_center_folder.endswith("/"):
        zero_center_folder += "/"
    train_files_list_name = os.path.basename(orga.cfg.get_list_file())
    key_samples = orga.cfg.key_samples

    orga.io.print_log("Zero centering", logging)
    orga.io.print_log("--------------", logging)
    orga.io.print_log("Zero center folder:   " + zero_center_folder, logging)

    xs_mean = {}
    for input_key, train_filepaths in all_train_files.items():
        xs_mean_path = get_xs_mean_path(zero_center_folder, train_filepaths)

        if xs_mean_path is not None:
            orga.io.print_log('{}:   Loading saved zero centering'.format(
                input_key), logging)
            xs_mean_ip_i = np.load(xs_mean_path)["xs_mean"]
            orga.io.print_log("\tLoaded file: {}".format(
                os.path.basename(xs_mean_path)), logging)

        else:
            orga.io.print_log('{}:   Making new zero centering'.format(
                input_key), logging)

            xs_mean_ip_i = make_xs_mean(train_filepaths, key_samples)
            filename = zero_center_folder + train_files_list_name \
                                          + '_input_' + str(input_key) + '.npz'
            np.savez(filename, xs_mean=xs_mean_ip_i,
                     zero_center_used_ip_files=train_filepaths)

            orga.io.print_log('\tSaved as {} with shape {}'.format(
                os.path.basename(filename), xs_mean_ip_i.shape), logging)

        xs_mean[input_key] = xs_mean_ip_i

    orga.io.print_log("", logging)
    return xs_mean


def get_xs_mean_path(zero_center_folder, train_filepaths):
    """
    Search for precalculated xs_mean arrays in the zero_center_folder.

    The function opens every .npz file in the zero center folder and checks
    if the files used to generate this xs_mean (stored as subarray
    'zero_center_used_ip_files') is the same as the given train_filepaths.

    Parameters
    ----------
    zero_center_folder : str
        Full path to the folder where the zero_centering arrays are stored.
    train_filepaths : list
        The filepaths of all train_files.

    Returns
    -------
    xs_mean_path : None/ndarray
        The zero center filepath for the given train_filepaths if
        it exists in the zero_center_files. If not, returns None.

    """
    xs_mean_path = None

    if not os.path.isdir(zero_center_folder):
        os.mkdir(zero_center_folder)

    for file in os.listdir(zero_center_folder):
        if not file.endswith('.npz'):
            continue
        file = zero_center_folder + file
        used_ip_files = np.load(file)['zero_center_used_ip_files']
        if np.array_equal(used_ip_files, train_filepaths):
            xs_mean_path = file
            break

    return xs_mean_path


def make_xs_mean(filepaths, key_samples, total_memory=4e9):
    """
    Calculates the zero center image of a dataset.

    Calculating still works if xs is larger than the available memory
    and also if the file is compressed.

    Parameters
    ----------
    filepaths : List
        Filepaths of the data files with the samples for which the
        mean_image will be calculated.
    key_samples : str
        The name of the datagroup in your h5 input files which contains
        the samples to the network.
    total_memory : int
        check available memory and divide the mean calculation in steps
        total_memory = 4e9  # * n_gpu # In bytes.
        Take max. 1/2 of what is available per GPU (16G), just to make sure.

    Returns
    -------
    xs_mean : ndarray
        The zero center image.

    """
    xs_means = []
    file_sizes = []

    for filepath in filepaths:

        with h5py.File(filepath, "r") as file:
            filesize = get_array_memsize(file['x'])
            steps = int(np.ceil(filesize/total_memory))
            n_rows = file[key_samples].shape[0]
            stepsize = int(n_rows / float(steps))

            # create xs_mean_arr that stores intermediate mean_temp results
            xs_mean_arr = np.zeros((steps, ) + file['x'].shape[1:],
                                   dtype=np.float64)
            print("\tCalculating for file: " + filepath)
            for i in range(steps):
                if i % 5 == 0:
                    print('\t   Step ' + str(i) + " of " + str(steps))

                # for the last step, calculate mean till the end of the file
                if i == steps-1 or steps == 1:
                    xs_mean_temp = np.mean(
                        file[key_samples][i * stepsize: n_rows],
                        axis=0, dtype=np.float64)
                else:
                    xs_mean_temp = np.mean(
                        file[key_samples][i*stepsize: (i+1) * stepsize],
                        axis=0, dtype=np.float64)

                xs_mean_arr[i] = xs_mean_temp

        print("\tDone!")
        # The mean for this file
        xs_means.append(np.mean(xs_mean_arr, axis=0,
                                dtype=np.float64).astype(np.float32))
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
