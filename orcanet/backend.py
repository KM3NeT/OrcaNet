#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for training and validating NN's, as well as evaluating them.
"""

import os
import h5py
from contextlib import ExitStack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from orcanet.utilities.visualization import (
    plot_activations, plot_weights)
from orcanet.logging import BatchLogger

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def train_model(orga, model, epoch, batch_logger=False):
    """
    Train a model on one file and return the history.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras model
        A compiled keras model.
    epoch : tuple
        Current epoch and the no of the file to train on.
    batch_logger : bool
        Use the orcanet batchlogger to log the training.

    Returns
    -------
    history : dict
        The history of the training on this file. A record of training
        loss values and metrics values.

    """
    files_dict = orga.io.get_file("train", epoch[1])

    if orga.cfg.n_events is not None:
        # TODO Can throw an error if n_events is larger than the file
        f_size = orga.cfg.n_events  # for testing purposes
    else:
        f_size = orga.io.get_file_sizes("train")[epoch[1] - 1]

    callbacks = []
    if batch_logger:
        callbacks.append(BatchLogger(orga, epoch))
    if orga.cfg.callback_train is not None:
        try:
            callbacks.extend(orga.cfg.callback_train)
        except TypeError:
            callbacks.append(orga.cfg.callback_train)

    training_generator = hdf5_batch_generator(
        orga, files_dict, f_size=f_size,
        zero_center=orga.cfg.zero_center_folder is not None,
        shuffle=orga.cfg.shuffle_train)

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=int(f_size / orga.cfg.batchsize),
        verbose=orga.cfg.verbose_train,
        max_queue_size=orga.cfg.max_queue_size,
        callbacks=callbacks,
        initial_epoch=epoch[0] - 1,
        epochs=epoch[0],
    )

    # get a dict with losses and metrics
    # only trained for one epoch, so value is list of len 1
    history = {key: value[0] for key, value in history.history.items()}
    return history


def validate_model(orga, model):
    """
    Validates a model on all validation files and return the history.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras model
        A compiled keras model.

    Returns
    -------
    history : dict
        The history of the validation on all files. A record of validation
        loss values and metrics values.

    """
    # One history for each val file
    histories = []
    f_sizes = orga.io.get_file_sizes("val")

    for i, files_dict in enumerate(orga.io.yield_files("val")):
        f_size = f_sizes[i]
        if orga.cfg.n_events is not None:
            f_size = orga.cfg.n_events  # for testing purposes

        val_generator = hdf5_batch_generator(
            orga, files_dict, f_size=f_size,
            zero_center=orga.cfg.zero_center_folder is not None)

        history_file = model.evaluate_generator(
            val_generator,
            steps=int(f_size / orga.cfg.batchsize),
            max_queue_size=orga.cfg.max_queue_size,
            verbose=orga.cfg.verbose_val)
        if not isinstance(history_file, list):
            history_file = [history_file, ]
        histories.append(history_file)

    # average over all val files
    history = weighted_average(histories, f_sizes)

    # This history is just a list, not a dict like with fit_generator
    # so transform to dict
    history = dict(zip(model.metrics_names, history))

    return history


def weighted_average(histories, f_sizes):
    """
    Average multiple histories, weighted with the file size.

    Each history can have multiple metrics, which are averaged seperatly.

    Parameters
    ----------
    histories : List
        List of histories, one for each file. Each history is also
        a list: each entry is a different loss or metric.
    f_sizes : List
        List of the file sizes, in the same order as the histories, i.e.
        the file of histories[0] has the length f_sizes[0].

    Returns
    -------
    wgtd_average : List
        The weighted averaged history. Has the same length as each
        history in the histories List, i.e. one entry per loss or metric.

    """
    assert len(histories) == len(f_sizes)
    rltv_fsizes = [f_size/sum(f_sizes) for f_size in f_sizes]
    wgtd_average = np.dot(np.transpose(histories), rltv_fsizes)

    return wgtd_average.tolist()


def hdf5_batch_generator(orga, files_dict, f_size=None, zero_center=False,
                         yield_mc_info=False, shuffle=False):
    """
    Yields batches of input data from h5 files.

    This will go through one file, or multiple files in parallel, and yield
    one batch of data, which can then be used as an input to a model.
    Since multiple filepaths can be given to read out in parallel,
    this can also be used for models with multiple inputs.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    files_dict : dict
        Pathes of the files to train on.
        Keys: The name of every input (from the toml list file, can be multiple).
        Values: The filepath of a single h5py file to read samples from.
    f_size : int or None
        Specifies the number of samples to be read from the .h5 file.
        If none, the whole .h5 file will be used.
    zero_center : bool
        Whether to use zero centering.
        Requires orga.zero_center_folder to be set.
    yield_mc_info : bool
        Specifies if mc-infos (y_values) should be yielded as well. The
        mc-infos are used for evaluation after training and testing is finished.
    shuffle : bool
        Randomize the order in which batches are read from the file.
        Significantly reduces read out speed.

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
    batchsize = orga.cfg.batchsize
    # name of the datagroups in the file
    samples_key = orga.cfg.key_samples
    mc_key = orga.cfg.key_mc_info

    # If the batchsize is larger than the f_size, make batchsize smaller
    # or nothing would be yielded
    if f_size is not None:
        if f_size < batchsize:
            batchsize = f_size

    if orga.cfg.label_modifier is not None:
        label_modifier = orga.cfg.label_modifier
    else:
        assert orga._auto_label_modifier is not None, \
            "Auto label modifier has not been set up (can be done with " \
            "nn_utilities.get_auto_label_modifier)"
        label_modifier = orga._auto_label_modifier

    # get xs_mean or load/create if not stored yet
    if zero_center:
        xs_mean = orga.get_xs_mean()
    else:
        xs_mean = None

    with ExitStack() as stack:
        # a dict with the names of list inputs as keys, and the opened
        # h5 files as values
        files = {}
        file_lengths = []
        # open the files and make sure they have the same length
        for input_key in files_dict:
            files[input_key] = stack.enter_context(
                h5py.File(files_dict[input_key], 'r'))
            file_lengths.append(len(files[input_key][samples_key]))

        if not file_lengths.count(file_lengths[0]) == len(file_lengths):
            raise ValueError("All data files must have the same length! "
                             "Yours have:\n " + str(file_lengths))

        if f_size is None:
            f_size = file_lengths[0]
        total_no_of_batches = int(np.ceil(f_size/batchsize))
        # positions of the samples in the file
        sample_pos = np.arange(total_no_of_batches) * batchsize
        if shuffle:
            np.random.shuffle(sample_pos)
        # append some samples due to preloading by the fit_generator method
        sample_pos = np.append(sample_pos, sample_pos[:orga.cfg.max_queue_size])

        for sample_n in sample_pos:
            # A dict with every input name as key, and a batch of data as values
            xs = {}
            # Read one batch of samples from the files and zero center
            for input_key in files:
                xs[input_key] = files[input_key][samples_key][
                                sample_n: sample_n + batchsize]
                if xs_mean is not None:
                    xs[input_key] = np.subtract(xs[input_key],
                                                xs_mean[input_key])
            # Get labels for the nn. Since the labels are hopefully the same
            # for all the files, use the ones from the first TODO add check
            y_values = list(files.values())[0][mc_key][
                       sample_n:sample_n + batchsize]

            # Modify the samples and labels before feeding them into the network
            if orga.cfg.sample_modifier is not None:
                xs = orga.cfg.sample_modifier(xs)

            ys = label_modifier(y_values)

            if not yield_mc_info:
                yield xs, ys
            else:
                yield xs, ys, y_values


def save_actv_wghts_plot(orga, model, epoch, samples=1):
    """
    Plots the weights of a model and the activations for samples from
    the validation set to one .pdf file each.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.models.Model
        The model to do the predictions with.
    epoch : tuple
        Current epoch and fileno.
    samples : int
        Number of samples to make the plot for.

    """
    plt.ioff()

    file = next(orga.io.yield_files("val"))
    generator = hdf5_batch_generator(
        orga, file, f_size=samples,
        zero_center=orga.cfg.zero_center_folder is not None,
        yield_mc_info=True)
    xs, ys, y_values = next(generator)

    pdf_name_act = "{}/activations_epoch_{}_file_{}.pdf".format(
        orga.io.get_subfolder("activations", create=True), epoch[0], epoch[1])

    with PdfPages(pdf_name_act) as pdf:
        for layer in model.layers:
            fig = plot_activations(model, xs, layer.name, mode='test')
            pdf.savefig(fig)
            plt.close(fig)

    pdf_name_wght = "{}/weights_epoch_{}_file_{}.pdf".format(
        orga.io.get_subfolder("activations", create=True), epoch[0], epoch[1])

    with PdfPages(pdf_name_wght) as pdf:
        for layer in model.layers:
            fig = plot_weights(model, layer.name)
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)


def make_model_prediction(orga, model, epoch, fileno, samples=None):
    """
    Let a model predict on all validation samples, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup mc_info straight from
    the given files, as well as two datagroups per output layer of the network,
    which have the labels and the predicted values in them as numpy arrays,
    respectively.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Trained Keras model of a neural network.
    epoch : int
        Epoch of the last model training step in the epoch, file_no tuple.
    fileno : int
        File number of the last model training step in the epoch, file_no tuple.
    samples : int or None
        Number of events that should be predicted.
        If samples=None, the whole file will be used.

    """
    batchsize = orga.cfg.batchsize
    compression = ("gzip", 1)
    file_sizes = orga.io.get_file_sizes("val")

    latest_pred_file_no = orga.io.get_latest_prediction_file_no(epoch, fileno)
    if latest_pred_file_no is None:
        latest_pred_file_no = -1

    i = 0
    # For every val file set (one set can have multiple files if
    # the model has multiple inputs):
    for f_number, files_dict in enumerate(orga.io.yield_files("val")):
        if f_number <= latest_pred_file_no:
            continue

        pred_filepath = orga.io.get_next_pred_path(epoch, fileno, latest_pred_file_no + i)
        try:
            with h5py.File(pred_filepath, 'w') as h5_file:

                file_size = file_sizes[f_number]
                generator = hdf5_batch_generator(
                    orga, files_dict,
                    zero_center=orga.cfg.zero_center_folder is not None,
                    yield_mc_info=True)

                if samples is None:
                    steps = int(file_size / batchsize)
                    if file_size % batchsize != 0:
                        # add a smaller step in the end
                        steps += 1
                else:
                    steps = int(samples / batchsize)

                for s in range(steps):
                    if s % 1000 == 0:
                        print('Predicting in step {}/{} on '
                              'file {}'.format(s, steps, f_number + 1))
                    # y_true is a dict of ndarrays, mc_info is a structured
                    # array, y_pred is a list of ndarrays
                    xs, y_true, mc_info = next(generator)

                    y_pred = model.predict_on_batch(xs)
                    if not isinstance(y_pred, list):
                        # if only one output, transform to a list
                        y_pred = [y_pred]
                    # transform y_pred to dict
                    y_pred = {out: y_pred[i] for i, out in enumerate(model.output_names)}

                    if orga.cfg.dataset_modifier is None:
                        datasets = get_datasets(mc_info, y_true, y_pred)
                    else:
                        datasets = orga.cfg.dataset_modifier(mc_info, y_true, y_pred)

                    # TODO maybe add attr to data, like used files or orcanet version number?
                    if s == 0:  # create datasets in the first step
                        for dataset_name, data in datasets.items():
                            maxshape = (file_size,) + data.shape[1:]
                            chunks = True  # (batchsize,) + data.shape[1:]
                            h5_file.create_dataset(
                                dataset_name, data=data, maxshape=maxshape,
                                chunks=chunks, compression=compression[0],
                                compression_opts=compression[1])

                    else:
                        for dataset_name, data in datasets.items():
                            # append data at the end of the dataset
                            h5_file[dataset_name].resize(
                                h5_file[dataset_name].shape[0] + data.shape[0], axis=0)
                            h5_file[dataset_name][-data.shape[0]:] = data

            i += 1

        except BaseException as exception:
            os.remove(pred_filepath)
            raise exception


def get_datasets(mc_info, y_true, y_pred):
    """
    Get the dataset names and numpy array contents.

    Every output layer will get one dataset each for both the label and
    the prediction. E.g. if your model has an output layer called "energy",
    the datasets "label_energy" and "pred_energy" will be made.

    Parameters
    ----------
    mc_info : ndarray
        A structured array containing infos for every event, right from
        the input files.
    y_true : dict
        The labels for each output layer of the network.
    y_pred : dict
        The predictions of each output layer of the network.

    Returns
    -------
    datasets : dict
        Keys are the name of the datagroups, values the content in the
        form of numpy arrays.

    """
    datasets = dict()
    datasets["mc_info"] = mc_info
    for out_layer_name in y_true:
        datasets["label_" + out_layer_name] = y_true[out_layer_name]
    for out_layer_name in y_pred:
        datasets["pred_" + out_layer_name] = y_pred[out_layer_name]
    return datasets
