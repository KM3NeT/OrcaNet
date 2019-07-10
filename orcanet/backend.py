#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for training and validating NN's, as well as evaluating them.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from orcanet.utilities.layer_plotting import plot_activations, plot_weights
from orcanet.logging import BatchLogger
from orcanet.in_out import h5_get_number_of_rows
from orcanet.h5_generator import get_h5_generator

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

    training_generator = get_h5_generator(
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

        val_generator = get_h5_generator(
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
    generator = get_h5_generator(
        orga, file, f_size=samples,
        zero_center=orga.cfg.zero_center_folder is not None,
        keras_mode=True)
    xs, ys = next(generator)

    pdf_name_act = "{}/activations_epoch_{}_file_{}.pdf".format(
        orga.io.get_subfolder("activations", create=True), epoch[0], epoch[1])

    with PdfPages(pdf_name_act) as pdf:
        for layer in model.layers:
            plot_activations(model, xs, layer.name, mode='test')
            pdf.savefig()
            plt.clf()
        plt.close()

    pdf_name_wght = "{}/weights_epoch_{}_file_{}.pdf".format(
        orga.io.get_subfolder("activations", create=True), epoch[0], epoch[1])

    with PdfPages(pdf_name_wght) as pdf:
        for layer in model.layers:
            try:
                fig = plot_weights(model, layer.name)
            except ValueError:
                continue
            pdf.savefig(fig)
            plt.clf()
        plt.close()


def h5_inference(orga, model, files_dict, output_path, samples=None):
    """
    Let a model predict on all samples in a h5 file, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup y_values straight from
    the given files, as well as two datagroups per output layer of the network,
    which have the labels and the predicted values in them as numpy arrays,
    respectively.

    Parameters
    ----------
    orga : object Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Trained Keras model of a neural network.
    files_dict : dict
        Dict mapping model input names to h5 file paths.
    output_path : str
        Name of the output h5 file containing the predictions.
    samples : int or None
        Number of events that should be predicted.
        If samples=None, the whole file will be used.

    """
    batchsize = orga.cfg.batchsize
    compression = ("gzip", 1)

    file_size = h5_get_number_of_rows(
        list(files_dict.values())[0],
        datasets=[orga.cfg.key_x_values])
    generator = get_h5_generator(
        orga, files_dict,
        zero_center=orga.cfg.zero_center_folder is not None,
        keras_mode=False)

    if samples is None:
        steps = int(file_size / batchsize)
        if file_size % batchsize != 0:
            # add a smaller step in the end
            steps += 1
    else:
        steps = int(samples / batchsize)

    with h5py.File(output_path, 'w') as h5_file:
        for s in range(steps):
            if s % 1000 == 0:
                print('Predicting in step {}/{} ({:0.2%})'.format(
                    s, steps, s/steps))

            info_blob = next(generator)

            y_pred = model.predict_on_batch(info_blob["xs"])
            if not isinstance(y_pred, list):
                # if only one output, transform to a list
                y_pred = [y_pred]
            # transform y_pred to dict
            y_pred = {out: y_pred[i] for i, out in
                      enumerate(model.output_names)}

            info_blob["y_pred"] = y_pred

            if orga.cfg.dataset_modifier is None:
                datasets = get_datasets(info_blob)
            else:
                datasets = orga.cfg.dataset_modifier(info_blob)

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


def make_model_prediction(orga, model, epoch, fileno, samples=None):
    """
    Let a model predict on all validation samples, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup y_values straight from
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
    latest_pred_file_no = orga.io.get_latest_prediction_file_no(epoch, fileno)
    if latest_pred_file_no is None:
        latest_pred_file_no = 0

    # For every val file set (one set can have multiple files if
    # the model has multiple inputs):
    for f_number, files_dict in enumerate(orga.io.yield_files("val"), 1):
        if f_number <= latest_pred_file_no:
            continue

        pred_filepath = orga.io.get_pred_path(epoch, fileno, f_number)
        h5_inference(orga, model, files_dict, pred_filepath, samples=samples)


def get_datasets(info_blob):
    """
    Get the dataset names and numpy array contents.

    Every output layer will get one dataset each for both the label and
    the prediction. E.g. if your model has an output layer called "energy",
    the datasets "label_energy" and "pred_energy" will be made.

    Parameters
    ----------
    info_blob : dict
        Contains the following infos as keys/values:
        xs : ndarray
            Input to the network, after sample modifier has been applied.
        y_values : ndarray or None
            A structured array containing infos for every event, right from
            the input files.
            Can also be None, e.g. for when there are no y_values.
        ys : dict or None
            The labels for each output layer of the network.
            Can also be None, e.g. for when there are no y_values.
        y_pred : dict
            The predictions of each output layer of the network.

    Returns
    -------
    datasets : dict
        Keys are the name of the datagroups, values the content in the
        form of numpy arrays.

    """
    datasets = dict()
    if "y_values" in info_blob:
        datasets["y_values"] = info_blob["y_values"]

    if "ys" in info_blob:
        y_true = info_blob["ys"]
        for out_layer_name in y_true:
            datasets["label_" + out_layer_name] = y_true[out_layer_name]

    y_pred = info_blob["y_pred"]
    for out_layer_name in y_pred:
        datasets["pred_" + out_layer_name] = y_pred[out_layer_name]

    return datasets
