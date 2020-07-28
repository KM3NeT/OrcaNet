#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for training and validating NN's, as well as evaluating them.
"""
import time
import h5py
import numpy as np
import os

import orcanet
from orcanet.logging import BatchLogger
import orcanet.utilities.nn_utilities as nn_utilities
from orcanet.in_out import h5_get_number_of_rows
from orcanet.h5_generator import get_h5_generator


def train_model(orga, model, epoch, batch_logger=False):
    """
    Train a model on one file and return the history.

    Parameters
    ----------
    orga : orcanet.core.Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras.Model
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

    callbacks = [
        nn_utilities.RaiseOnNaN(),
        nn_utilities.TimeModel(print_func=orga.io.print_log),
    ]
    if batch_logger:
        callbacks.append(BatchLogger(orga, epoch))
    if orga.cfg.callback_train is not None:
        try:
            callbacks.extend(orga.cfg.callback_train)
        except TypeError:
            callbacks.append(orga.cfg.callback_train)

    training_generator = get_h5_generator(
        orga, files_dict, f_size=f_size, phase="training",
        zero_center=orga.cfg.zero_center_folder is not None,
        shuffle=orga.cfg.shuffle_train)

    history = model.fit(
        training_generator,
        steps_per_epoch=int(f_size / orga.cfg.batchsize),
        verbose=orga.cfg.verbose_train,
        max_queue_size=orga.cfg.max_queue_size,
        callbacks=callbacks,
        initial_epoch=epoch[0] - 1,
        epochs=epoch[0],
    )
    training_generator.print_timestats(print_func=orga.io.print_log)
    # get a dict with losses and metrics
    # only trained for one epoch, so value is list of len 1
    history = {key: value[0] for key, value in history.history.items()}
    return history


def validate_model(orga, model):
    """
    Validates a model on all validation files and return the history.

    Parameters
    ----------
    orga : orcanet.core.Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras.Model
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
            orga, files_dict, f_size=f_size, phase="validation",
            zero_center=orga.cfg.zero_center_folder is not None)

        history_file = model.evaluate(
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


def h5_inference(orga, model, files_dict, output_path, samples=None, use_def_label=True):
    """
    Let a model predict on all samples in a h5 file, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup y_values straight from
    the given files, as well as two datagroups per output layer of the network,
    which have the labels and the predicted values in them as numpy arrays,
    respectively.

    Parameters
    ----------
    orga : orcanet.core.Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras.Model
        Trained Keras model of a neural network.
    files_dict : dict
        Dict mapping model input names to h5 file paths.
    output_path : str
        Name of the output h5 file containing the predictions.
    samples : int, optional
        Dont use all events in the file, but instead only the given number.
    use_def_label : bool
        If True and no label modifier is given by user, use the default
        label modifier instead of none.

    """
    file_size = h5_get_number_of_rows(
        list(files_dict.values())[0],
        datasets=[orga.cfg.key_x_values])
    generator = get_h5_generator(
        orga,
        files_dict,
        zero_center=orga.cfg.zero_center_folder is not None,
        keras_mode=False,
        use_def_label=use_def_label,
        phase="inference",
    )
    itergen = iter(generator)

    if samples is None:
        steps = len(generator)
    else:
        steps = int(samples / orga.cfg.batchsize)
    print_every = max(100, min(int(round(steps/10, -2)), 1000))
    model_time_total = 0.

    temp_output_path = os.path.join(
        os.path.dirname(output_path),
        "temp_" + os.path.basename(output_path) + "_" +
        time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime()))
    print(f"Creating temporary file {temp_output_path}")
    with h5py.File(temp_output_path, 'x') as h5_file:
        h5_file.attrs.create("orcanet", orcanet.__version__, dtype="S6")

        for s in range(steps):
            if s % print_every == 0:
                print('Predicting in step {}/{} ({:0.2%})'.format(
                    s, steps, s/steps))

            info_blob = next(itergen)

            start_time = time.time()
            y_pred = model.predict_on_batch(info_blob["xs"])
            model_time_total += time.time() - start_time

            if not isinstance(y_pred, list):
                # if only one output, transform to a list
                y_pred = [y_pred]
            # transform y_pred to dict
            y_pred = {
                out: y_pred[i] for i, out in enumerate(model.output_names)}
            info_blob["y_pred"] = y_pred

            if info_blob.get("org_batchsize") is not None:
                _slice_to_size(info_blob)

            if orga.cfg.dataset_modifier is None:
                datasets = get_datasets(info_blob)
            else:
                datasets = orga.cfg.dataset_modifier(info_blob)

            if s == 0:  # create datasets in the first step
                for dataset_name, data in datasets.items():
                    h5_file.create_dataset(
                        dataset_name,
                        data=data,
                        maxshape=(file_size,) + data.shape[1:],
                        chunks=True,  # (batchsize,) + data.shape[1:]
                        compression="gzip",
                        compression_opts=1,
                    )

            else:
                for dataset_name, data in datasets.items():
                    # append data at the end of the dataset
                    h5_file[dataset_name].resize(
                        h5_file[dataset_name].shape[0] + data.shape[0], axis=0)
                    h5_file[dataset_name][-data.shape[0]:] = data

    if os.path.exists(output_path):
        raise FileExistsError(
            f"{output_path} exists already! But file {temp_output_path} "
            f"is finished and can be safely used.")
    os.rename(temp_output_path, output_path)
    generator.print_timestats()
    print("Statistics of model prediction:")
    print(f"\tTotal time:\t{model_time_total / 60:.2f} min")
    print(f"\tPer batch:\t{1000 * model_time_total / steps:.5} ms")


def _slice_to_size(info_blob):
    org_batchsize = info_blob["org_batchsize"]
    for input_key, x in info_blob["xs"].items():
        info_blob["xs"][input_key] = x[:org_batchsize]
    for output_key, y_pred in info_blob["y_pred"].items():
        info_blob["y_pred"][output_key] = y_pred[:org_batchsize]
    if info_blob.get("ys") is not None:
        for output_key, y in info_blob["ys"].items():
            info_blob["ys"][output_key] = y[:org_batchsize]


def make_model_prediction(orga, model, epoch, fileno, samples=None):
    """
    Let a model predict on all validation samples, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup y_values straight from
    the given files, as well as two datagroups per output layer of the network,
    which have the labels and the predicted values in them as numpy arrays,
    respectively.

    Parameters
    ----------
    orga : orcanet.core.Organizer
        Contains all the configurable options in the OrcaNet scripts.
    model : keras.Model
        A compiled keras model.
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
    If there are no labels (e.g. during orga.inference), the label dataset
    will not be generated.

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

    if info_blob.get("ys") is not None:
        y_true = info_blob["ys"]
        for out_layer_name in y_true:
            datasets["label_" + out_layer_name] = y_true[out_layer_name]

    y_pred = info_blob["y_pred"]
    for out_layer_name in y_pred:
        datasets["pred_" + out_layer_name] = y_pred[out_layer_name]

    return datasets
