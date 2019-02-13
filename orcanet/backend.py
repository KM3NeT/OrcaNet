#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for training and validating NN's, as well as evaluating them.
"""

import os
from inspect import signature
import keras.backend as K
import h5py

from orcanet.in_out import write_summary_logfile, write_full_logfile, read_logfiles
from orcanet.utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger, generate_batches_from_hdf5_file
from orcanet.utilities.visualization import plot_all_metrics_to_pdf
from orcanet_contrib.contrib import orca_learning_rates

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def get_learning_rate(orca, epoch):
    """
    Get the learning rate for the current epoch and file number.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    epoch : tuple
        Epoch and file number.

    Returns
    -------
    lr : float
        The learning rate.

    Raises
    ------
    AssertionError
        If the type of the user_lr is not right.

    """
    user_lr = orca.cfg.learning_rate
    error_msg = "The learning rate must be either a float, a tuple of two floats or a function."
    if isinstance(user_lr, float):
        # Constant LR
        lr = user_lr

    elif isinstance(user_lr, tuple) or isinstance(user_lr, list):
        if len(user_lr) == 2:
            # Exponentially decaying LR
            lr = user_lr[0] * (1 - user_lr[1])**(epoch[1] + epoch[0]*len(orca.io.get_no_of_files("train")))
        else:
            raise AssertionError(error_msg, "(Your tuple has length "+str(len(user_lr))+")")

    elif isinstance(user_lr, str):
        if user_lr == "triple_decay":
            lr_schedule = orca_learning_rates("triple_decay")
            lr = lr_schedule(epoch[0], epoch[1], orca)
        else:
            raise NameError(user_lr, "is an unknown learning rate string!")

    elif callable(user_lr):
        # User defined function
        assert len(signature(user_lr).parameters) == 3, "A custom learning rate function must have three input parameters: \
        The epoch, the file number and the Configuration object."
        lr = user_lr(epoch[0], epoch[1], orca)

    else:
        raise AssertionError(error_msg, "(You gave a " + str(type(user_lr)) + ")")
    return lr


def update_summary_plot(orca):
    """
    Refresh the summary plot of a model directory, found in ./plots/summary_plot.pdf.

    Validation and Train-data will be read out automatically, and the loss as well as every metric will be plotted in
    a seperate page in the pdf.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.

    """
    pdf_name = orca.io.get_subfolder("plots", create=True) + "/summary_plot.pdf"
    summary_data, full_train_data = read_logfiles(orca)
    plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name)


def train_and_validate_model(orca, model, start_epoch):
    """
    Train a model for one epoch, i.e. on all (remaining) train files once.

    Trains (fit_generator) and validates (evaluate_generator) a Keras model on the provided
    training and validation files for one epoch. The model is saved with an automatically generated filename based on the epoch,
    log files are written and summary plots are made.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.Models.model
        Compiled keras model to use for training and validating.
    start_epoch : tuple
        Upcoming epoch and file number.

    """
    if orca.cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(orca)
    else:
        xs_mean = None
    f_sizes = orca.io.get_file_sizes("train")

    for file_no, files_dict in enumerate(orca.io.yield_files("train")):
        # no of samples in current file
        f_size = f_sizes[file_no]
        # Only the file number changes during training, as this function trains only for one epoch
        curr_epoch = (start_epoch[0], file_no + 1)
        # skip to the file with the target file number given in the start_epoch tuple.
        if curr_epoch[1] < start_epoch[1]:
            continue

        lr = get_learning_rate(orca, curr_epoch)
        K.set_value(model.optimizer.lr, lr)
        print("Set learning rate to " + str(lr))

        # Train the model on one file and save it afterwards
        model_filename = orca.io.get_model_path(curr_epoch[0], curr_epoch[1])
        assert not os.path.isfile(model_filename), "You tried to train your model in epoch {} file {}, but this model \
            has already been trained and saved!".format(curr_epoch[0], curr_epoch[1])
        history_train = train_model(orca, model, files_dict, f_size, xs_mean, curr_epoch)
        model.save(model_filename)
        print("Saved model as " + model_filename)

        # Validate after every n-th file, starting with the first
        if (curr_epoch[1] - 1) % orca.cfg.validate_after_n_train_files == 0:
            history_val = validate_model(orca, model, xs_mean)
        else:
            history_val = None

        # Write logfiles and make plots
        write_summary_logfile(orca, curr_epoch, model, history_train, history_val, K.get_value(model.optimizer.lr))
        write_full_logfile(orca, model, history_train, history_val, K.get_value(model.optimizer.lr), curr_epoch, files_dict)
        update_summary_plot(orca)
        # TODO reimplement, this function throws errors all the time!
        # plot_weights_and_activations(cfg, model, xs_mean, curr_epoch)


def train_model(orca, model, files_dict, f_size, xs_mean, curr_epoch):
    """
    Trains a model on one file based on the Keras fit_generator method.

    The progress of the training is also logged.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    files_dict : dict
        The name of every input as a key, the path to the n-th training file as values.
    f_size : int
        Number of images contained in f
    xs_mean : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.
    curr_epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.

    Returns
    -------
    history : keras history object
        The history of the training on this file.

    """
    print('Training in epoch ' + str(curr_epoch[0]) + ' on file ' + str(curr_epoch[1]) + ' ,', str(files_dict))
    if orca.cfg.n_events is not None:
        # TODO Can throw an error if n_events is larger than the file
        f_size = orca.cfg.n_events  # for testing purposes
    callbacks = [BatchLevelPerformanceLogger(orca, model, curr_epoch), ]
    training_generator = generate_batches_from_hdf5_file(orca, files_dict, f_size=f_size, zero_center_image=xs_mean, shuffle=orca.cfg.shuffle_train)
    history = model.fit_generator(training_generator, steps_per_epoch=int(f_size / orca.cfg.batchsize), epochs=1,
                                  verbose=orca.cfg.verbose_train, max_queue_size=orca.cfg.max_queue_size, callbacks=callbacks)
    return history


def validate_model(orca, model, xs_mean):
    """
    Validates a model on all the validation datafiles based on the Keras evaluate_generator method.

    This is usually done after a session of training has been finished.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    xs_mean : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.

    Returns
    -------
    history_val : List
        The history of the validation.

    """
    # One history for each val file
    histories = []
    f_sizes = orca.io.get_file_sizes("val")
    for i, files_dict in enumerate(orca.io.yield_files("val")):
        print('Validating on file ', i+1, ',', str(files_dict))
        f_size = f_sizes[i]
        if orca.cfg.n_events is not None:
            f_size = orca.cfg.n_events  # for testing purposes
        val_generator = generate_batches_from_hdf5_file(orca, files_dict, f_size=f_size, zero_center_image=xs_mean)
        history = model.evaluate_generator(val_generator, steps=int(f_size / orca.cfg.batchsize), max_queue_size=orca.cfg.max_queue_size, verbose=orca.cfg.verbose_val)
        # This history object is just a list, not a dict like with fit_generator!
        print('Validation sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)
    history_val = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0]  # average over all val files if necessary

    return history_val


def make_model_prediction(orca, model, xs_mean, eval_filename, samples=None):
    """
    Let a model predict on all samples of the validation set in the toml list, and save it as a h5 file.

    Per default, the h5 file will contain a datagroup mc_info straight from the given files, as well as two datagroups
    per output layer of the network, which have the labels and the predicted values in them as numpy arrays, respectively.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Trained Keras model of a neural network.
    xs_mean : dict
        Mean images of the x dataset.
    eval_filename : str
        Name and path of the h5 file.
    samples : int or None
        Number of events that should be predicted. If samples=None, the whole file will be used.

    """
    batchsize = orca.cfg.batchsize
    compression = ("gzip", 1)
    file_sizes = orca.io.get_file_sizes("val")
    total_file_size = sum(file_sizes)
    datagroups_created = False

    with h5py.File(eval_filename, 'w') as h5_file:
        # For every val file set (one set can have multiple files if the model has multiple inputs):
        for f_number, files_dict in enumerate(orca.io.yield_files("val")):
            file_size = file_sizes[f_number]
            generator = generate_batches_from_hdf5_file(orca, files_dict, zero_center_image=xs_mean, yield_mc_info=True)

            if samples is None:
                steps = int(file_size / batchsize)
                if file_size % batchsize != 0:
                    # add a smaller step in the end
                    steps += 1
            else:
                steps = int(samples / batchsize)

            for s in range(steps):
                if s % 100 == 0:
                    print('Predicting in step ' + str(s) + ' on file ' + str(f_number))
                # y_true is a dict of ndarrays, mc_info is a structured array, y_pred is a list of ndarrays
                xs, y_true, mc_info = next(generator)

                y_pred = model.predict_on_batch(xs)
                if not isinstance(y_pred, list): # if only one output, transform to a list for the hacky below
                    y_pred = [y_pred]
                # transform y_pred to dict TODO hacky!
                y_pred = {out: y_pred[i] for i, out in enumerate(model.output_names)}

                if orca.cfg.dataset_modifier is None:
                    datasets = get_datasets(mc_info, y_true, y_pred)
                else:
                    datasets = orca.cfg.dataset_modifier(mc_info, y_true, y_pred)

                # TODO maybe add attr to data, like used files or orcanet version number?
                if not datagroups_created:
                    for dataset_name, data in datasets.items():
                        maxshape = (total_file_size,) + data.shape[1:]
                        chunks = True  # (batchsize,) + data.shape[1:]
                        h5_file.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=chunks,
                                               compression=compression[0], compression_opts=compression[1])
                        datagroups_created = True
                else:
                    for dataset_name, data in datasets.items():
                        # append data at the end of the dataset
                        h5_file[dataset_name].resize(h5_file[dataset_name].shape[0] + data.shape[0], axis=0)
                        h5_file[dataset_name][-data.shape[0]:] = data


def get_datasets(mc_info, y_true, y_pred):
    """
    Get the dataset names and numpy array contents.

    Every output layer will get one dataset each for both the label and the prediction.
    E.g. if your model has an output layer called "energy", the datasets
    "label_energy" and "pred_energy" will be made.

    Parameters
    ----------
    mc_info : ndarray
        A structured array containing infos for every event, right from the input files.
    y_true : dict
        The labels for each output layer of the network.
    y_pred : dict
        The predictions of each output layer of the network.

    Returns
    -------
    datasets : dict
        Keys are the name of the datagroups, values the content in the form of numpy arrays.

    """
    datasets = dict()
    datasets["mc_info"] = mc_info
    for out_layer_name in y_true:
        datasets["label_" + out_layer_name] = y_true[out_layer_name]
    for out_layer_name in y_pred:
        datasets["pred_" + out_layer_name] = y_pred[out_layer_name]
    return datasets
