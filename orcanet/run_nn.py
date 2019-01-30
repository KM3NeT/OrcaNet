#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code for training and validating NN's.
"""

import os
import matplotlib as mpl
from inspect import signature
import keras.backend as K
mpl.use('Agg')
from orcanet.utilities.input_output_utilities import write_summary_logfile, write_full_logfile, read_logfiles
from orcanet.utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger, generate_batches_from_hdf5_file
from orcanet.utilities.visualization.visualization_tools import plot_all_metrics_to_pdf, plot_weights_and_activations
from orcanet_contrib.contrib import orca_learning_rates

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def get_learning_rate(cfg, epoch):
    """
    Get the learning rate for the current epoch and file number.

    Parameters
    ----------
    cfg : object  Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
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
    user_lr = cfg.learning_rate
    error_msg = "The learning rate must be either a float, a tuple of two floats or a function."
    if isinstance(user_lr, float):
        # Constant LR
        lr = user_lr

    elif isinstance(user_lr, tuple) or isinstance(user_lr, list):
        if len(user_lr) == 2:
            # Exponentially decaying LR
            lr = user_lr[0] * (1 - user_lr[1])**(epoch[1] + epoch[0]*len(cfg.get_no_of_train_files()))
        else:
            raise AssertionError(error_msg, "(Your tuple has length "+str(len(user_lr))+")")

    elif isinstance(user_lr, str):
        if user_lr == "triple_decay":
            lr_schedule = orca_learning_rates("triple_decay")
            lr = lr_schedule(epoch[0], epoch[1], cfg)
        else:
            raise NameError(user_lr, "is an unknown learning rate string!")

    elif callable(user_lr):
        # User defined function
        assert len(signature(user_lr).parameters) == 3, "A custom learning rate function must have three input parameters: \
        The epoch, the file number and the Configuration object."
        lr = user_lr(epoch[0], epoch[1], cfg)

    else:
        raise AssertionError(error_msg, "(You gave a " + str(type(user_lr)) + ")")
    return lr


def update_summary_plot(main_folder):
    """
    Refresh the summary plot of a model directory, found in ./plots/summary_plot.pdf.

    Validation and Train-data will be read out automatically, and the loss as well as every metric will be plotted in
    a seperate page in the pdf.

    Parameters
    ----------
    main_folder : str
        Name of the main folder with the summary.txt in it.

    """
    summary_logfile = main_folder + "summary.txt"
    summary_data, full_train_data = read_logfiles(summary_logfile)
    pdf_name = main_folder + "plots/summary_plot.pdf"
    plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name)


def train_and_validate_model(cfg, model, start_epoch):
    """
    Train a model for one epoch, i.e. on all (remaining) train files once.

    Trains (fit_generator) and validates (evaluate_generator) a Keras model on the provided
    training and validation files for one epoch. The model is saved with an automatically generated filename based on the epoch,
    log files are written and summary plots are made.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.Models.model
        Compiled keras model to use for training and validating.
    start_epoch : tuple
        Upcoming epoch and file number.

    """
    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None
    f_sizes = cfg.get_train_file_sizes()

    for file_no, files_dict in enumerate(cfg.yield_train_files()):
        # no of samples in current file
        f_size = f_sizes[file_no]
        # Only the file number changes during training, as this function trains only for one epoch
        curr_epoch = (start_epoch[0], file_no + 1)
        # skip to the file with the target file number given in the start_epoch tuple.
        if curr_epoch[1] < start_epoch[1]:
            continue

        lr = get_learning_rate(cfg, curr_epoch)
        K.set_value(model.optimizer.lr, lr)
        print("Set learning rate to " + str(lr))

        # Train the model on one file and save it afterwards
        model_filename = cfg.get_model_path(curr_epoch[0], curr_epoch[1])
        assert not os.path.isfile(model_filename), "You tried to train your model in epoch {} file {}, but this model \
            has already been trained and saved!".format(curr_epoch[0], curr_epoch[1])
        history_train = train_model(cfg, model, files_dict, f_size, xs_mean, curr_epoch)
        model.save(model_filename)
        print("Saved model as " + model_filename)

        # Validate after every n-th file, starting with the first
        if (curr_epoch[1] - 1) % cfg.validate_after_n_train_files == 0:
            history_val = validate_model(cfg, model, xs_mean)
        else:
            history_val = None

        # Write logfiles and make plots
        write_summary_logfile(cfg, curr_epoch, model, history_train, history_val, K.get_value(model.optimizer.lr))
        write_full_logfile(cfg, model, history_train, history_val, K.get_value(model.optimizer.lr), curr_epoch, files_dict)
        update_summary_plot(cfg.main_folder)
        # TODO reimplement, this function throws errors all the time!
        # plot_weights_and_activations(cfg, model, xs_mean, curr_epoch)


def train_model(cfg, model, files_dict, f_size, xs_mean, curr_epoch):
    """
    Trains a model on one file based on the Keras fit_generator method.

    The progress of the training is also logged.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
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

    """
    print('Training in epoch ' + str(curr_epoch[0]) + ' on file ' + str(curr_epoch[1]) + ' ,', str(files_dict))
    if cfg.n_events is not None:
        # TODO Can throw an error if n_events is larger than the file
        f_size = cfg.n_events  # for testing purposes
    callbacks = [BatchLevelPerformanceLogger(cfg, model, curr_epoch), ]
    training_generator = generate_batches_from_hdf5_file(cfg, files_dict, f_size=f_size, zero_center_image=xs_mean, shuffle=cfg.shuffle_train)
    history = model.fit_generator(training_generator, steps_per_epoch=int(f_size / cfg.batchsize), epochs=1,
                                  verbose=cfg.verbose_train, max_queue_size=cfg.max_queue_size, callbacks=callbacks)
    return history


def validate_model(cfg, model, xs_mean):
    """
    Validates a model on all the validation datafiles based on the Keras evaluate_generator method.

    This is usually done after a session of training has been finished.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    xs_mean : dict
        Mean image of the dataset used for zero-centering. Every input as a key, ndarray as values.

    """
    # One history for each val file
    histories = []
    f_sizes = cfg.get_val_file_sizes()
    for i, files_dict in enumerate(cfg.yield_val_files()):
        print('Validating on file ', i+1, ',', str(files_dict))
        f_size = f_sizes[i]
        if cfg.n_events is not None:
            f_size = cfg.n_events  # for testing purposes
        val_generator = generate_batches_from_hdf5_file(cfg, files_dict, f_size=f_size, zero_center_image=xs_mean)
        history = model.evaluate_generator(val_generator, steps=int(f_size / cfg.batchsize), max_queue_size=cfg.max_queue_size, verbose=cfg.verbose_val)
        # This history object is just a list, not a dict like with fit_generator!
        print('Validation sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)
    history_val = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0]  # average over all val files if necessary

    return history_val

