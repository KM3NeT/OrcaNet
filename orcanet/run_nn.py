#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main code for training and validating NN's.
"""

import matplotlib as mpl
from inspect import signature
import keras.backend as K
mpl.use('Agg')
from orcanet.utilities.input_output_utilities import write_summary_logfile, write_full_logfile, read_logfiles
from orcanet.utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger, generate_batches_from_hdf5_file
from orcanet.utilities.visualization.visualization_tools import plot_all_metrics_to_pdf, plot_weights_and_activations


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
            lr = user_lr[0] * (1 - user_lr[1])**(epoch[1] + epoch[0]*len(cfg.get_train_files()))
        else:
            raise AssertionError(error_msg)

    elif isinstance(user_lr, str):
        if user_lr == "triple_decay":
            lr = triple_decay(epoch[0], epoch[1], cfg)
        else:
            raise NameError(user_lr, "is an unknown learning rate string!")

    elif callable(user_lr):
        # User defined function
        assert len(signature(user_lr).parameters) == 3, "A custom learning rate function must have three input parameters: \
        The epoch, the file number and the Configuration object."
        lr = user_lr(epoch[0], epoch[1], cfg)

    else:
        raise AssertionError(error_msg)
    return lr


def triple_decay(n_epoch, n_file, cfg):
    """
    Function that calculates the current learning rate based on the number of already trained epochs.

    Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                          lr_decay = 4% for 0.0003 >= lr > 0.0001
                                          lr_decay = 2% for 0.0001 >= lr

    Parameters
    ----------
    n_epoch : int
        The number of the current epoch which is used to calculate the new learning rate.
    n_file : int
        The number of the current filenumber which is used to calculate the new learning rate.
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    Returns
    -------
    lr_temp : float
        Calculated learning rate for this epoch.

    """

    n_lr_decays = (n_epoch - 1) * len(cfg.get_train_files()) + (n_file - 1)
    lr_temp = 0.005  # * n_gpu TODO think about multi gpu lr

    for i in range(n_lr_decays):
        if lr_temp > 0.0003:
            lr_decay = 0.07  # standard for regression: 0.07, standard for PID: 0.02
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04  # standard for regression: 0.04, standard for PID: 0.01
        else:
            lr_decay = 0.02  # standard for regression: 0.02, standard for PID: 0.005
        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp


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

    Trains (fit_generator) and validates (evaluate_generator) a Keras model once on the provided
    training and validation files. The model is saved with an automatically generated filename based on the epoch,
    log files are written and plots are made.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.Models.model
        Compiled keras model to use for training and validating.
    start_epoch : tuple
        Upcoming epoch and file number to start this training with.

    """
    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None

    for file_no, (files, f_size) in enumerate(cfg.get_train_files(), 1):
        # Only the file number changes during training, as this function trains only for one epoch
        curr_epoch = (start_epoch[0], file_no)
        # skip to the file with the target file number given in the start_epoch tuple.
        if curr_epoch[1] < start_epoch[1]:
            continue
        lr = get_learning_rate(cfg, curr_epoch)
        K.set_value(model.optimizer.lr, lr)
        print("Set learning rate to " + str(lr))
        # Train the model on one file and save it afterwards
        history_train = train_model(cfg, model, files, f_size, xs_mean, curr_epoch)
        model_filename = cfg.main_folder + 'saved_models/model_epoch_' + str(curr_epoch[0]) + '_file_' + str(curr_epoch[1]) + '.h5'
        model.save(model_filename)
        print("Saved model as " + model_filename)
        # Validate after every n-th file, starting with the first
        if (curr_epoch[1] - 1) % cfg.validate_after_n_train_files == 0:
            history_val = validate_model(cfg, model, xs_mean)
        else:
            history_val = None
        # Write logfiles
        write_summary_logfile(cfg, curr_epoch, model, history_train, history_val, K.get_value(model.optimizer.lr))
        write_full_logfile(cfg, model, history_train, history_val, K.get_value(model.optimizer.lr), curr_epoch, files)
        # Make plots
        update_summary_plot(cfg.main_folder)
        plot_weights_and_activations(cfg, model, xs_mean, curr_epoch)


def train_model(cfg, model, files, f_size, xs_mean, curr_epoch):
    """
    Trains a model on one file based on the Keras fit_generator method.

    The progress of the training is also logged.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    files : list
        Full filepath of the file (or files for multiple inputs) that should be used for training.
    f_size : int
        Number of images contained in f.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/val-data.
    curr_epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.

    """
    print('Training in epoch ' + str(curr_epoch[0]) + ' on file ' + str(curr_epoch[1]) + ' ,', files)
    if cfg.n_events is not None:
        # TODO Can throw an error if n_events is larger than the file
        f_size = cfg.n_events  # for testing purposes
    callbacks = [BatchLevelPerformanceLogger(cfg, model, curr_epoch), ]
    training_generator = generate_batches_from_hdf5_file(cfg, files, f_size=f_size, zero_center_image=xs_mean)
    history = model.fit_generator(training_generator, steps_per_epoch=int(f_size / cfg.batchsize), epochs=1,
                                  verbose=cfg.verbose_train, max_queue_size=10, callbacks=callbacks)
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
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/val-data.

    """
    # One history for each val file
    histories = []
    for i, (f, f_size) in enumerate(cfg.get_val_files()):
        print('Validating on file ', i, ',', str(f))

        if cfg.n_events is not None:
            f_size = cfg.n_events  # for testing purposes
        val_generator = generate_batches_from_hdf5_file(cfg, f, f_size=f_size, zero_center_image=xs_mean)
        history = model.evaluate_generator(val_generator, steps=int(f_size / cfg.batchsize), max_queue_size=10, verbose=cfg.verbose_val)
        # This history object is just a list, not a dict like with fit_generator!
        if type(history) != list:
            history = [history]
        print('Validation sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)
    history_val = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0]  # average over all val files if necessary

    return history_val

