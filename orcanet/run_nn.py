#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main code for training NN's. The main function for training, validating, logging and plotting the progress is orca_train.
It can also be called via a parser by running this python module as follows:

Usage:
    run_nn.py FOLDER LIST CONFIG MODEL
    run_nn.py (-h | --help)

Arguments:
    FOLDER  Path to the folder where everything gets saved to, e.g. the summary.txt, the plots, the trained models, etc.
    LIST    A .toml file which contains the pathes of the training and validation files.
            An example can be found in config/lists/example_list.toml
    CONFIG  A .toml file which sets up the training.
            An example can be found in config/models/example_config.toml. The possible parameters are listed in
            utilities/input_output_utilities.py in the class Configuration.
    MODEL   Path to a .toml file with infos about a model.

Options:
    -h --help                       Show this screen.

"""

import os
import keras as ks
import matplotlib as mpl
from docopt import docopt
from inspect import signature
import warnings
mpl.use('Agg')
from orcanet.utilities.input_output_utilities import write_summary_logfile, write_full_logfile, read_logfiles, write_full_logfile_startup, Configuration
from orcanet.utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger
from orcanet.utilities.visualization.visualization_tools import *
from orcanet.utilities.evaluation_utilities import *
from orcanet.utilities.losses import *
from orcanet.model_setup import build_nn_model


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

    elif isinstance(user_lr, function):
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
    Refresh the summary plot of a model directory, found in ./plots/summary_plot.pdf. Val- and train-data
    will be read out automatically, and the loss and every metric will be plotted in a seperate page in the pdf.

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
    training and validation files. The model is saved with an automatically generated filename based on the epoch.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.Models.model
        Compiled keras model to use for training and validating.
    start_epoch : tuple
        Epoch and file number to start this training with.

    """
    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None

    for file_no, (f, f_size) in enumerate(cfg.get_train_files(), 1):
        # Only the file number changes during training, as this function trains only for one epoch
        curr_epoch = (start_epoch[0], file_no)
        # skip to the file with the target file number given in the epoch tuple.
        if curr_epoch[1] < start_epoch[1]:
            continue
        lr = get_learning_rate(cfg, curr_epoch)
        K.set_value(model.optimizer.lr, lr)
        print("Set learning rate to " + str(lr))
        # Train the model on one file
        history_train = train_model(cfg, model, f, f_size, xs_mean, curr_epoch)
        model.save(cfg.main_folder + 'saved_models/model_epoch_' + str(curr_epoch[0]) + '_file_' + str(curr_epoch[1]) + '.h5')
        # Validate after every n-th file, starting with the first
        if (curr_epoch[1] - 1) % cfg.validate_after_n_train_files == 0:
            history_val = validate_model(cfg, model, xs_mean)
        else:
            history_val = None
        # Write logfiles
        write_summary_logfile(cfg, curr_epoch, model, history_train, history_val, K.get_value(model.optimizer.lr))
        write_full_logfile(cfg, model, history_train, history_val, K.get_value(model.optimizer.lr), curr_epoch, f)
        # Make plots
        update_summary_plot(cfg.main_folder)
        plot_weights_and_activations(cfg, xs_mean, curr_epoch)


def train_model(cfg, model, f, f_size, xs_mean, curr_epoch):
    """
    Trains a model on a file based on the Keras fit_generator method.

    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the val_files is used.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    f : list
        Full filepath of the file (or files for multiple inputs) that should be used for training.
    f_size : int
        Number of images contained in f.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/val-data.
    curr_epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.

    """
    validation_data, validation_steps, callbacks = None, None, []
    if cfg.n_events is not None:
        f_size = cfg.n_events  # for testing purposes
    callbacks.append(BatchLevelPerformanceLogger(cfg, model, curr_epoch))
    print('Training in epoch ' + str(curr_epoch[0]) + ' on file ' + str(curr_epoch[1]) + ' ,', f)

    history = model.fit_generator(
        generate_batches_from_hdf5_file(cfg, f, f_size=f_size, zero_center_image=xs_mean),
        steps_per_epoch=int(f_size / cfg.batchsize), epochs=1, verbose=cfg.train_verbose, max_queue_size=10,
        validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)
    return history


def validate_model(cfg, model, xs_mean):
    """
    Validates a model on the validation data based on the Keras evaluate_generator method.
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
    histories = []
    for i, (f, f_size) in enumerate(cfg.get_val_files()):
        print('Validating on file ', i, ',', str(f))

        if cfg.n_events is not None:
            f_size = cfg.n_events  # for testing purposes

        history = model.evaluate_generator(
            generate_batches_from_hdf5_file(cfg, f, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / cfg.batchsize), max_queue_size=10, verbose=1)

        # This history object is just a list, not a dict like with fit_generator!
        print('Validation sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)
    history_val = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0]  # average over all val files if necessary

    return history_val


def orca_train(cfg, initial_model=None):
    """
    Core code that trains a neural network.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    initial_model : ks.models.Model
        Compiled keras model to use for training and validation. Only required for the first epoch of training, as
        the most recent saved model will be loaded otherwise.

    """
    if cfg.filter_out_tf_garbage:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg.make_folder_structure()
    write_full_logfile_startup(cfg)
    # The epoch that will be incremented during the scripts:
    epoch = (cfg.initial_epoch, cfg.initial_fileno)
    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))

    if epoch[0] == 0 and epoch[1] == 1:
        assert initial_model is not None, "You need to provide a compiled keras model for the start of the training! (You gave None)"
        model = initial_model
    else:
        # Load an existing model
        if initial_model is not None:
            warnings.warn("You provided a model even though this is not the start of the training. Provided model is ignored!")
        path_of_model = cfg.main_folder + 'saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
        print("Loading saved model: "+path_of_model)
        model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    model.summary()
    if cfg.use_scratch_ssd:
        cfg.use_local_node()

    trained_epochs = 0
    while trained_epochs < cfg.epochs_to_train or cfg.epochs_to_train == -1:
        train_and_validate_model(cfg, model, epoch)
        trained_epochs += 1
        epoch = (epoch[0] + 1, 1)


def example_run(main_folder, list_file, config_file, model_file):
    """
    This shows how to use OrcaNet.

    Parameters
    ----------
    main_folder : str
        Path to the folder where everything gets saved to, e.g. the summary log file, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which overwrite some of the default settings for training and validating a model.
    model_file : str
        Path to a file with parameters to build a model of a predefined architecture with OrcaNet.

    """
    # Set up the cfg object with the input data
    cfg = Configuration(main_folder, list_file, config_file)
    # If this is the start of the training, a compiled model needs to be handed to the orca_train function
    if cfg.get_latest_epoch() == (0, 1):
        # Add Info for building a model with OrcaNet to the cfg object
        cfg.set_from_model_file(model_file)
        # Build it
        initial_model = build_nn_model(cfg)
    else:
        # No model is required if the training is continued, as it will be loaded automatically
        initial_model = None
    orca_train(cfg, initial_model)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    main_folder = args['FOLDER']
    list_file = args['LIST']
    config_file = args['CONFIG']
    model_file = args['MODEL']
    example_run(main_folder, list_file, config_file, model_file)


if __name__ == '__main__':
    parse_input()
