#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Main code for training NN's. The main function for training, validating, logging and plotting the progress is orca_train.
It can also be called via a parser by running this python module as follows:

Usage:
    run_nn.py CONFIG LIST [FOLDER]
    run_nn.py (-h | --help)

Arguments:
    CONFIG  A .toml file which sets up the model and training.
            An example can be found in config/models/example_config.toml
    LIST    A .toml file which contains the pathes of the training and validation files.
            An example can be found in config/lists/example_list.toml
    FOLDER  A new subfolder will be generated in this folder, where everything from this model gets saved to.
            Default is the current working directory.

Options:
    -h --help                       Show this screen.

"""

import os
import keras as ks
import matplotlib as mpl
from docopt import docopt
import warnings

from utilities.input_output_utilities import write_summary_logfile, write_full_logfile, read_logfiles, write_full_logfile_startup, Settings
from utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger
from utilities.visualization.visualization_tools import *
from utilities.evaluation_utilities import *
from utilities.losses import *
from model_setup import build_nn_model

mpl.use('Agg')

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=0.003, manual_mode=(False, None, 0.0, None)):
    """
    Function that schedules a learning rate during training.

    If manual_mode[0] is False, the current lr will be automatically calculated if the training is resumed, based on the epoch variable.
    If manual_mode[0] is True, the final lr during the last training session (manual_mode[1]) and the lr_decay (manual_mode[1])
    have to be set manually.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    epoch : tuple(int, int)
        Declares if a previously trained model or a new model (=0) should be loaded.
        The first argument specifies the last epoch, and the second argument is the last train file number if the train
        dataset is split over multiple files.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
        Needed, because the lr is modified with multi GPU training.
    train_files : list(([train_filepaths], train_filesize))
        List with the paths and the filesizes of the train_files. Needed for calculating the lr.
    lr_initial : float
        Initial learning rate for the first epoch (and first file).
    manual_mode : tuple
        tuple(bool, None/float, float, None/float)
        Tuple that controls the options for the manual mode.
        manual_mode[0] = flag to enable the manual mode
        manual_mode[1] = lr value, of which the manual mode should start off
        manual_mode[2] = lr_decay during epochs
        manual_mode[3] = current lr, only used to check if this is the first instance of the while loop

    Returns
    -------
    epoch : tuple(int, int)
        The new train file number +=1 (& new epoch if last train_file).
    lr : int
        Learning rate that has been set for the model and for this epoch.
    lr_decay : float
        Learning rate decay that has been used to decay the lr rate used for this epoch.

    """
    # TODO set learning rate outside of this function
    if len(train_files) > epoch[1] and epoch[0] != 0:
        epoch = (epoch[0], epoch[1] + 1)  # resume from the same epoch but with the next file
    else:
        epoch = (epoch[0] + 1, 1)  # start new epoch from file 1

    if manual_mode[0] is True:
        lr = manual_mode[1] if manual_mode[3] is None else K.get_value(model.optimizer.lr)
        lr_decay = manual_mode[2]
        K.set_value(model.optimizer.lr, lr)

        if epoch[0] > 1 and lr_decay > 0:
            lr *= 1 - float(lr_decay)
            K.set_value(model.optimizer.lr, lr)
            print('Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) +
                  ' before epoch ' + str(epoch[0]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')')

    else:
        if epoch[0] == 1 and epoch[1] == 1:  # set initial learning rate for the training
            lr, lr_decay = lr_initial, 0.00
            # lr, lr_decay = lr_initial * n_gpu[0], 0.00
            K.set_value(model.optimizer.lr, lr)
            print('Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch[0]) +
                  ' and file ' + str(epoch[1]))
        else:
            n_train_files = len(train_files)
            lr, lr_decay = get_new_learning_rate(epoch, lr_initial, n_train_files, n_gpu[0])
            K.set_value(model.optimizer.lr, lr)
            print('Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) +
                  ' before epoch ' + str(epoch[0]) + ' and file ' + str(epoch[1]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')')

    return epoch, lr, lr_decay


def get_new_learning_rate(epoch, lr_initial, n_train_files, n_gpu):
    """
    Function that calculates the current learning rate based on the number of already trained epochs.

    Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                          lr_decay = 4% for 0.0003 >= lr > 0.0001
                                          lr_decay = 2% for 0.0001 >= lr

    Parameters
    ----------
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber which is used to calculate the new learning rate.
    lr_initial : float
        Initial lr for the first epoch. Typically 0.01 for SGD and 0.001 for Adam.
    n_train_files : int
        Specifies into how many files the training dataset is split.
    n_gpu : int
        Number of gpu's that are used during the training. Used for scaling the lr.

    Returns
    -------
    lr_temp : float
        Calculated learning rate for this epoch.
    lr_decay : float
        Latest learning rate decay that has been used.

    """
    n_epoch, n_file = epoch[0], epoch[1]
    n_lr_decays = (n_epoch - 1) * n_train_files + (n_file - 1)

    lr_temp = lr_initial  # * n_gpu TODO think about multi gpu lr
    lr_decay = None

    for i in range(n_lr_decays):

        if lr_temp > 0.0003:
            lr_decay = 0.07  # standard for regression: 0.07, standard for PID: 0.02
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04  # standard for regression: 0.04, standard for PID: 0.01
        else:
            lr_decay = 0.02  # standard for regression: 0.02, standard for PID: 0.005

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay


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


def train_and_validate_model(cfg, model, epoch):
    """
    Train a model for one epoch.

    Trains (fit_generator) and validates (evaluate_generator) a Keras model once on the provided
    training and validation files. The model is saved with an automatically generated filename based on the epoch.

    Parameters
    ----------
    cfg : Object Settings
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.Models.model
        Compiled keras model to use for training and validating.
    epoch : tuple
        Current Epoch and Fileno.

    """
    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg.get_train_files(), cfg.n_gpu[0], cfg.zero_center_folder, cfg.get_list_name())
    else:
        xs_mean = None

    lr = None
    lr_initial, manual_mode = 0.005, (False, 0.0003, 0.07, lr)
    epoch, lr, lr_decay = schedule_learning_rate(model, epoch, cfg.n_gpu, cfg.get_train_files(), lr_initial=lr_initial, manual_mode=manual_mode)  # begin new training step

    train_iter_step = 0  # loop n
    for file_no, (f, f_size) in enumerate(cfg.get_train_files(), 1):
        if file_no < epoch[1]:
            continue  # skip if this file for this epoch has already been used for training

        train_iter_step += 1
        if train_iter_step > 1:
            epoch, lr, lr_decay = schedule_learning_rate(model, epoch, cfg.n_gpu, cfg.get_train_files(), lr_initial=lr_initial, manual_mode=manual_mode)

        history_train = train_model(cfg, model, f, f_size, file_no, xs_mean, epoch)
        model.save(cfg.main_folder + 'saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5')

        # Validate every n-th file, starting with the first
        if (file_no - 1) % cfg.validate_after_n_train_files == 0:
            history_val = validate_model(cfg, model, xs_mean)
        else:
            history_val = None
        write_summary_logfile(cfg, epoch, model, history_train, history_val, lr)
        write_full_logfile(cfg, model, history_train, history_val, lr, lr_decay, epoch)
        update_summary_plot(cfg.main_folder)
        plot_weights_and_activations(cfg, xs_mean, epoch[0], file_no)

    return epoch, lr


def train_model(cfg, model, f, f_size, file_no, xs_mean, epoch):
    """
    Trains a model based on the Keras fit_generator method.

    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the val_files is used.

    Parameters
    ----------
    cfg : Object Settings
        Contains all the configurable options in the OrcaNet scripts.
    model : ks.model.Model
        Keras model instance of a neural network.
    f : list
        Full filepaths of the files that should be used for training.
    f_size : int
        Number of images contained in f.
    file_no : int
        If the full data is split into multiple files, this parameter indicates the current file number.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/val-data.
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.

    """
    validation_data, validation_steps, callbacks = None, None, []
    if cfg.n_events is not None:
        f_size = cfg.n_events  # for testing purposes
    callbacks.append(BatchLevelPerformanceLogger(cfg, epoch, model))
    print('Training in epoch', epoch[0], 'on file ', file_no, ',', f)

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
    cfg : Object Settings
        Contains all the configurable options in the OrcaNet scripts.
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
    cfg : Object Settings
        Contains all the configurable options in the OrcaNet scripts.
    initial_model : ks.models.Model
        Compiled keras model to use for training and validation. Only required for the first epoch of training, as a
        the most recent saved model will be loaded otherwise.

    """
    make_folder_structure(cfg.main_folder)
    write_full_logfile_startup(cfg)
    epoch = (cfg.initial_epoch, cfg.initial_fileno)
    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))

    if epoch[0] == 0 and epoch[1] == 1:
        if initial_model is None:
            raise ValueError("You need to provide a compiled keras model for the start of the training! (You gave None)")
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
        epoch, lr = train_and_validate_model(cfg, model, epoch)
        trained_epochs += 1


def make_folder_structure(main_folder):
    """
    Make subfolders for a specific model if they don't exist already. These subfolders will contain e.g. saved models,
    logfiles, etc.

    Parameters
    ----------
    main_folder : str
        Name of the main folder where everything gets saved to.

    """
    folders_to_create = [main_folder+"log_train", main_folder+"saved_models",
                         main_folder+"plots/activations", main_folder+"predictions"]
    for directory in folders_to_create:
        if not os.path.exists(directory):
            print("Creating directory: "+directory)
            os.makedirs(directory)


def example_run(main_folder, list_file, config_file):
    """
    This shows how to use OrcaNet.

    Parameters
    ----------
    main_folder : str
        Path to the folder where everything gets saved to, e.g. the summary.txt, the plots, the trained models, etc.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and validation.
    config_file : str
        Path to a .toml file which contains all the infos for training and validating a model.

    """
    cfg = Settings(main_folder, list_file, config_file)
    if cfg.get_latest_epoch() == (0, 1):
        initial_model = build_nn_model(cfg)
    else:
        initial_model = None
    orca_train(cfg, initial_model)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    config_file = args['CONFIG']
    list_file = args['LIST']
    main_folder = args['FOLDER'] if args['FOLDER'] is not None else "./"
    example_run(main_folder, list_file, config_file)


if __name__ == '__main__':
    parse_input()



