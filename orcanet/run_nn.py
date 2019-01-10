#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Main code for training NN's. The main function for training, testing, logging and plotting the progress is orca_train.
It can also be called via a parser by running this python module as follows:

Usage:
    run_nn.py CONFIG LIST [FOLDER]
    run_nn.py (-h | --help)

Arguments:
    CONFIG  A .toml file which sets up the model and training.
            An example can be found in config/models/example_model.toml
    LIST    A .toml file which contains the pathes of the training and evaluation files.
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
mpl.use('Agg')

from utilities.input_output_utilities import use_node_local_ssd_for_input, read_out_list_file, read_out_config_file, write_summary_logfile, write_full_logfile, read_logfiles, look_for_latest_epoch, h5_get_n_bins, write_full_logfile_startup
from utilities.nn_utilities import load_zero_center_data, BatchLevelPerformanceLogger
from utilities.data_tools.shuffle_h5 import shuffle_h5
from utilities.visualization.visualization_tools import *
from utilities.evaluation_utilities import *
from utilities.losses import *
from model_setup import build_nn_model

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
    #TODO set learning rate outside of this function
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
            print('Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch[0]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')')

    else:
        if epoch[0] == 1 and epoch[1] == 1: # set initial learning rate for the training
            lr, lr_decay = lr_initial, 0.00
            #lr, lr_decay = lr_initial * n_gpu[0], 0.00
            K.set_value(model.optimizer.lr, lr)
            print('Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch[0]) + \
                  ' and file ' + str(epoch[1]))
        else:
            n_train_files = len(train_files)
            lr, lr_decay = get_new_learning_rate(epoch, lr_initial, n_train_files, n_gpu[0])
            K.set_value(model.optimizer.lr, lr)
            print('Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
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

    lr_temp = lr_initial # * n_gpu TODO think about multi gpu lr
    lr_decay = None

    for i in range(n_lr_decays):

        if lr_temp > 0.0003:
            lr_decay = 0.07 # standard for regression: 0.07, standard for PID: 0.02
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04 # standard for regression: 0.04, standard for PID: 0.01
        else:
            lr_decay = 0.02 # standard for regression: 0.02, standard for PID: 0.005

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay


def train_and_test_model(model, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, swap_4d_channels, str_ident, n_gpu, folder_name, train_logger_display, train_logger_flush,
                         train_verbose, n_events):
    """
    Convenience function that trains (fit_generator), tests (evaluate_generator) and saves a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    lr_initial, manual_mode = 0.005, (False, 0.0003, 0.07, lr)
    test_after_n_train_files = 2

    epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=lr_initial, manual_mode=manual_mode) # begin new training step
    train_iter_step = 0 # loop n
    for file_no, (f, f_size) in enumerate(train_files, 1):
        if file_no < epoch[1]:
            continue # skip if this file for this epoch has already been used for training

        train_iter_step += 1
        if train_iter_step > 1:
            epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=lr_initial, manual_mode=manual_mode)

        history_train = fit_model(model, train_files, f, f_size, file_no, batchsize, n_bins, class_type, xs_mean, epoch,
                                  shuffle, swap_4d_channels, str_ident, folder_name, train_logger_display,
                                  train_logger_flush, train_verbose, n_events)
        model.save(folder_name + '/saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5')

        # test after the first and else after every n-th file
        if file_no == 1 or file_no % test_after_n_train_files == 0:
            history_test = evaluate_model(model, test_files, batchsize, n_bins, class_type,
                                          xs_mean, swap_4d_channels, str_ident, n_events)
        else:
            history_test = None
        write_summary_logfile(train_files, batchsize, epoch, folder_name, model, history_train, history_test, lr)
        write_full_logfile(model, history_train, history_test, lr, lr_decay, epoch,
                                              f, test_files, batchsize, n_bins, class_type, swap_4d_channels, str_ident,
                                              folder_name)
        update_summary_plot(folder_name)
        plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels,
                                     epoch[0], file_no, str_ident, folder_name)

    return epoch, lr


def update_summary_plot(folder_name):
    """
    Refresh the summary plot of a model directory, found in ./plots/summary_plot.pdf. Test- and train-data
    will be read out automatically, and the loss and every metric will be plotted in a seperate page in the pdf.

    Parameters
    ----------
    folder_name : str
        Name of the main folder with the summary.txt in it.

    """
    summary_logfile = folder_name + "/summary.txt"
    summary_data, full_train_data = read_logfiles(summary_logfile)
    pdf_name = folder_name + "/plots/summary_plot.pdf"
    plot_all_metrics_to_pdf(summary_data, full_train_data, pdf_name)


def fit_model(model, train_files, f, f_size, file_no, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, str_ident, folder_name, train_logger_display, train_logger_flush, train_verbose,
              n_events=None):
    """
    Trains a model based on the Keras fit_generator method.

    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.

    Parameters
    ----------
    model : ks.model.Model
        Keras model instance of a neural network.
    train_files : list(([train_filepaths], train_filesize))
        List of tuples with the filepaths and the filesizes of the train_files.
    f : list
        Full filepaths of the files that should be used for training.
    f_size : int
        Number of images contained in f.
    file_no : int
        If the full data is split into multiple files, this parameter indicates the current file number.
    batchsize : int
        Batchsize that is used in the fit_generator method.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.
    shuffle : tuple(bool, None/int)
        Declares if the training data should be shuffled before the next training epoch.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    folder_name : str
        Path of the main folder.
    train_logger_display : int
        How many batches should be averaged for one line in the training logs.
    train_logger_flush : int
        After how many lines the file should be flushed. -1 for flush at the end of the epoch only.
    train_verbose : int
        verbose option of keras.model.fit_generator.
    n_events : None/int
        For testing purposes if not the whole .h5 file should be used for training.

    """
    validation_data, validation_steps, callbacks = None, None, []
    if n_events is not None: f_size = n_events  # for testing purposes
    logger = BatchLevelPerformanceLogger(train_files=train_files, batchsize=batchsize, display=train_logger_display, model=model,
                                         folder_name=folder_name, epoch=epoch, flush_after_n_lines=train_logger_flush)
    callbacks.append(logger)

    if epoch[0] > 1 and shuffle[0] is True: # just for convenience, we don't want to wait before the first epoch each time
        print('Shuffling file ', f, ' before training in epoch ', epoch[0], ' and file ', file_no)
        shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)

    if shuffle[1] is not None:
        n_preshuffled = shuffle[1]
        f = f.replace('0.h5', str(epoch[0]-1) + '.h5') if epoch[0] <= n_preshuffled else f.replace('0.h5', str(np.random.randint(0, n_preshuffled+1)) + '.h5')

    print('Training in epoch', epoch[0], 'on file ', file_no, ',', f)

    history = model.fit_generator(
        generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, str_ident, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels),
        steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=train_verbose, max_queue_size=10,
        validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)

    return history


def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, str_ident, n_events=None):
    """
    Evaluates a model on the validation data based on the Keras evaluate_generator method.
    This is usually done after a session of training has been finished.

    Parameters
    ----------
    model : ks.model.Model
        Keras model instance of a neural network.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows.
    batchsize : int
        Batchsize that is used in the evaluate_generator method.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    n_events : None/int
        For testing purposes if not the whole .h5 file should be used for testing.

    """
    histories = []
    for i, (f, f_size) in enumerate(test_files):
        print('Testing on file ', i, ',', str(f))

        if n_events is not None: f_size = n_events  # for testing purposes

        history = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, str_ident, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / batchsize), max_queue_size=10, verbose=1)
        #This history object is just a list, not a dict like with fit_generator!
        print('Test sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)
    history_test = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0] # average over all test files if necessary

    return history_test


def execute_nn(list_filename, folder_name, loss_opt, class_type, nn_arch,
               swap_4d_channels=None, batchsize=64, epoch=[-1,-1], epochs_to_train=-1, n_gpu=(1, 'avolkov'), use_scratch_ssd=False,
               zero_center=False, shuffle=(False,None), str_ident='', train_logger_display=100, train_logger_flush=-1,
               train_verbose=2, n_events=None):
    """
    Core code that trains a neural network.

    Parameters
    ----------
    list_filename : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and evaluation.
    folder_name : str
        Name of the folder of this model in which everything will be saved. E.g., the summary.txt log file is located in here.
    loss_opt : tuple(dict, dict/str/None,)
        Tuple that contains 1) the loss_functions and loss_weights as dicts (this is the losses table from the toml file)
        and 2) the metrics.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower')
    nn_arch : str
        Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    batchsize : int
        Batchsize that should be used for the training / inferencing of the cnn.
    epoch : List[int, int]
        Declares if a previously trained model or a new model (=0) should be loaded.
        The first argument specifies the last epoch, and the second argument is the last train file number if the train
        dataset is split over multiple files. Can also give [-1,-1] to automatically load the most recent epoch.
    epochs_to_train : int
        How many new epochs should be trained by running this function. -1 for infinite.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X, TODO add multi input options
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    use_scratch_ssd : bool
        Declares if the input files should be copied to the node-local SSD scratch space (only working at Erlangen CC).
    zero_center : bool
        Declares if the input images ('xs') should be zero-centered before training.
    shuffle : tuple(bool, None/int)
        Declares if the training data should be shuffled before the next training epoch [0].
        If the train dataset is too large to be shuffled in place, one can preshuffle them n times before running
        OrcaNet, the number n should then be put into [1].
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    train_logger_display : int
        How many batches should be averaged for one line in the training log files.
    train_logger_flush : int
        After how many lines the training log file should be flushed. -1 for flush at the end of the epoch only.
    train_verbose : int
        verbose option of keras.model.fit_generator.
    n_events : None/int
        For testing purposes. If not the whole .h5 file should be used for training, define the number of events.

    """
    train_files, test_files, multiple_inputs = read_out_list_file(list_filename)
    if epoch[0] == -1 and epoch[1] == -1:
        epoch = look_for_latest_epoch(folder_name)
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))
    n_bins = h5_get_n_bins(train_files)

    if epoch[0] == 0 and epoch[1] == 1:
        # Create and compile a new model
        model = build_nn_model(nn_arch, n_bins, class_type, swap_4d_channels, str_ident, loss_opt, n_gpu, batchsize)
    else:
        # Load an existing model
        path_of_model = folder_name + '/saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
        model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    model.summary()

    if zero_center:
        xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu[0])
    else:
        xs_mean = None
    if use_scratch_ssd:
        train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs)

    lr = None
    trained_epochs = 0
    while trained_epochs<epochs_to_train or epochs_to_train==-1:
        epoch, lr = train_and_test_model(model, train_files, test_files, batchsize, n_bins, class_type,
                                         xs_mean, epoch, shuffle, lr, swap_4d_channels, str_ident, n_gpu,
                                         folder_name, train_logger_display, train_logger_flush, train_verbose,
                                         n_events)
        trained_epochs+=1


def make_folder_structure(folder_name):
    """
    Make subfolders for a specific model if they don't exist already. These subfolders will contain e.g. saved models,
    logfiles, etc.

    Parameters
    ----------
    folder_name : str
        Name of the main folder, e.g. "user/trained_models/example_model".

    """
    folders_to_create = [folder_name+"/log_train", folder_name+"/saved_models",
                         folder_name+"/plots/activations", folder_name+"/predictions"]
    for directory in folders_to_create:
        if not os.path.exists(directory):
            print("Creating directory: "+directory)
            os.makedirs(directory)


def orca_train(trained_models_folder, config_file, list_file):
    """
    Frontend function for training networks.

    Parameters
    ----------
    trained_models_folder : str
        Path to the folder where everything gets saved to.
        Every model (from a .toml file) will get its own folder in here, with the name being the
        same as the one from the .toml file.
    config_file : str
        Path to a .toml file which contains all the infos for training and testing of a model.
    list_file : str
        Path to a list file which contains pathes to all the h5 files that should be used for training and evaluation.

    """
    keyword_arguments = read_out_config_file(config_file)
    folder_name = trained_models_folder + str(os.path.splitext(os.path.basename(config_file))[0])
    make_folder_structure(folder_name)
    write_full_logfile_startup(folder_name, list_file, keyword_arguments)
    execute_nn(list_file, folder_name, **keyword_arguments)


def parse_input():
    """ Run the orca_train function with a parser. """
    args = docopt(__doc__)
    config_file = args['CONFIG']
    list_file = args['LIST']
    trained_models_folder = args['FOLDER'] if args['FOLDER'] is not None else "./"
    orca_train(trained_models_folder, config_file, list_file)


if __name__ == '__main__':
    parse_input()



