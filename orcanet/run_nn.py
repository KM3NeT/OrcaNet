#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main code for running CNN's.
CONFIG is a .toml file which sets up the model. An example can be found in config/models/example_model.toml
LIST is a .list file which contains the files to be trained on. an example can be found in config/lists/example_list.list

Usage:
    run_cnn.py CONFIG LIST
    run_cnn.py (-h | --help)

Options:
    -h --help                       Show this screen.
"""

import os
import keras as ks
import matplotlib as mpl
from docopt import docopt
mpl.use('Agg')

from utilities.input_output_utilities import use_node_local_ssd_for_input, read_out_list_file, read_out_config_file, write_summary_logfile, write_full_logfile, read_logfiles, look_for_latest_epoch, get_n_bins
from model_archs.short_cnn_models import create_vgg_like_model_multi_input_from_single_nns, create_vgg_like_model
from model_archs.wide_resnet import create_wide_residual_network
from utilities.nn_utilities import load_zero_center_data, get_modelname, BatchLevelPerformanceLogger
from utilities.multi_gpu.multi_gpu import get_available_gpus, make_parallel, print_mgpu_modelsummary
from utilities.data_tools.shuffle_h5 import shuffle_h5
from utilities.visualization.visualization_tools import *
from utilities.evaluation_utilities import *
from utilities.losses import *

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def build_or_load_nn_model(epoch, nn_arch, n_bins, batchsize, class_type, swap_4d_channels, str_ident, path_of_model, custom_objects):
    """
    Function that either loads or builds (epoch = 0) a Keras nn model.

    Parameters
    ----------
    epoch : tuple(int, int)
        Declares if a previously trained model or a new model (=0) should be loaded, more info in the execute_nn function.
    nn_arch : str
        Architecture of the neural network.
    n_bins : list(tuple(int))
        Declares the number of bins for each dimension (e.g. (x,y,z,t)) in the train- and testfiles.
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    path_of_model : str
        Name and path of the nn model.
    custom_objects : dict
        Keras custom objects variable that contains custom loss functions for loading nn models.

    Returns
    -------
    model : ks.models.Model
        A Keras nn instance.

    """
    if epoch[0] == 0:
        if nn_arch == 'WRN': model = create_wide_residual_network(n_bins[0], batchsize, nb_classes=class_type[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=swap_4d_channels)

        elif nn_arch == 'VGG':
            if 'multi_input_single_train' in str_ident:
                model = create_vgg_like_model_multi_input_from_single_nns(n_bins, batchsize, str_ident, nb_classes=class_type[0], dropout=(0,0.1), swap_4d_channels=swap_4d_channels)

            else:
                model = create_vgg_like_model(n_bins, batchsize, class_type, dropout=0.0,
                                              n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), swap_4d_channels=swap_4d_channels) # 2 more layers

        else: raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')
    else:
        model = ks.models.load_model(path_of_model, custom_objects=custom_objects)

    # plot model, install missing packages with conda install if it throws a module error
    #ks.utils.plot_model(model, to_file='./models/model_plots/' + modelname + '.png', show_shapes=True, show_layer_names=True)

    return model


def get_optimizer_info(loss_opt, optimizer='adam'):
    """
    Returns optimizer information for the training procedure.

    Parameters
    ----------
    loss_opt : tuple
        A Tuple with len=2.
        loss_opt[0]: dict with dicts of loss functions and optionally weights that should be used for each nn output.
            Typically read in from a .toml file.
            Format: { loss : { function, weight } }
        loss_opt[1]: dict with metrics that should be used for each nn output.
    optimizer : str
        Specifies, if "Adam" or "SGD" should be used as optimizer.

    Returns
    -------
    loss_functions : dict
        Cf. loss_opt[0].
    metrics : dict
        Cf. loss_opt[1].
    loss_weight : dict
        Cf. loss_opt[2].
    optimizer : ks.optimizers
        Keras optimizer instance, currently either "Adam" or "SGD".

    """
    if optimizer == 'adam':
        optimizer = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0) # epsilon=1 for deep networks
    elif optimizer == "sgd":
        optimizer = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
    else:
        raise NameError("Unknown optimizer name ({})".format(optimizer))
    custom_objects = get_all_loss_functions()
    loss_functions, loss_weight = {},{}
    for loss_key in loss_opt[0].keys():
        # Replace function string with the actual function if it is custom
        if loss_opt[0][loss_key]["function"] in custom_objects:
            loss_function=custom_objects[loss_opt[0][loss_key]["function"]]
        else:
            loss_function=loss_opt[0][loss_key]["function"]
        # Use given weight, else use default weight of 1
        if "weight" in loss_opt[0][loss_key]:
            weight = loss_opt[0][loss_key]["weight"]
        else:
            weight = 1.0
        loss_functions[loss_key] = loss_function
        loss_weight[loss_key] = weight
    metrics = loss_opt[1] if not loss_opt[1] is None else []

    return loss_functions, metrics, loss_weight, optimizer


def parallelize_model_to_n_gpus(model, n_gpu, batchsize, loss_functions, optimizer, metrics, loss_weight):
    """
    Parallelizes the nn-model to multiple gpu's.

    Currently, up to 4 GPU's at Tiny-GPU are supported.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    batchsize : int
        Batchsize that is used for the training / inferencing of the cnn.
    loss_functions : dict/str
        Dict/str with loss functions that should be used for each nn output. # TODO fix, make single loss func also use dict
    optimizer : ks.optimizers
        Keras optimizer instance, currently either "Adam" or "SGD".
    metrics : dict/str/None
        Dict/str with metrics that should be used for each nn output.
    loss_weight : dict/None
        Dict with loss weights that should be used for each sub-loss.

    Returns
    -------
    model : ks.models.Model
        The parallelized Keras nn instance (multi_gpu_model).
    batchsize : int
        The new batchsize scaled by the number of used gpu's.

    """
    if n_gpu[1] == 'avolkov':
        if n_gpu[0] == 1:
            return model, batchsize
        else:
            assert n_gpu[0] > 1 and isinstance(n_gpu[0], int), 'You probably made a typo: n_gpu must be an int with n_gpu >= 1!'

            gpus_list = get_available_gpus(n_gpu[0])
            ngpus = len(gpus_list)
            print('Using GPUs: {}'.format(', '.join(gpus_list)))
            batchsize = batchsize * ngpus

            # Data-Parallelize the model via function
            model = make_parallel(model, gpus_list, usenccl=False, initsync=True, syncopt=False, enqueue=False)
            print_mgpu_modelsummary(model)

            model.compile(loss=loss_functions, optimizer=optimizer, metrics=metrics, loss_weights=loss_weight)  # TODO check if necessary

            return model, batchsize

    else:
        raise ValueError('Currently, no multi_gpu mode other than "avolkov" is available.')


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
                         n_events):
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
                                  train_logger_flush, n_events)
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
              shuffle, swap_4d_channels, str_ident, folder_name, train_logger_display, train_logger_flush, n_events=None):
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
        steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=2, max_queue_size=10,
        validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)

    return history


def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, str_ident, n_events=None):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.

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


def predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, folder_name):
    """
    Function that 1) makes predictions based on a Keras nn model and 2) investigates the performance of the model based on the predictions.

    Parameters
    ----------
    model : ks.model.Model
        Keras model of a neural network.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    batchsize : int
        Batchsize that is used for predicting.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
    str_ident : str
        Optional string identifier that gets appended to the modelname.
    modelname : str
        Name of the model.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    folder_name : str
        Name of the main folder, e.g. "user/trained_models/example_model".

    """
    # for layer in model.layers: # temp
    #     if 'batch_norm' in layer.name:
    #         layer.stateful = False
    arr_nn_pred = get_nn_predictions_and_mc_info(model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, modelname, samples=None)
    np.save(folder_name + '/predictions/arr_nn_pred_' + modelname + '.npy', arr_nn_pred)
    arr_nn_pred = np.load(folder_name + '/predictions/arr_nn_pred_' + modelname + '.npy')

    #arr_nn_pred = np.load('results/plots/saved_predictions/arr_nn_pred_' + modelname + '_final_stateful_false.npy')
    #arr_nn_pred = np.load('results/plots/saved_predictions//arr_nn_pred_model_VGG_4d_xyz-t_and_yzt-x_and_4d_xyzt_track-shower_multi_input_single_train_tight-1_tight-2_lr_0.003_tr_st_test_st_final_stateful_false_1-100GeV_precut.npy')

    if class_type[1] == 'track-shower':  # categorical
        precuts = (False, '3-100_GeV_prod')

        make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track', filename='results/plots/1d/track_shower/ts_' + modelname,
                                                      precuts=precuts, corr_cut_pred_0=0.5)

        make_prob_hists(arr_nn_pred, modelname=modelname, precuts=precuts)
        make_hist_2d_property_vs_property(arr_nn_pred, modelname, property_types=('bjorken-y', 'probability'),
                                          e_cut=(1, 100), precuts=precuts)
        calculate_and_plot_separation_pid(arr_nn_pred, modelname, precuts=precuts)

    else:  # regression
        arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')
        precuts = (True, 'regr_3-100_GeV_prod_and_1-3_GeV_prod')

        if 'energy' in class_type[1]:
            print('Generating plots for energy performance investigations')

            # DL
            make_2d_energy_resolution_plot(arr_nn_pred, modelname, precuts=precuts,
                                           correct_energy=(True, 'median'))
            make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median_relative', precuts=precuts,
                                                      correct_energy=(True, 'median'), compare_shallow=(True, arr_nn_pred_shallow))
            make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, precuts=precuts,
                                               compare_shallow=(True, arr_nn_pred_shallow), correct_energy=(True, 'median'))
            # shallow reco
            make_2d_energy_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', precuts=precuts)

        if 'dir' in class_type[1]:
            print('Generating plots for directional performance investigations')

            # DL
            make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', precuts=precuts,
                                              compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_dir_correlation_plot(arr_nn_pred, modelname, precuts=precuts)
            # shallow reco
            make_2d_dir_correlation_plot(arr_nn_pred_shallow, 'shallow_reco', precuts=precuts)

        if 'bjorken-y' in class_type[1]:
            print('Generating plots for bjorken-y performance investigations')

            # DL
            make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', precuts=precuts,
                                                    compare_shallow=(True, arr_nn_pred_shallow))
            make_2d_bjorken_y_resolution_plot(arr_nn_pred, modelname, precuts=precuts)
            # shallow reco
            make_2d_bjorken_y_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', precuts=precuts)

        if 'errors' in class_type[1]:
            print('Generating plots for error performance investigations')

            make_1d_reco_err_div_by_std_plot(arr_nn_pred, modelname, precuts=precuts) # TODO take precuts from above?
            make_1d_reco_err_to_reco_residual_plot(arr_nn_pred, modelname, precuts=precuts)
            make_2d_dir_correlation_plot_different_sigmas(arr_nn_pred, modelname, precuts=precuts)


def execute_nn(list_filename, folder_name,
               loss_opt, class_type, nn_arch, mode, swap_4d_channels,
                batchsize=64, epoch=[-1,-1], n_gpu=(1, 'avolkov'), use_scratch_ssd=False, zero_center=False, shuffle=(False,None),
                str_ident='', train_logger_display=100, train_logger_flush=-1, n_events=None):
    """
    Core code that trains or evaluates a neural network.

    Parameters
    ----------
    list_filename : str
        Path to a list file which contains pathes to all the h5 files that should be used for training.
    folder_name : str
        Name of the folder in the cnns directory in which everything will be saved.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower')
    nn_arch : str
        Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    batchsize : int
        Batchsize that should be used for the training / inferencing of the cnn.
    epoch : list[int, int]
        Declares if a previously trained model or a new model (=0) should be loaded.
        The first argument specifies the last epoch, and the second argument is the last train file number if the train
        dataset is split over multiple files. Can also give [-1,-1] to automatically load the most recent epoch.
    mode : str
        Specifies what the function should do - train & test a model or evaluate a 'finished' model?
        Currently, there are two modes available: 'train' & 'eval'.
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
    loss_opt : tuple(dict, dict/str/None,)
        Tuple that contains 1) the loss_functions and loss_weights as dicts, 2) the metrics.
    train_logger_display : int
        How many batches should be averaged for one line in the training logs.
    train_logger_flush : int
        After how many lines the file should be flushed. -1 for flush at the end of the epoch only.
    n_events : None/int
        For testing purposes. If not the whole .h5 file should be used for training, define the number of events.

    """
    make_folder_structure(folder_name)
    train_files, test_files, multiple_inputs = read_out_list_file(list_filename)
    n_bins = get_n_bins(train_files)
    if epoch == [-1, -1]:
        epoch = look_for_latest_epoch(folder_name)
        print("Found a saved model with epoch {} file {}, continuing training.".format(epoch[0], epoch[1]))
    if zero_center:
        xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu[0])
    else:
        xs_mean = None
    if use_scratch_ssd:
        train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs)
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)
    custom_objects = get_all_loss_functions()

    path_to_initial_model = folder_name + '/saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
    model = build_or_load_nn_model(epoch, nn_arch, n_bins, batchsize, class_type, swap_4d_channels, str_ident, path_to_initial_model, custom_objects)
    loss_functions, metrics, loss_weight, optimizer = get_optimizer_info(loss_opt, optimizer='adam')

    model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize, loss_functions, optimizer, metrics, loss_weight)
    model.summary()

    #model.compile(loss=loss_functions, optimizer=model.optimizer, metrics=model.metrics, loss_weights=loss_weight)
    if epoch[0] == 0:
        model.compile(loss=loss_functions, optimizer=optimizer, metrics=metrics, loss_weights=loss_weight)

    if mode == 'train':
        lr = None
        while 1:
            epoch, lr = train_and_test_model(model, train_files, test_files, batchsize, n_bins, class_type,
                                             xs_mean, epoch, shuffle, lr, swap_4d_channels, str_ident, n_gpu,
                                             folder_name, train_logger_display, train_logger_flush, n_events)

    elif mode == 'eval':
        predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                                  str_ident, modelname, xs_mean, folder_name)

    else:
        raise ValueError('Mode "', str(mode), '" is not known. Needs to be "train" or "eval".')


def parse_input():
    """
    Parses and returns all necessary input options from a .toml and a .list file.

    Returns
    -------
    config_file : str
        Path and name of the .toml file that defines the properties of the model.
    list_file : str
        Path and name of the .list file containing the names of the files that will be used for training.
    """
    args = docopt(__doc__)
    config_file = args['CONFIG']
    list_file = args['LIST']
    return config_file, list_file

def make_folder_structure(folder_name):
    """
    Make folders and subfolders if they don't exist already.

    Parameters
    ----------
    folder_name : str
        Name of the main folder, e.g. "user/trained_models/example_model".
    """
    folders_to_create = [folder_name, folder_name+"/log_train", folder_name+"/saved_models",
                         folder_name+"/plots/activations", folder_name+"/predictions"]
    for directory in folders_to_create:
        if not os.path.exists(directory):
            print("Creating directory: "+directory)
            os.makedirs(directory)

def main():
    """
    Parse the input and execute the script. The folder name where everything is saved to is the name of the .toml file.
    """
    config_file, list_file = parse_input()
    positional_arguments, keyword_arguments = read_out_config_file(config_file)
    folder_name = "user/trained_models/" + config_file.split("/")[-1].split(".")[:-1][0]
    execute_nn(list_file, folder_name, *positional_arguments, **keyword_arguments)


if __name__ == '__main__':
    main()



