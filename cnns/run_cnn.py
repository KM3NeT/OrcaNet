#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import os
import time
from time import gmtime, strftime
import shutil
import sys
import keras as ks
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')

from utilities.input_utilities import *
from models.short_cnn_models import *
from models.wide_resnet import *
from utilities.cnn_utilities import *
from utilities.multi_gpu.multi_gpu import *
from utilities.data_tools.shuffle_h5 import shuffle_h5
from utilities.visualization.visualization_tools import *
from utilities.evaluation_utilities import *
from utilities.losses import *
from utilities.losses import get_all_loss_functions

# for debugging
# from tensorflow.python import debug as tf_debug
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def build_or_load_nn_model(epoch, nn_arch, n_bins, batchsize, class_type, swap_4d_channels, str_ident, modelname, custom_objects):
    """
    Function that either loads or builds (epoch = 0) a Keras nn model.

    Parameters
    ----------
    epoch : tuple(int, int)
        Declares if a previously trained model or a new model (=0) should be loaded, more info in the execute_cnn function.
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
    modelname : str
        Name of the nn model.
    custom_objects : dict
        Keras custom objects variable that contains custom loss functions for loading nn models.

    Returns
    -------
    model : ks.models
        A Keras nn instance.

    """
    if epoch[0] == 0:
        if nn_arch is 'WRN': model = create_wide_residual_network(n_bins[0], batchsize, nb_classes=class_type[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=swap_4d_channels)

        elif nn_arch is 'VGG':
            if 'multi_input_single_train' in str_ident:
                model = create_vgg_like_model_multi_input_from_single_nns(n_bins, batchsize, str_ident, nb_classes=class_type[0], dropout=(0,0.1), swap_4d_channels=swap_4d_channels)

            else:
                model = create_vgg_like_model(n_bins, batchsize, class_type, dropout=0.0,
                                              n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), swap_4d_channels=swap_4d_channels) # 2 more layers

        else: raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')
    else:
        model = ks.models.load_model('models/trained/trained_' + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5', custom_objects=custom_objects)

    # plot model, install missing packages with conda install if it throws a module error
    ks.utils.plot_model(model, to_file='./models/model_plots/' + modelname + '.png', show_shapes=True, show_layer_names=True)

    return model


def get_optimizer_info(loss_opt, optimizer='adam'):
    """
    Returns optimizer information for the training procedure.

    Parameters
    ----------
    loss_opt : tuple
        A Tuple with len=3.
        loss_opt[0]: dict with loss functions that should be used for each nn output.
        loss_opt[1]: dict with metrics that should be used for each nn output.
        loss_opt[2]: dict with loss weights that should be used for each sub-loss.
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
    sgd = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
    adam = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0) # epsilon=1 for deep networks
    optimizer = adam if optimizer == 'adam' else sgd

    loss_functions = loss_opt[0]
    metrics = loss_opt[1] if not loss_opt[1] is None else []
    loss_weight = loss_opt[2]

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
    model : ks.models
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
    manual_mode : tuple(bool, None/float, float, None/float)
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
        if epoch[0] == 1 and epoch[1] == 1:
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
            lr_decay = 0.07 # standard for PID: 0.07, standard for regression: 0.02
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04 # standard for PID: 0.04, standard for regression: 0.01
        else:
            lr_decay = 0.02 # standard for PID: 0.02, standard for regression: 0.005

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay


def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, tb_logger, swap_4d_channels, str_ident, n_gpu):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    lr_initial, manual_mode = 0.005, (True, 0.0003, 0.07, lr)
    test_after_n_train_files = 2

    epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=lr_initial, manual_mode=manual_mode) # begin new training step
    train_iter_step = 0 # loop n
    for file_no, (f, f_size) in enumerate(train_files, 1):

        if file_no < epoch[1]:
            continue # skip if this file for this epoch has already been used for training

        train_iter_step += 1

        if train_iter_step > 1: epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=lr_initial, manual_mode=manual_mode)

        history_train = fit_model(model, modelname, train_files, f, f_size, file_no, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                                            shuffle, swap_4d_channels, str_ident, n_events=None, tb_logger=tb_logger)

        history_test = None
        # test after the first and else after every 5th file
        if file_no == 1 or file_no % test_after_n_train_files == 0:
            history_test = evaluate_model(model, modelname, test_files, train_files, batchsize, n_bins, class_type,
                                          xs_mean, epoch, swap_4d_channels, str_ident, n_events=None)

        save_train_and_test_statistics_to_txt(model, history_train, history_test, modelname, lr, lr_decay, epoch,
                                              f, test_files, batchsize, n_bins, class_type, swap_4d_channels, str_ident)
        plot_train_and_test_statistics(modelname, model)
        plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels, modelname, epoch[0], file_no, str_ident)

    return epoch, lr


def fit_model(model, modelname, train_files, f, f_size, file_no, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, str_ident, n_events=None, tb_logger=False):
    """
    Trains a model based on the Keras fit_generator method.

    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.

    Parameters
    ----------
    model : ks.model.Model
        Keras model instance of a neural network.
    modelname : str
        Name of the model.
    train_files : list(([train_filepaths], train_filesize))
        List of tuples with the filepaths and the filesizes of the train_files.
    f : str
        Full filepath of the file that should be used for training.
    f_size : int
        Number of images contained in f.
    file_no : int
        If the full data is split into multiple files, this parameter indicates the current file number.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows for the tb_callback.
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
    n_events : None/int
        For testing purposes if not the whole .h5 file should be used for training.
    tb_logger : bool
        Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log during training!).

    """
    if tb_logger is True:
        callbacks = [TensorBoardWrapper(generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, str_ident, zero_center_image=xs_mean),
                                     nb_steps=int(5000 / batchsize), log_dir='models/trained/tb_logs/' + modelname + '_{}'.format(time.time()),
                                     histogram_freq=1, batch_size=batchsize, write_graph=False, write_grads=True, write_images=True)]
        validation_data = generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, str_ident, swap_col=swap_4d_channels, zero_center_image=xs_mean) #f_size=None is ok here
        validation_steps = int(5000 / batchsize)
    else:
        validation_data, validation_steps, callbacks = None, None, []

    if n_events is not None: f_size = n_events  # for testing purposes

    logger = BatchLevelPerformanceLogger(train_files=train_files, batchsize=batchsize, display=100, model=model, modelname=modelname, epoch=epoch)
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
        steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=1, max_queue_size=10,
        validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)
    model.save('models/trained/trained_' + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(file_no) + '.h5')

    return history


def evaluate_model(model, modelname, test_files, train_files, batchsize, n_bins, class_type, xs_mean, epoch, swap_4d_channels, str_ident, n_events=None):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.

    Parameters
    ----------
    model : ks.model.Model
        Keras model instance of a neural network.
    modelname : str
        Name of the model.
    test_files : list(([test_filepaths], test_filesize))
        List of tuples that contains the testfiles and their number of rows.
    train_files : list(([train_filepaths], train_filesize))
        List of tuples with the filepaths and the filesizes of the train_files.
    batchsize : int
        Batchsize that is used in the evaluate_generator method.
    n_bins : list(tuple(int))
        Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
    xs_mean : ndarray
        Mean_image of the x (train-) dataset used for zero-centering the train-/testdata.
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.
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
        print('Test sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')')
        histories.append(history)

    history_averaged = [sum(col) / float(len(col)) for col in zip(*histories)] if len(histories) > 1 else histories[0] # average over all test files if necessary

    # Save test log
    steps_per_total_epoch, steps_cum = 0, [0] # get this for the epoch_number_float in the logfile
    for f, f_size in train_files:
        steps_per_file = int(f_size / batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)

    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)

    logfile_fname = 'models/trained/perf_plots/log_test_' + modelname + '.txt'
    logfile = open(logfile_fname, 'a+')
    if os.stat(logfile_fname).st_size == 0:
        logfile.write('#Epoch\t')
        for i, metric in enumerate(model.metrics_names):
            logfile.write(metric)
            logfile.write('\t') if i + 1 < len(model.metrics_names) else logfile.write('\n') #

    logfile.write(str(epoch_number_float) + '\t')

    for i in range(len(model.metrics_names)):
        logfile.write(str(history_averaged[i]))
        logfile.write('\t') if i + 1 < len(model.metrics_names) else logfile.write('\n')

    return history_averaged


def save_train_and_test_statistics_to_txt(model, history_train, history_test, modelname, lr, lr_decay, epoch, train_file,
                                          test_files, batchsize, n_bins, class_type, swap_4d_channels, str_ident):
    """
    Function for saving various information during training and testing to a .txt file.
    """
    with open('models/trained/train_logs/log_' + modelname + '.txt', 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('\n')
        f_out.write('Current time: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        f_out.write('Decayed learning rate to ' + str(lr) + ' before epoch ' + str(epoch[0]) +
                    ' and file ' + str(epoch[1]) + ' (minus ' + str(lr_decay) + ')\n')
        f_out.write('Trained in epoch ' + str(epoch) + ' on file ' + str(epoch[1]) + ', ' + str(train_file) + '\n')
        if history_test is not None:
            f_out.write('Tested in epoch ' + str(epoch) + ', file ' + str(epoch[1]) + ' on test_files ' + str(test_files) + '\n')
        f_out.write('History for training / testing: \n')
        f_out.write('Train: ' + str(history_train.history) + '\n')
        if history_test is not None:
            f_out.write('Test: ' + str(history_test) + ' (' + str(model.metrics_names) + ')' + '\n')
        f_out.write('\n')
        f_out.write('Additional Info:\n')
        f_out.write('Batchsize=' + str(batchsize) + ', n_bins=' + str(n_bins) +
                    ', class_type=' + str(class_type) + '\n' +
                    'swap_4d_channels=' + str(swap_4d_channels) + ', str_ident=' + str_ident + '\n')
        f_out.write('\n')


def predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean):
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

    """
    # for layer in model.layers: # temp
    #     if 'batch_norm' in layer.name:
    #         layer.stateful = False
    arr_nn_pred = get_nn_predictions_and_mc_info(model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, modelname, samples=None)
    np.save('results/plots/saved_predictions/arr_nn_pred_' + modelname + '.npy', arr_nn_pred)
    arr_nn_pred = np.load('results/plots/saved_predictions/arr_nn_pred_' + modelname + '.npy')

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


def execute_cnn(n_bins, class_type, nn_arch, batchsize, epoch, n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None,
                use_scratch_ssd=False, zero_center=False, shuffle=(False,None), tb_logger=False, str_ident='',
                loss_opt=('categorical_crossentropy', 'accuracy', None)):
    """
    Code that trains or evaluates a convolutional neural network.

    Parameters
    ----------
    n_bins : list(tuple(int))
        Declares the number of bins for each dimension (e.g. (x,y,z,t)) in the train- and testfiles. Can contain multiple n_bins tuples.
        Multiple n_bins tuples are currently used for multi-input models with multiple input files per batch.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower')
    nn_arch : str
        Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    batchsize : int
        Batchsize that should be used for the training / inferencing of the cnn.
    epoch : tuple(int, int)
        Declares if a previously trained model or a new model (=0) should be loaded.
        The first argument specifies the last epoch, and the second argument is the last train file number if the train
        dataset is split over multiple files.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    mode : str
        Specifies what the function should do - train & test a model or evaluate a 'finished' model?
        Currently, there are two modes available: 'train' & 'eval'.
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X, TODO add multi input options
    use_scratch_ssd : bool
        Declares if the input files should be copied to the node-local SSD scratch space (only working at Erlangen CC).
    zero_center : bool
        Declares if the input images ('xs') should be zero-centered before training.
    shuffle : tuple(bool, None/int)
        Declares if the training data should be shuffled before the next training epoch [0].
        If the train dataset is too large to be shuffled in place, one can preshuffle them n times before running
        OrcaNet, the number n should then be put into [1].
    tb_logger : bool
        Declares if a tb_callback should be used during training (takes longer to train due to overhead!).
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    loss_opt : tuple(dict/str, dict/str/None, dict/None)
        Tuple that contains 1) the loss_functions, 2) the metrics and 3) the loss_weights.

    """
    train_files, test_files, multiple_inputs = parse_input()
    xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu[0]) if zero_center is True else None
    if use_scratch_ssd is True: train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs)
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)
    custom_objects = get_all_loss_functions()

    model = build_or_load_nn_model(epoch, nn_arch, n_bins, batchsize, class_type, swap_4d_channels, str_ident, modelname, custom_objects)
    loss_functions, metrics, loss_weight, optimizer = get_optimizer_info(loss_opt, optimizer='adam')

    model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize, loss_functions, optimizer, metrics, loss_weight)
    model.summary()

    #model.compile(loss=loss_functions, optimizer=model.optimizer, metrics=model.metrics, loss_weights=loss_weight)
    if epoch[0] == 0: model.compile(loss=loss_functions, optimizer=optimizer, metrics=metrics, loss_weights=loss_weight)

    if mode == 'train':
        lr = None
        while 1:
            epoch, lr = train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type,
                                             xs_mean, epoch, shuffle, lr, tb_logger, swap_4d_channels, str_ident, n_gpu)

    elif mode == 'eval':
        predict_and_investigate_model_performance(model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                                  str_ident, modelname, xs_mean)

    else:
        raise ValueError('Mode "', str(mode), '" is not known. Needs to be "train" or "eval".')


if __name__ == '__main__':
    # available class_types:
    # - (2, 'track-shower')
    # - (5, 'energy_and_direction_and_bjorken-y')

    ###############
    #--- YZT-X ---#
    ###############

    ### Larger Production
    # tight-1, pad valid, dense 128
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(1,2), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1_pad-valid')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_no_tau.list lists/lp/xyz-t_lp_tight-1_test_no_tau.list

    # tight-2, pad valid, dense 128
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(1,2), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-2_bs64_dp0.1_pad-valid')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-2_test_no_tau.list

    ## Regression
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(5, 'energy_and_direction_and_bjorken-y'), nn_arch='VGG', batchsize=64, epoch=(0,1), use_scratch_ssd=True, loss_opt=('mean_absolute_error', 'mean_squared_error'),
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_muon-CC_and_elec-CC_even_split.list lists/lp/xyz-t_lp_tight-1_test_muon-CC_and_elec-CC_even_split.list

    ###############
    #--- XYZ-C ---#
    ###############
    # xyz-channel, timecut all, has new geo Stefan
    # execute_cnn(n_bins=[(11,13,18,31)], class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='VGG', batchsize=32, epoch=(26,1), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='channel_id_all_time')

    ###############
    #--- XYZ-T ---#
    ###############

    ### Larger Production
    # standard tight-1 // 2 additional layers: n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}, bs 64, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(8,4), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1')

    # standard tight-1 as above, padding valid, but 128 first dense! // 2 additional layers: n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}, bs 64, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(2,2), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1_pad-valid_dense-128')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_no_tau.list lists/lp/xyz-t_lp_tight-1_test_no_tau.list

    # standard tight-2, padding valid, but 12 first dense! // 2 additional layers: n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}, bs 64, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(2,1), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-2_bs64_dp0.1_pad-valid_dense-128')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-2_test_no_tau.list

    ######## REGRESSION, Larger Production
    # e+dir+by dp 0.0
    # losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
    #           'energy': 'mean_absolute_error', 'bjorken-y': 'mean_absolute_error'}
    # loss_metrics = {'dir_x': 'mean_squared_error', 'dir_y': 'mean_squared_error', 'dir_z': 'mean_squared_error',
    #           'energy': 'mean_squared_error', 'bjorken-y': 'mean_squared_error'}
    # loss_weights = {'dir_x': 40, 'dir_y': 40, 'dir_z': 40, 'energy': 1, 'bjorken-y': 40}
    # # # TODO change lr back, change cnn_utilities supplier back, enable plot functions in training loop
    # #
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(5, 'energy_and_direction_and_bjorken-y'), nn_arch='VGG', batchsize=64, epoch=(28,4), use_scratch_ssd=False, loss_opt=(losses, None, loss_weights),
    #             n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_muon-CC_and_elec-CC_even_split.list lists/lp/xyz-t_lp_tight-1_test_muon-CC_and_elec-CC_even_split.list # OLD

    # e+dir+by dp 0.0, with new loss functions + errors
    # losses = {'dir': 'mean_absolute_error', 'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
    #           'dir_err': loss_uncertainty_gaussian_likelihood_dir,
    #           'e_err': loss_uncertainty_gaussian_likelihood,
    #           'by_err': loss_uncertainty_gaussian_likelihood}
    # loss_weights = {'dir': 6, 'e': 1, 'by': 10, 'dir_err': 1, 'e_err': 1, 'by_err': 1}
    # mae dir, mae en
    # execute_cnn(n_bins=[(11,13,18,300)], class_type=(6, 'energy_dir_bjorken-y_and_errors_dir_new_loss'), nn_arch='VGG', batchsize=64, epoch=(11,18), use_scratch_ssd=False, loss_opt=[losses, None, loss_weights],
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0_errors')

    # mae dir, mae en, n_filters=(256, 128, 64, 64, 64, 64, 128, 128, 128, 128), mae en , not mre as specified in str_ident
    # execute_cnn(n_bins=[(11,13,18,300)], class_type=(6, 'energy_dir_bjorken-y_and_errors_dir_new_loss'), nn_arch='VGG', batchsize=64, epoch=(7,15), use_scratch_ssd=False, loss_opt=[losses, None, loss_weights],
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0_errors_mre_e_more_filters')

    # mae dir, mae en, standard filter, different loss weights
    # loss_weights = {'dir': 60, 'e': 1, 'by': 15, 'dir_err': 1, 'e_err': 1, 'by_err': 1}
    # execute_cnn(n_bins=[(11,13,18,300)], class_type=(6, 'energy_dir_bjorken-y_and_errors_dir_new_loss'), nn_arch='VGG', batchsize=64, epoch=(5,19), use_scratch_ssd=False, loss_opt=[losses, None, loss_weights],
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0_errors_dir_more_loss-w')

    # python run_cnn.py -l lists/lp/all_e/xyz-t_lp_e_1-100_t-all_train_e-CC_mu-CC.list lists/lp/all_e/xyz-t_lp_e_1-100_t-all_test_e-CC_mu-CC.list
    # python run_cnn.py -l lists/lp/all_e/xyz-t_lp_e_1-100_t-all_train_e-CC_mu-CC.list lists/lp/all_e/xyz-t_lp_e_1-100_t-all_test_e-CC_mu-CC_half.list

    ## -- Regression 1-100GeV, xyz-t 60b, train elec-cc-nc and muon-cc, with errors, without vertex -- ##
    # # standard
    # losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
    #           'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
    #           'dir_x_err': loss_uncertainty_mse, 'dir_y_err': loss_uncertainty_mse, 'dir_z_err': loss_uncertainty_mse,
    #           'e_err': loss_uncertainty_mse, 'by_err': loss_uncertainty_mse}
    # loss_weights = {'dir_x': 35, 'dir_y': 35, 'dir_z': 50, 'e': 1, 'by': 80,
    #                 'dir_x_err': 1, 'dir_y_err': 1, 'dir_z_err': 1, 'e_err': 0.0005, 'by_err': 1}
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(10, 'energy_dir_bjorken-y_errors'), nn_arch='VGG', batchsize=64, epoch=(11,3),
    #             use_scratch_ssd=False, loss_opt=(losses, None, loss_weights), n_gpu=(1, 'avolkov'), mode='eval',
    #             swap_4d_channels=None, zero_center=True, str_ident='lp_tight-1_60b_errors_mse')

    # mae errors
    # losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
    #           'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
    #           'dir_x_err': loss_uncertainty_mae, 'dir_y_err': loss_uncertainty_mae, 'dir_z_err': loss_uncertainty_mae,
    #           'e_err': loss_uncertainty_mae, 'by_err': loss_uncertainty_mae}
    # loss_weights = {'dir_x': 35, 'dir_y': 35, 'dir_z': 35, 'e': 1, 'by': 30,
    #                 'dir_x_err': 1, 'dir_y_err': 1, 'dir_z_err': 1, 'e_err': 0.07, 'by_err': 1.5}
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(10, 'energy_dir_bjorken-y_errors'), nn_arch='VGG', batchsize=64, epoch=(13,2),
    #             use_scratch_ssd=False, loss_opt=(losses, None, loss_weights), n_gpu=(1, 'avolkov'), mode='eval',
    #             swap_4d_channels=None, zero_center=True, str_ident='lp_tight-1_60b_errors_mae')

    # mse errors (faulty, actually mae errors!!), more (double) dense errors
    # losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
    #           'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
    #           'dir_x_err': loss_uncertainty_mae, 'dir_y_err': loss_uncertainty_mae, 'dir_z_err': loss_uncertainty_mae,
    #           'e_err': loss_uncertainty_mae, 'by_err': loss_uncertainty_mae}
    # loss_weights = {'dir_x': 35, 'dir_y': 35, 'dir_z': 50, 'e': 1, 'by': 65,
    #                 'dir_x_err': 1, 'dir_y_err': 1, 'dir_z_err': 1, 'e_err': 0.05, 'by_err': 1.3}
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(10, 'energy_dir_bjorken-y_errors'), nn_arch='VGG', batchsize=64, epoch=(13,3),
    #             use_scratch_ssd=False, loss_opt=(losses, None, loss_weights), n_gpu=(1, 'avolkov'), mode='eval',
    #             swap_4d_channels=None, zero_center=True, str_ident='lp_tight-1_60b_errors_mse_double_dense_err')

    # python run_cnn.py -l lists/lp/all_e/xyz-t_lp_60b_e_1-100_all_train_e-cc-nc_mu-cc.list lists/lp/all_e/xyz-t_lp_60b_e_1-100_half_test_e-cc-nc_mu-cc.list
    # python run_cnn.py -l lists/lp/all_e/xyz-t_lp_60b_e_1-100_all_train_e-cc-nc_mu-cc.list lists/lp/all_e/xyz-t_lp_60b_e_1-100_all_test_e-cc-nc_mu-cc.list

    ## -- Regression 1-100GeV, xyz-t 60b + xyz-c 31b, train elec-cc-nc and muon-cc, with errors, without vertex -- ##
    # # standard
    # losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
    #           'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
    #           'dir_x_err': loss_uncertainty_mse, 'dir_y_err': loss_uncertainty_mse, 'dir_z_err': loss_uncertainty_mse,
    #           'e_err': loss_uncertainty_mse, 'by_err': loss_uncertainty_mse}
    # loss_weights = {'dir_x': 35, 'dir_y': 35, 'dir_z': 65, 'e': 1, 'by': 65,
    #                 'dir_x_err': 1, 'dir_y_err': 1, 'dir_z_err': 1, 'e_err': 0.0005, 'by_err': 1}
    # execute_cnn(n_bins=[(11,13,18,60), (11,13,18,31)], class_type=(10, 'energy_dir_bjorken-y_errors'), nn_arch='VGG', batchsize=64, epoch=(7,1),
    #             use_scratch_ssd=False, loss_opt=(losses, None, loss_weights), n_gpu=(1, 'avolkov'), mode='eval',
    #             swap_4d_channels='xyz-t_and_xyz-c_single_input', zero_center=True, str_ident='lp_tight-1_60b_errors_mse')

    # e-NC by fix
    losses = {'dir_x': 'mean_absolute_error', 'dir_y': 'mean_absolute_error', 'dir_z': 'mean_absolute_error',
              'e': 'mean_absolute_error', 'by': 'mean_absolute_error',
              'dir_x_err': loss_uncertainty_mse, 'dir_y_err': loss_uncertainty_mse, 'dir_z_err': loss_uncertainty_mse,
              'e_err': loss_uncertainty_mse, 'by_err': loss_uncertainty_mse}
    loss_weights = {'dir_x': 35, 'dir_y': 35, 'dir_z': 65, 'e': 1, 'by': 65,
                    'dir_x_err': 1, 'dir_y_err': 1, 'dir_z_err': 1, 'e_err': 0.0005, 'by_err': 1}
    execute_cnn(n_bins=[(11,13,18,60), (11,13,18,31)], class_type=(10, 'energy_dir_bjorken-y_errors'), nn_arch='VGG', batchsize=64, epoch=(8,5),
                use_scratch_ssd=False, loss_opt=(losses, None, loss_weights), n_gpu=(1, 'avolkov'), mode='eval',
                swap_4d_channels='xyz-t_and_xyz-c_single_input', zero_center=True, str_ident='lp_tight-1_60b_errors_mse_by_fix')

    # python run_cnn.py -m lists/lp/all_e/xyz-t_xyz-c_lp_60b_e_1-100_all_train_e-cc-nc_mu-cc.list lists/lp/all_e/xyz-t_xyz-c_lp_60b_e_1-100_half_test_e-cc-nc_mu-cc.list
    # python run_cnn.py -m lists/lp/all_e/xyz-t_xyz-c_lp_60b_e_1-100_all_train_e-cc-nc_mu-cc.list lists/lp/all_e/xyz-t_xyz-c_lp_60b_e_1-100_all_test_e-cc-nc_mu-cc.list



# lp, xyz-t, yzt-x, tight-1 + tight-2
#     execute_cnn(n_bins=[(11,13,18,60), (11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=32, epoch=(0,1), use_scratch_ssd=False,
#                 n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='xyz-t_and_yzt-x', zero_center=True, str_ident='multi_input_single_train_tight-1_tight-2_lr_0.0003')
#     execute_cnn(n_bins=[(11, 13, 18, 60), (11, 13, 18, 60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=32, epoch=(1,1), use_scratch_ssd=False,
#                 n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels='xyz-t_and_yzt-x', zero_center=True, str_ident='multi_input_single_train_tight-1_tight-2_lr_0.003_tr_st_test_st')
# python run_cnn.py -m lists/lp/xyz-t_lp_tight-1_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-1_tight-2_test_no_tau.list
# python run_cnn.py -m lists/lp/xyz-t_lp_tight-1_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-1_tight-2_test_all_tau.list
# pred 1-5 GeV
# python run_cnn.py -m lists/lp/xyz-t_lp_tight-1_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-1_tight-2_pred_1_to_5_GeV.list


