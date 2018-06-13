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
from utilities.losses import custom_metric_mean_relative_error_5_labels


def parallelize_model_to_n_gpus(model, n_gpu, batchsize):
    """
    Parallelizes the nn-model to multiple gpu's.
    Currently, up to 4 GPU's at Tiny-GPU are supported.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param int batchsize: original batchsize that should be used for training/testing the nn.
    :return: int batchsize, float lr: new batchsize/lr scaled by the number of used gpu's.
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

            return model, batchsize

    else:
        raise ValueError('Currently, no multi_gpu mode other than "avolkov" is available.')


def schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=0.001, manual_mode=(False, None, 0.0, None)):
    """
    Function that schedules a learning rate during training.
    If manual_mode[0] is False, the current lr will be automatically calculated if the training is resumed, based on the epoch variable.
    If manual_mode[0] is True, the final lr during the last training session (manual_mode[1]) and the lr_decay (manual_mode[1])
    have to be set manually.
    :param Model model: Keras nn model instance. Used for setting the lr.
    :param (int, int) epoch: The epoch number and the file number at which this training session is resumed (last finished epoch).
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param list train_files: list of tuples that contains the trainfiles and their number of rows (filepath, f_size).
    :param float lr_initial: Initial lr that is used with the automatic mode. Typically 0.01 for SGD and 0.001 for Adam.
    :param (bool, None/float, float, None/float) manual_mode: Tuple that controls the options for the manual mode.
            manual_mode[0] = flag to enable the manual mode, manual_mode[1] = lr value, of which the mode should start off
            manual_mode[2] = lr_decay during epochs, manual_mode[3] = current lr, only used to check if this is the first instance of the while loop
    :return (int, int) epoch: The epoch number of the new epoch (+= 1) and the filenumber.
    :return float lr: Learning rate that has been set for the model and for this epoch.
    :return float lr_decay: Learning rate decay that has been used to decay the lr rate used for this epoch.
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
            print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch[0]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'

    else:
        if epoch[0] == 1 and epoch[1] == 1:
            lr, lr_decay = lr_initial, 0.00
            #lr, lr_decay = lr_initial * n_gpu[0], 0.00
            K.set_value(model.optimizer.lr, lr)
            print 'Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch[0]) + \
                  ' and file ' + str(epoch[1])
        else:
            n_train_files = len(train_files)
            lr, lr_decay = get_new_learning_rate(epoch, lr_initial, n_train_files, n_gpu[0])
            K.set_value(model.optimizer.lr, lr)
            print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch[0]) + ' and file ' + str(epoch[1]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'

    return epoch, lr, lr_decay


def get_new_learning_rate(epoch, lr_initial, n_train_files, n_gpu):
    """
    Function that calculates the current learning rate based on the number of already trained epochs.
    Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                          lr_decay = 4% for 0.0003 >= lr > 0.0001
                                          lr_decay = 2% for 0.0001 >= lr
    :param (int, int) epoch: The number of the current epoch and the current filenumber which is used to calculate the new learning rate.
    :param float lr_initial: Initial lr for the first epoch. Typically 0.01 for SGD and 0.001 for Adam.
    :param int n_train_files: specifies to how many files the training dataset is split.
    :param int n_gpu: number of gpu's that are used during the training. Used for scaling the lr.
    :return float lr_temp: Calculated learning rate for this epoch.
    :return float lr_decay: Latest learning rate decay used.
    """
    n_epoch, n_file = epoch[0], epoch[1]
    n_lr_decays = (n_epoch - 1) * n_train_files + (n_file - 1)

    lr_temp = lr_initial # * n_gpu TODO think about multi gpu lr
    lr_decay = None

    for i in xrange(n_lr_decays):

        if lr_temp > 0.0003:
            lr_decay = 0.07
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04
        else:
            lr_decay = 0.02

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay


def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, tb_logger, swap_4d_channels, str_ident, n_gpu):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=0.003, manual_mode=(False, 0.0003, 0.07, lr)) # begin new training step
    train_iter_step = 0
    for file_no, (f, f_size) in enumerate(train_files, 1):

        if file_no < epoch[1]:
            continue # skip if this file for this epoch has already been used for training

        train_iter_step += 1

        if train_iter_step > 1: epoch, lr, lr_decay = schedule_learning_rate(model, epoch, n_gpu, train_files, lr_initial=0.003, manual_mode=(False, 0.0003, 0.07, lr))

        history_train = fit_model(model, modelname, train_files, f, f_size, file_no, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                                            shuffle, swap_4d_channels, str_ident, n_events=None, tb_logger=tb_logger)
        history_test = evaluate_model(model, modelname, test_files, train_files, batchsize, n_bins, class_type, xs_mean, epoch, swap_4d_channels, str_ident, n_events=None)

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
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param str modelname: Name of the model.
    :param list train_files: list of tuples that contains the trainfiles and their number of rows (filepath, f_size).
    :param str f: full filepath of the file that should be used for training.
    :param int f_size: number of images contained in f.
    :param int file_no: if the full data is split into multiple files, this param indicates the file number.
    :param list test_files: list of tuples that contains the testfiles and their number of rows for the tb_callback.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param (int, int) epoch: Epoch of the model if it has been trained before.
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str str_ident: string identifier for the projection type / model input that is parsed to the image generator. Needed for some specific models.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
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
        print 'Shuffling file ', f, ' before training in epoch ', epoch[0], ' and file ', file_no
        shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)

    if shuffle[1] is not None:
        n_preshuffled = shuffle[1]
        f = f.replace('0.h5', str(epoch[0]-1) + '.h5') if epoch[0] <= n_preshuffled else f.replace('0.h5', str(np.random.randint(0, n_preshuffled+1)) + '.h5')

    print 'Training in epoch', epoch[0], 'on file ', file_no, ',', f

    history = model.fit_generator(
        generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, str_ident, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels),
        steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=1, max_queue_size=10,
        validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)
    model.save("models/trained/trained_" + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(file_no) + '.h5')

    return history


def evaluate_model(model, modelname, test_files, train_files, batchsize, n_bins, class_type, xs_mean, epoch, swap_4d_channels, str_ident, n_events=None):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.
    :param ks.model.Model/Sequential model: Keras model (trained) of a neural network.
    :param str modelname: Name of the model.
    :param list test_files: list of tuples that contains the testfiles and their number of rows.
    :param list train_files: list of tuples that contains the trainfiles and their number of rows (filepath, f_size).
    :param int batchsize: Batchsize that is used in the evaluate_generator method.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) in the test_files. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param (int, int) epoch: Current epoch of the training.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str str_ident: string identifier for the projection type / model input that is parsed to the image generator. Needed for some specific models.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for evaluating.
    """
    histories = []
    for i, (f, f_size) in enumerate(test_files):
        print 'Testing on file ', i, ',', str(f)

        if n_events is not None: f_size = n_events  # for testing

        history = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, str_ident, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / batchsize), max_queue_size=10, verbose=1)
        print 'Test sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')'
        histories.append(history)

    history_averaged = [sum(e)/float(len(e)) for e in histories] if len(histories) > 1 else histories[0] # average over all test files if necessary

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
    for i in xrange(len(model.metrics_names)):
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
        f_out.write('Tested in epoch ' + str(epoch) + ', file ' + str(epoch[1]) + ' on test_files ' + str(test_files) + '\n')
        f_out.write('History for training and testing: \n')
        f_out.write('Train: ' + str(history_train.history) + '\n')
        f_out.write('Test: ' + str(history_test) + ' (' + str(model.metrics_names) + ')' + '\n')
        f_out.write('\n')
        f_out.write('Additional Info:\n')
        f_out.write('Batchsize=' + str(batchsize) + ', n_bins=' + str(n_bins) +
                    ', class_type=' + str(class_type) + '\n' +
                    'swap_4d_channels=' + str(swap_4d_channels) + ', str_ident=' + str_ident + '\n')
        f_out.write('\n')


def execute_cnn(n_bins, class_type, nn_arch, batchsize, epoch, n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None,
                use_scratch_ssd=False, zero_center=False, shuffle=(False,None), tb_logger=False, str_ident='', loss_opt=('categorical_crossentropy', 'accuracy')):
    """
    Runs a convolutional neural network.
    :param list(tuple) n_bins: Declares the number of bins for each dimension (x,y,z,t) in the train- and testfiles. Can contain multiple n_bins tuples.
                               Multiple n_bins tuples are currently only used for multi-input models with multiple input files per batch.
    :param (int, str) class_type: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'track-shower')
    :param str nn_arch: Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    :param int batchsize: Batchsize that should be used for the cnn.
    :param (int, int) epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param str mode: Specifies what the function should do - train & test a model or evaluate a 'finished' model?
                     Currently, there are two modes available: 'train' & 'eval'.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped.
                                      Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X
    :param bool use_scratch_ssd: Declares if the input files should be copied to the node-local SSD scratch before executing the cnn.
    :param bool zero_center: Declares if the input images ('xs') should be zero-centered before training.
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param bool tb_logger: Declares if a tb_callback should be used during training (takes longer to train due to overhead!).
    :param str str_ident: Optional str identifier that gets appended to the modelname. Useful when training models which would have the same modelname.
                          Also used for defining models and projections!
    :param (str, str) loss_opt: tuple that contains 1) the loss and 2) the metric during training.
    """
    train_files, test_files, multiple_inputs = parse_input()
    xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu[0]) if zero_center is True else None # TODO if ssd is true, still loads from woody!
    if use_scratch_ssd is True: train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs) # TODO check if files already on ssd
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)

    if epoch[0] == 0:
        if nn_arch is 'WRN': model = create_wide_residual_network(n_bins[0], batchsize, nb_classes=class_type[0], n=1, k=1, dropout=0.2, k_size=3, swap_4d_channels=swap_4d_channels)

        elif nn_arch is 'VGG':
            input_modes = {'xyz-t_and_yzt-x', 'yzt-x_all-t_and_yzt-x_tight-1-t', 'xyz-t-tight-1-w-geo-fix_and_yzt-x-tight-1-wout-geo-fix'}
            if swap_4d_channels in input_modes and 'multi_input_single_train' not in str_ident:
                model = create_vgg_like_model_double_input(n_bins, batchsize, nb_classes=class_type[0], dropout=0.2,
                                                               n_filters=(64, 64, 64, 64, 64, 128, 128, 128), swap_4d_channels=swap_4d_channels) # TODO not working for multiple input files
            elif 'multi_input_single_train' in str_ident:
                model = create_vgg_like_model_multi_input_from_single_nns(n_bins, batchsize, str_ident, nb_classes=class_type[0], dropout=(0,0.2), swap_4d_channels=swap_4d_channels)

            else:
                model = create_vgg_like_model(n_bins, batchsize, class_type, dropout=0.1,
                                              #n_filters=(64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128), swap_4d_channels=swap_4d_channels) # 4 more layers
                                              n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), swap_4d_channels=swap_4d_channels) # 2 more layers

        elif nn_arch is 'Conv_LSTM':
            model = create_convolutional_lstm(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
                                              n_filters=(32, 32, 64, 64, 64, 64, 128, 128))

        else: raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')
    else:
        model = ks.models.load_model('models/trained/trained_' + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5',
                                     custom_objects={'custom_metric_mean_relative_error_5_labels': custom_metric_mean_relative_error_5_labels})

    # plot model, install missing packages with conda install if it throws a module error
    ks.utils.plot_model(model, to_file='./models/model_plots/' + modelname + '.png', show_shapes=True, show_layer_names=True)

    sgd = ks.optimizers.SGD(momentum=0.9, decay=0, nesterov=True)
    adam = ks.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0) # epsilon=1 for deep networks
    optimizer = adam # Choose optimizer, only used if epoch == 0

    # if epoch[0] == 1 and 'double_input_single_train' in str_ident:
    #     model = change_dropout_rate_for_double_input_model(n_bins, batchsize, model, dropout=(0.1, 0.1), trainable=(True, True), swap_4d_channels=swap_4d_channels)
    #     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize) #TODO compile after restart????
    metrics = [loss_opt[1]] #if not class_type[1] == 'energy_and_direction_and_bjorken-y' else [loss_opt[1], custom_metric_mean_absolute_error]

    if n_gpu[0] > 1: model.compile(loss=loss_opt[0], optimizer=optimizer, metrics=metrics) # TODO check

    model.summary()

    if epoch[0] == 0: model.compile(loss=loss_opt[0], optimizer=optimizer, metrics=metrics)

    if mode == 'train':
        lr = None
        while 1:
            epoch, lr = train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean,
                                 epoch, shuffle, lr, tb_logger, swap_4d_channels, str_ident, n_gpu)

    if mode == 'eval':

        # After training is finished, investigate model performance
        # arr_nn_pred = get_nn_predictions_and_mc_info(model, test_files, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, str_ident, modelname, samples=None)
        # np.save('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy', arr_nn_pred)

        arr_nn_pred = np.load('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy')

        if class_type[1] == 'track-shower': # categorical
            make_energy_to_accuracy_plot_multiple_classes(arr_nn_pred, title='Classified as track',
                                                      filename='results/plots/PT_' + modelname, compare_pheid=True, corr_cut_pred_0=0.4) #TODO think about more automatic savenames
            # TODO fix code below for more classes
            make_prob_hists(arr_nn_pred[:, ], modelname=modelname, compare_pheid=True)
            make_hist_2d_property_vs_property(arr_nn_pred, modelname, property_types=('bjorken-y', 'probability'), e_cut=(3, 100), compare_pheid=True)
            calculate_and_plot_separation_pid(arr_nn_pred, modelname, compare_pheid=True)

        else: # regression
            arr_nn_pred_shallow = np.load('/home/woody/capn/mppi033h/Data/various/arr_nn_pred.npy')

            if 'energy' in class_type[1]:
                # DL
                make_2d_energy_resolution_plot(arr_nn_pred, modelname, compare_pheid=(True, '3-100_GeV_prod_energy_comparison'), correct_energy=(True, 1))
                make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median_relative', energy_bins=np.linspace(3,100,32),
                                                          compare_pheid=(True, '3-100_GeV_prod_energy_comparison'), correct_energy=(False, 1))
                make_1d_energy_std_div_e_true_plot(arr_nn_pred, modelname, precuts=(True, '3-100_GeV_prod_energy_comparison'), compare_shallow=True)

                # shallow reco
                make_2d_energy_resolution_plot(arr_nn_pred_shallow, 'shallow_reco', compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))
                make_1d_energy_reco_metric_vs_energy_plot(arr_nn_pred_shallow, 'shallow_reco', metric='median_relative',  energy_bins=np.linspace(3,100,32), compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))
                make_1d_energy_std_div_e_true_plot(arr_nn_pred_shallow, 'shallow_reco', precuts=(True, '3-100_GeV_prod_energy_comparison'))
                import sys
                sys.exit()
            if 'direction' in class_type[1]:
                # DL
                make_1d_dir_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', energy_bins=np.linspace(3, 100, 32), compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))
                make_2d_dir_correlation_plot(arr_nn_pred, modelname, dir_bins=np.linspace(-1, 1, 100), compare_pheid=(False, '3-100_GeV_prod'))
                # shallow reco
                make_1d_dir_metric_vs_energy_plot(arr_nn_pred_shallow, 'shallow_reco', metric='median', energy_bins=np.linspace(3, 100, 32), compare_pheid=(True, '3-100_GeV_prod_energy_comparison'))
                make_2d_dir_correlation_plot(arr_nn_pred_shallow, 'shallow_reco', dir_bins=np.linspace(-1, 1, 100), compare_pheid=(False, '3-100_GeV_prod'))

            if 'bjorken-y' in class_type[1]:
                # DL
                make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred, modelname, metric='median', energy_bins=np.linspace(3, 100, 32), compare_pheid=(False, '3-100_GeV_prod'))
                # shallow reco
                make_1d_bjorken_y_metric_vs_energy_plot(arr_nn_pred_shallow, 'shallow_reco', metric='median', energy_bins=np.linspace(3, 100, 32), compare_pheid=(False, '3-100_GeV_prod'))


if __name__ == '__main__':
    # available class_types:
    # - (2, 'track-shower')
    # - (5, 'energy_and_direction_and_bjorken-y')

    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='Conv_LSTM', batchsize=16, epoch=(2,1), use_scratch_ssd=True,
    #             n_gpu=(4, 'avolkov'), mode='train', swap_4d_channels='conv_lstm', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='2nd_try_tight-1_w-geo-fix')

    ###############
    #--- YZT-X ---#
    ###############

    ### Larger Production
    # tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(4,1), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1')

    # tight-1, pad same
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(2,3), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1_padsame')

    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_no_tau.list lists/lp/xyz-t_lp_tight-1_test_no_tau.list

    # tight-2, padsame
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(2,3), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels='yzt-x', zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-2_bs64_dp0.1_padsame')
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

    # standard tight-1 as above, but padding same! // 2 additional layers: n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}, bs 64, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(5,4), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1_pad_same')

    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_no_tau.list lists/lp/xyz-t_lp_tight-1_test_no_tau.list

    # standard tight-2, padsame// 2 additional layers: n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128), max_pool_sizes = {5: (2, 2, 2), 9: (2, 2, 2)}, bs 64, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'track-shower'), nn_arch='VGG', batchsize=64, epoch=(6,1), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-2_bs64_dp0.1_padsame')
    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-2_train_no_tau.list lists/lp/xyz-t_lp_tight-2_test_no_tau.list

    ######## REGRESSION, Larger Production
    # e+dir+by dp 0.1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(5, 'energy_and_direction_and_bjorken-y'), nn_arch='VGG', batchsize=64, epoch=(14,4), use_scratch_ssd=False, loss_opt=('mean_absolute_error', 'mean_squared_error'),
    #             n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.1')

    # e+dir+by dp 0.0
    execute_cnn(n_bins=[(11,13,18,60)], class_type=(5, 'energy_and_direction_and_bjorken-y'), nn_arch='VGG', batchsize=64, epoch=(15,1), use_scratch_ssd=False, loss_opt=('mean_absolute_error', 'mean_squared_error'),
                n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0')

    # energy only, this time with dp 0.0 for real!
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(1, 'energy'), nn_arch='VGG', batchsize=64, epoch=(4,2), use_scratch_ssd=True, loss_opt=('mean_absolute_error', 'mean_squared_error'),
    #            n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0_for_real')

    # energy only, mre
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(1, 'energy'), nn_arch='VGG', batchsize=64, epoch=(4,2), use_scratch_ssd=True, loss_opt=(custom_metric_mean_relative_error_5_labels, 'mean_absolute_error'),
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='lp_tight-1_bs64_dp0.0_mre')

    # python run_cnn.py -l lists/lp/xyz-t_lp_tight-1_train_muon-CC_and_elec-CC_even_split.list lists/lp/xyz-t_lp_tight-1_test_muon-CC_and_elec-CC_even_split.list


    #####------------- DENSER DETECTOR STUDY
    # xyz-t, n_samples = 880000 -> factor 0.26 less -> 1/4, bs 64, dp0.1/0.2, initial lr = 0.003, tight-1
    # execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='VGG', batchsize=64, epoch=(25,1), use_scratch_ssd=True,
    #             n_gpu=(1, 'avolkov'), mode='train', swap_4d_channels=None, zero_center=True, tb_logger=False, shuffle=(False, None), str_ident='xyz-t_tight-1_w-geo-fix_bs64_dp0.1_2.640mio')



# tight_1, with geo fix Stefan, 09.02.18
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-250+500_geo-fix_60b/concatenated/without_cuts_all_events/elec-CC_and_muon-CC_xyzt_train_1_to_480_shuffled_0.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-250+500_geo-fix_60b/concatenated/without_cuts_all_events/elec-CC_and_muon-CC_xyzt_test_481_to_600_shuffled_0.h5

# tight_2, with geo fix, 23.02.18
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-150+200-tight-2_w_geo_fix/concatenated/without_cuts_all_events/elec-CC_and_muon-CC_xyzt_train_1_to_480_shuffled_0.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-150+200-tight-2_w_geo_fix/concatenated/without_cuts_all_events/elec-CC_and_muon-CC_xyzt_test_481_to_600_shuffled_0.h5

# xyz-t-tight-1-w-geo-fix_and_yzt-x-tight-1-w-geo-fix, double input, 2 more layers each, later pooling
#     execute_cnn(n_bins=[(11,13,18,60)], class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='VGG', batchsize=32, epoch=(1,1), use_scratch_ssd=False,
#                 n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels='xyz-t_and_yzt-x', zero_center=True, str_ident='multi_input_single_train_tight-1_more-layers-both_lr0.0003_dense64-16')
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-250+500_geo-fix_60b/concatenated/elec-CC_and_muon-CC_xyzt_train_1_to_480_shuffled_0.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/time_-250+500_geo-fix_60b/concatenated/elec-CC_and_muon-CC_xyzt_test_481_to_600_shuffled_0.h5

# python run_cnn.py -m lists/xyz-t-tight-1-w-geo-fix_and_xyz-t-tight-2-w-geo-fix.list

# xyz-t-tight-1 + tight-2 and yzt-x-tight-1 + tight-2, quadruple input
#     execute_cnn(n_bins=[(11,13,18,60), (11,13,18,60)], class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='VGG', batchsize=32, epoch=(1,1), use_scratch_ssd=False,
#                 n_gpu=(1, 'avolkov'), mode='eval', swap_4d_channels='xyz-t_and_yzt-x', zero_center=True, str_ident='multi_input_single_train_tight-1_tight-2_2-more-layers-everywhere')
# python run_cnn.py -m lists/xyz-t_and_yzt-x_tight-1-and-tight-2.list
# python run_cnn.py -m lists/xyz-t_and_yzt-x_tight-1-and-tight-2_with-precuts.list
# pred
# python run_cnn.py -m lists/xyz-t_and_yzt-x_tight-1-and-tight-2_pred_ORCA_1-5GeV.list
# python run_cnn.py -m lists/xyz-t_and_yzt-x_tight-1-and-tight-2_pred_ORCA_3-100GeV.list
