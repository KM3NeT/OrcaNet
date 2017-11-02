#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import os
import time
import shutil
import sys
import argparse
import keras as ks
from keras import backend as K

from utilities.input_utilities import *
from models.short_cnn_models import *
from models.wide_resnet import *
from utilities.cnn_utilities import *
from utilities.multi_gpu.multi_gpu import *
from utilities.data_tools.shuffle_h5 import shuffle_h5
from utilities.visualization.ks_visualize_activations import *
from utilities.evaluation_utilities import *


def parallelize_model_to_n_gpus(model, n_gpu, batchsize, lr, lr_decay):
    """
    Parallelizes the nn-model to multiple gpu's.
    Currently, up to 4 GPU's at Tiny-GPU are supported.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param int n_gpu: Number of gpu's that the model should be parallelized to.
    :param int batchsize: original batchsize that should be used for training/testing the nn.
    :param float lr: learning rate of the optimizer used in training.
    :param float lr_decay: learning rate decay during training.
    :return: int batchsize, float lr: new batchsize/lr scaled by the number of used gpu's.
    """
    if n_gpu == 1:
        return model, batchsize, lr, lr_decay
    else:
        assert n_gpu > 1 and isinstance(n_gpu, int), 'You probably made a typo: n_gpu must be an int with n_gpu >= 1!'

        gpus_list = get_available_gpus(n_gpu)
        ngpus = len(gpus_list)
        print('Using GPUs: {}'.format(', '.join(gpus_list)))
        batchsize = batchsize * ngpus
        lr = lr #* ngpus #TODO not sure if this makes sense
        lr_decay = lr_decay #* ngpus

        # Data-Parallelize the model via function
        model = make_parallel(model, gpus_list, usenccl=False, initsync=True, syncopt=False, enqueue=False)
        print_mgpu_modelsummary(model)

        return model, batchsize, lr, lr_decay


def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, lr_decay, tb_logger, swap_4d_channels):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    epoch += 1
    if epoch > 1 and lr_decay > 0:
        lr *= 1 - float(lr_decay)
        K.set_value(model.optimizer.lr, lr)
        print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
              ' before epoch ' + str(epoch) + ' (minus ' + str(lr_decay) + ')'

    fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch, shuffle, swap_4d_channels, n_events=None, tb_logger=tb_logger)
    evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=None)

    return epoch, lr


def fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, n_events=None, tb_logger=False):
    """
    Trains a model based on the Keras fit_generator method.
    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param str modelname: Name of the model.
    :param list train_files: list of tuples that contains the testfiles and their number of rows (filepath, f_size).
    :param list test_files: list of tuples that contains the testfiles and their number of rows for the tb_callback.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in both the train- and test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Epoch of the model if it has been trained before.
    :param bool shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
    """
    if tb_logger is True:
        tb_callback = TensorBoardWrapper(generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, zero_center_image=xs_mean),
                                     nb_steps=int(5000 / batchsize), log_dir='models/trained/tb_logs/' + modelname + '_{}'.format(time.time()),
                                     histogram_freq=1, batch_size=batchsize, write_graph=False, write_grads=True, write_images=True)
        callbacks = [tb_callback]
        validation_data = generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, swap_col=swap_4d_channels, zero_center_image=xs_mean) #f_size=None is ok here
        validation_steps = int(5000 / batchsize)
    else:
        validation_data, validation_steps, callbacks = None, None, None

    for i, (f, f_size) in enumerate(train_files):  # process all h5 files, full epoch
        if epoch > 1 and shuffle is True: # just for convenience, we don't want to wait before the first epoch each time
            print 'Shuffling file ', f, ' before training in epoch ', epoch
            shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
        print 'Training in epoch', epoch, 'on file ', i, ',', f

        if n_events is not None: f_size = n_events  # for testing

        model.fit_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels),
            steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=1, max_queue_size=10,
            validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks)
        model.save("models/trained/trained_" + modelname + '_epoch' + str(epoch) + '.h5')


def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=None):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.
    :param ks.model.Model/Sequential model: Keras model (trained) of a neural network.
    :param list test_files: list of tuples that contains the testfiles and their number of rows.
    :param int batchsize: Batchsize that is used in the evaluate_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for evaluating.
    """
    for i, (f, f_size) in enumerate(test_files):
        print 'Testing on file ', i, ',', f

        if n_events is not None: f_size = n_events  # for testing

        evaluation = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / batchsize), max_queue_size=10)
        print 'Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')'


def execute_cnn(n_bins, class_type, nn_arch, batchsize, epoch, n_gpu=1, mode='train', swap_4d_channels=None,
                use_scratch_ssd=False, zero_center=False, shuffle=False, tb_logger=False):
    """
    Runs a convolutional neural network.
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z,t) in the train- and testfiles.
    :param (int, str) class_type: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    :param int batchsize: Batchsize that should be used for the cnn.
    :param int epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param int n_gpu: Number of gpu's that should be used. n > 1 for multi-gpu implementation.
    :param str mode: Specifies what the function should do - train & test a model or evaluate a 'finished' model?
                     Currently, there are two modes available: 'train' & 'eval'.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped.
                                      Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X
    :param bool use_scratch_ssd: Declares if the input files should be copied to the node-local SSD scratch before executing the cnn.
    :param bool zero_center: Declares if the input images ('xs') should be zero-centered before training.
    :param bool shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param bool tb_logger: Declares if a tb_callback should be used during training (takes longer to train due to overhead!).
    """
    if swap_4d_channels is not None and n_bins.count(1) != 0: raise ValueError('swap_4d_channels must be None if dim < 4.')

    train_files, test_files = parse_input(use_scratch_ssd)

    xs_mean = load_zero_center_data(train_files, batchsize, n_bins, n_gpu, swap_4d_channels=swap_4d_channels) if zero_center is True else None

    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels)

    if epoch == 0:
        if nn_arch is 'WRN': model = create_wide_residual_network(n_bins, batchsize, nb_classes=class_type[0], N=2, k=2, dropout=0.1, k_size=3)

        elif nn_arch is 'VGG': model = create_vgg_like_model(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
                                                           n_filters=(64, 64, 64, 64, 64, 128, 128, 128), swap_4d_channels=swap_4d_channels)
        else: raise ValueError('Currently, only "WRN" or "VGG" are available as nn_arch')
    else:
        model = ks.models.load_model('models/trained/trained_' + modelname + '_epoch' + str(epoch) + '.h5')

    model.summary()
    # plot model, install missing packages with conda install if it throws a module error
    ks.utils.plot_model(model, to_file='./models/model_plots/' + modelname + '.png', show_shapes=True, show_layer_names=True)

    lr = 0.00059 # 0.01 default for SGD, 0.001 for Adam
    lr_decay = 0.05 # % decay for each epoch, e.g. if 0.1 -> lr_new = lr*(1-0.1)=0.9*lr
    model, batchsize, lr, lr_decay = parallelize_model_to_n_gpus(model, n_gpu, batchsize, lr, lr_decay)

    sgd = ks.optimizers.SGD(lr=lr, momentum=0.9, decay=0, nesterov=True)
    adam = ks.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0) # epsilon=1 for deep networks, lr = 0.001 default
    if epoch == 0: model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if mode == 'train':
        while 1:
            epoch, lr = train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean,
                                             epoch, shuffle, lr, lr_decay, tb_logger, swap_4d_channels)

    if mode == 'eval':
        # After training is finished, investigate model performance
        #arr_energy_correct = make_performance_array_energy_correct(model, test_files[0][0], n_bins, class_type, batchsize, xs_mean, swap_4d_channels, samples=None)
        #np.save('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy', arr_energy_correct)

        arr_energy_correct = np.load('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy')
        make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct, title='Classification for muon-CC_and_elec-CC_3-100GeV',
                                                      filename='results/plots/PT_' + modelname) #TODO think about more automatic savenames
        make_prob_hists(arr_energy_correct[:, ], modelname=modelname)


if __name__ == '__main__':
    # TODO still need to change some stuff in execute_cnn() directly like optimizers (lr, decay, sgd/adam, ...)
    # available class_types:
    # - (2, 'muon-CC_to_elec-NC'), (1, 'muon-CC_to_elec-NC')
    # - (2, 'muon-CC_to_elec-CC'), (1, 'muon-CC_to_elec-CC')
    # - (2, 'up_down'), (1, 'up_down')
    execute_cnn(n_bins=(11,13,18,50), class_type=(2, 'muon-CC_to_elec-CC'), nn_arch='VGG', batchsize=32, epoch=26,
                n_gpu=1, mode='eval', swap_4d_channels=None, zero_center=True, tb_logger=False) # standard 4D case: n_bins=[11,13,18,50]

# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/yzt/concatenated/train_muon-CC_and_elec-CC_each_240_yzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/yzt/concatenated/test_muon-CC_and_elec-CC_each_60_yzt_shuffled.h5
# python run_cnn.py --list /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_files.list /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_files.list placeholder placeholder

# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt/concatenated/train_muon-CC_and_elec-CC_each_480_xyzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/h5/xyzt/concatenated/test_muon-CC_and_elec-CC_each_120_xyzt_shuffled.h5

# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz/concatenated/train_muon-CC_and_elec-CC_10-100GeV_each_480_yz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz/concatenated/test_muon-CC_and_elec-CC_10-100GeV_each_120_yz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt/concatenated/train_muon-CC_10-100GeV_each_480_zt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt/concatenated/test_muon-CC_10-100GeV_each_120_zt_shuffled.h5