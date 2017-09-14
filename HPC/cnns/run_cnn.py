#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import os
import shutil
import sys
import argparse
import keras as ks
from keras import backend as K

from models.short_cnn_models import *
from models.wide_resnet import *
from utilities.cnn_utilities import *
from utilities.multi_gpu.multi_gpu import *
from utilities.data_tools.shuffle_h5 import shuffle_h5
from utilities.visualization.ks_visualize_activations import *


def parse_input(use_scratch_ssd):
    """
    Parses the user input for running the CNN.
    :param bool use_scratch_ssd: specifies if the input files should be copied to the node-local SSD scratch space.
    :return: list((train_filepath, train_filesize)) train_files: list of tuples that contains the trainfiles and their number of rows.
    :return: list((test_filepath, test_filesize)) test_files: list of tuples that contains the testfiles and their number of rows.
    """
    parser = argparse.ArgumentParser(description='E.g. < python run_cnn.py train_filepath test_filepath [...] > \n'
                                                 'Script that runs a CNN. \n'
                                                 'The input arguments are either single files for train- and testdata or \n'
                                                 'a .list file that contains the filepaths of the train-/testdata.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('train_file', metavar='train_file', type=str, nargs=1, help='the filepath of the traindata file.')
    parser.add_argument('test_file', metavar='test_file', type=str, nargs=1, help='the filepath of the testdata file.')
    parser.add_argument('-l', '--list', dest='listfile_train_and_test', type=str, nargs=2,
                        help='filepath of a .list file that contains all .h5 files that should be concatenated')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.listfile_train_and_test:
        train_files = []
        test_files = []

        for line in open(args.listfile_train_and_test[0]):
            line.rstrip('\n')
            train_files.append((line, h5_get_number_of_rows(line)))

        for line in open(args.listfile_train_and_test[1]):
            line.rstrip('\n')
            test_files.append((line, h5_get_number_of_rows(line)))

    else:
        train_files = [(args.train_file[0], h5_get_number_of_rows(args.train_file[0]))]
        test_files = [(args.test_file[0], h5_get_number_of_rows(args.test_file[0]))]

    if use_scratch_ssd is True:
        train_files, test_files = use_node_local_ssd_for_input(train_files, test_files)

    return train_files, test_files


def h5_get_number_of_rows(h5_filepath):
    """
    Gets the total number of rows of the first dataset of a .h5 file. Hence, all datasets should have the same number of rows!
    :param string h5_filepath: filepath of the .h5 file.
    :return: int number_of_rows: number of rows of the .h5 file in the first dataset.
    """
    f = h5py.File(h5_filepath, 'r')
    number_of_rows = f[f.keys()[0]].shape[0]
    f.close()

    return number_of_rows


def use_node_local_ssd_for_input(train_files, test_files):
    """
    Copies the test and train files to the node-local ssd scratch folder and returns the new filepaths of the train and test data.
    Speeds up I/O and reduces RRZE network load.
    :param list train_files: list that contains all train files in tuples (filepath, f_size).
    :param list test_files: list that contains all test files in tuples (filepath, f_size).
    :return: list train_files_ssd, test_files_ssd: new train/test list with updated SSD /scratch filepaths.
    """
    local_scratch_path = os.environ['TMPDIR']
    train_files_ssd = []
    test_files_ssd = []

    print 'Copying the input train/test data to the node-local SSD scratch folder'
    for file_tuple in train_files:
        input_filepath, f_size = file_tuple[0], file_tuple[1]

        shutil.copy2(input_filepath, local_scratch_path) # copy to /scratch node-local SSD
        input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
        train_files_ssd.append((input_filepath_ssd, f_size))

    for file_tuple in test_files:
        input_filepath, f_size = file_tuple[0], file_tuple[1]

        shutil.copy2(input_filepath, local_scratch_path) # copy to /scratch node-local SSD
        input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
        test_files_ssd.append((input_filepath_ssd, f_size))

    print 'Finished copying the input train/test data to the node-local SSD scratch folder'
    return train_files_ssd, test_files_ssd


def parallelize_model_to_n_gpus(model, n_gpu, batchsize, lr, lr_decay):
    """
    Parallelizes the nn-model to multiple gpu's.
    Currently, up to 4 GPU's at Tiny-GPU are supported.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param int n_gpu: Number of gpu's that the model should be parallelized to.
    :param int batchsize: original batchsize that should be used for training/testing the nn.
    :param float lr: learning rate of the optimizer used in training.
    :return: int batchsize, float lr: new batchsize/lr scaled by the number of used gpu's.
    """
    if n_gpu == 1:
        return model, batchsize, lr

    else:
        assert n_gpu > 1 and isinstance(n_gpu, int), 'You probably made a typo: n_gpu must be an int with n_gpu >= 1!'

        gpus_list = get_available_gpus(n_gpu)
        ngpus = len(gpus_list)
        print('Using GPUs: {}'.format(', '.join(gpus_list)))
        batchsize = batchsize * ngpus
        lr = lr * ngpus
        lr_decay = lr_decay * ngpus

        # Data-Parallelize the model via function
        model = make_parallel(model, gpus_list, usenccl=False, initsync=True, syncopt=False, enqueue=False)
        print_mgpu_modelsummary(model)

        return model, batchsize, lr, lr_decay


def execute_cnn(n_bins, class_type, batchsize = 32, epoch = 0, n_gpu=1, use_scratch_ssd=False, zero_center=True, shuffle=False):
    """
    Runs a convolutional neural network.
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z,t) in the train- and testfiles.
    :param (int, str) class_type: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for the cnn.
    :param int epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param int n_gpu: Number of gpu's that should be used. n > 1 for multi-gpu implementation.
    :param bool use_scratch_ssd: Declares if the input files should be copied to the node-local SSD scratch before executing the cnn.
    :param bool zero_center: Declares if the input images ('xs') should be zero-centered before training.
    :param bool shuffle: Declares if the training data should be shuffled before the next training epoch.
    """
    train_files, test_files = parse_input(use_scratch_ssd)

    xs_mean = load_zero_center_data(train_files, batchsize, n_bins) if zero_center is True else None

    modelname = get_modelname(n_bins, class_type)

    if epoch == 0:
        #model = define_3d_model_xyz(class_type[0], n_bins)
        #model = define_3d_model_xzt(class_type[0], n_bins)
        #model = define_2d_model_yz_test_batch_norm(class_type[0], n_bins)
        #model = define_2d_model_zt_test_batch_norm(class_type[0], n_bins)
        model = model_wide_residual_network(n_bins, batchsize, nb_classes=class_type[0], N=2, k=4, dropout=0, k_size=3)

    else:
        model = ks.models.load_model('models/trained/trained_' + modelname + str(epoch) + '.h5')

    model.summary()
    # plot model, install missing packages with conda install
    ks.utils.plot_model(model, to_file='./models/model_plots/' + modelname + '.png', show_shapes=True, show_layer_names=True)
    # visualize activations TODO make it work
    #xs = load_image_from_h5_file(train_files[0][0])
    #activations = get_activations(model, xs, print_shape_only=False, layer_name=None)
    #display_activations(activations)

    lr = 0.001 # 0.01 default for SGD, 0.001 for Adam
    lr_decay = 5e-5
    model, batchsize, lr, lr_decay = parallelize_model_to_n_gpus(model, n_gpu, batchsize, lr, lr_decay)

    sgd = ks.optimizers.SGD(lr=lr, momentum=0.9, decay=0, nesterov=True)
    adam = ks.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # epsilon=1 for deep networks, lr = 0.001 default
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])

    while 1:
        epoch +=1

        if epoch > 1 and lr_decay > 0:
            lr -= lr_decay
            K.set_value(model.optimizer.lr, lr)
            print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch) + ' (minus ' + str(lr_decay) + ')'

        for i, (f, f_size) in enumerate(train_files): # process all h5 files, full epoch
            if epoch > 1 and shuffle is True: # just for convenience, we don't want to wait before the first epoch each time
                print 'Shuffling file ', f, ' before training in epoch ', epoch
                shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
            print 'Training in epoch', epoch, 'on file ', i, ',', f
            #f_size = 500000 # for testing
            model.fit_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean),
                                steps_per_epoch=int(f_size / batchsize)-1, epochs=1, verbose=1, max_queue_size=10)
            model.save("models/trained/trained_" + modelname + '_f' + str(i) + '_epoch' + str(epoch) + '.h5')

        for i, (f, f_size) in enumerate(test_files):
            print 'Testing on file ', i, ',', f
            f_size = 50000 # for testing
            evaluation = model.evaluate_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean),
                                                  steps=int(f_size / batchsize)-1, max_queue_size=10)
            print 'Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')'

            #f_size = 50
            #predictions = model.predict_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean),
            #                                    steps=int(f_size / batchsize) - 1, max_queue_size=1)
            #print predictions
            #print predictions.argmax(axis=-1)


if __name__ == '__main__':
    # TODO still need to change some stuff in execute_cnn() directly like optimizers (lr, decay, sgd/adam, ...)
    # available class_types:
    # - (2, 'muon-CC_to_elec-NC'), (1, 'muon-CC_to_elec-NC')
    # - (2, 'muon-CC_to_elec-CC'), (1, 'muon-CC_to_elec-CC')
    # - (2, 'up_down'), (1, 'up_down')
    execute_cnn(n_bins=(1,1,18,50), class_type = (2, 'up_down'), batchsize = 32, epoch= 0,
                n_gpu=4, use_scratch_ssd=False, zero_center=True, shuffle=True) # standard 4D case: n_bins=[11,13,18,50]

# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/train_muon-CC_and_elec-NC_each_480_xyz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/test_muon-CC_and_elec-NC_each_120_xyz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_muon-CC_and_elec-CC_each_480_xzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_120_xzt_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz/concatenated/train_muon-CC_and_elec-CC_10-100GeV_each_480_yz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/yz/concatenated/test_muon-CC_and_elec-CC_10-100GeV_each_120_yz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt/concatenated/train_muon-CC_10-100GeV_each_480_zt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_10-100GeV/4dTo2d/h5/zt/concatenated/test_muon-CC_10-100GeV_each_120_zt_shuffled.h5