#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import os
import sys
import argparse
import shutil
import keras as ks

from models.short_cnn_models import *
from utilities.cnn_utilities import *
from utilities.shuffle_h5 import shuffle_h5


def parse_input(use_scratch_ssd=True):
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


def execute_cnn(n_bins, class_type, batchsize = 32, epoch = 0, use_scratch_ssd=False):
    """
    Runs a convolutional neural network.
    :param list n_bins: Declares the number of bins for each dimension (x,y,z) in the train- and testfiles.
    :param (int, str) class_type: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for the cnn.
    :param int epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param bool use_scratch_ssd: Declares if the input files should be copied to the node-local SSD scratch before executing the cnn.
    """
    # TODO list all available class types here (num_classes, class_name)
    train_files, test_files = parse_input(use_scratch_ssd)

    print 'Batchsize = ', batchsize
    modelname = 'model_3d_xzt_' + class_type[1]

    if epoch == 0:
        model = define_3d_model_xyz_test(class_type[0], n_bins)
    else:
        model = ks.models.load_model('models/trained/trained_' + modelname + str(epoch) + '.h5')

    model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mean_squared_error'])
    #model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    #printSize = 5

    while 1:
        epoch +=1
        i = 0
        # process all h5 files, full epoch
        for (f, f_size) in train_files:
            i += 1
            #if epoch > 1: # just for convenience, we don't want to wait before the first epoch each time
             #   shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
            print 'Training in epoch', epoch, 'on file ', i, ',', f
            f_size = 70000 # for testing
            model.fit_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type),
                                steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=1)
            # store the trained model
            #model.save("models/trained/trained_" + modelname + '_f' + str(i) + '_epoch' + str(epoch) + '.h5')
            # delete old model?

        for (f, f_size) in test_files:
            # probably one test file is enough
            f_size = 70000 # for testing
            evaluation = model.evaluate_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type),
                                                  steps=int(f_size / batchsize))
            print evaluation
            print model.metrics_names

            # if testfile != "":
            #    results = doTheEvaluation(model, number_of_classes, testfile, testsize, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, batchsize)


if __name__ == '__main__':
    # TODO still need to change some stuff in execute_cnn() directly like modelname and optimizers
    execute_cnn(n_bins=[11,13,18,1], class_type = (2, 'muon-CC_to_elec-NC'),
                batchsize = 32, epoch= 0, use_scratch_ssd=False) # standard 4D case: n_bins=[11,13,18,50]

# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/train_muon-CC_and_elec-NC_each_480_xyz_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/test_muon-CC_and_elec-NC_each_120_xyz_shuffled.h5
# python run_cnn.py /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/train_muon-CC_and_elec-CC_each_480_xzt_shuffled.h5 /home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_120_xzt_shuffled.h5
