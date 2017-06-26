#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code for running CNN's."""

import argparse
import sys
import keras as ks

from models.cnn_models import *
from utilities.cnn_utilities import *


def parse_input():
    """
    Parses the user input for running the CNN.
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

        #train_files = [line.rstrip('\n') for line in open(args.listfile_train_and_test[0])]
        #test_files = [line.rstrip('\n') for line in open(args.listfile_train_and_test[1])]
    else:
        train_files = [(args.train_file, h5_get_number_of_rows(args.train_file))]
        test_files = [(args.test_file, h5_get_number_of_rows(args.test_file))]

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


def execute_cnn(n_bins):
    """
    Runs a convolutional neural network.
    :param list n_bins: Declares the number of bins for each dimension (x,y,z) in the train- and testfiles.
    """

    train_files, test_files = parse_input()

    number_of_classes = 2
    batchsize = 32
    print "Batchsize = ", batchsize
    modelname = "model_3d_xyz_numuCC_vs_nueNC_epoch"

    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    #testfile, testsize = 'input/numuyztShufTail54921.csv.h5', 5000
    #trainfiles, trainsize = ['input/numuyztShufHead270k.csv.h5'], 270000

    restart_index = 0  # 4 targets, 6xhdf5, bs 32, mse, adam

    if restart_index == 0:
        model = define_3d_model_xyz(number_of_classes, [n_bins_x, n_bins_y, n_bins_z])
    else:
        model = ks.models.load_model("models/trained" + modelname + str(restart_index) + ".h5")

    model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=["mean_squared_error"])
    # model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
    # model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_absolute_error"])
    model.summary()

    printSize = 5
    i = restart_index

    #trainsize = 100000

    while 1:
        # process all hdf5 files, full epoch
        for (f, f_size) in train_files:
            i += 1
            print "Training ", i, " on file ", f
            model.fit_generator(generate_batches_from_hdf5_file(f, batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, number_of_classes),
                                steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=1, max_q_size=1)
            # store the trained model
            model.save("models/" + modelname + str(i) + ".h5")
            # delete old model?
            #if testfile != "":
             #   results = doTheEvaluation(model, number_of_classes, testfile, testsize, printSize, n_bins_x, n_bins_y, n_bins_z, n_bins_t, batchsize)


if __name__ == '__main__':
    execute_cnn(n_bins=[11,13,18,1]) # standard 4D case: n_bins=[11,13,18,50]

