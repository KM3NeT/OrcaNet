#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD."""

import os
import shutil
import sys
import argparse
import h5py


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

    parser.add_argument('train_file', metavar='train_file', type=str, nargs='?', help='the filepath of the traindata file.')
    parser.add_argument('test_file', metavar='test_file', type=str, nargs='?', help='the filepath of the testdata file.')
    parser.add_argument('-l', '--list', dest='listfile_train_and_test', type=str, nargs=2,
                        help='filepath of a .list file that contains all .h5 files that should be used for training/testing')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.listfile_train_and_test:
        train_files = []
        test_files = []

        for line in open(args.listfile_train_and_test[0]):
            line = line.rstrip('\n')
            train_files.append((line, h5_get_number_of_rows(line)))

        for line in open(args.listfile_train_and_test[1]):
            line = line.rstrip('\n')
            test_files.append((line, h5_get_number_of_rows(line)))

    else:
        train_files = [(args.train_file, h5_get_number_of_rows(args.train_file))]
        test_files = [(args.test_file, h5_get_number_of_rows(args.test_file))]

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