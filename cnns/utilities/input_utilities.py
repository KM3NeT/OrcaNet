#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD."""

import os
import glob
import shutil
import sys
import argparse
import h5py


def parse_input():
    """
    Parses the user input for running the CNN.

    There are three available input modes:
    1) Parse the train/test filepaths directly, if you only have a single file for training/testing
    2) Parse a .list file with arg -l that contains the paths to all train/test files, if the whole dataset is split over multiple files
    3) Parse a .list file with arg -m, if you need multiple input files for a single (!) batch during training.
       This is needed, if e.g. the images for a double input model are coming from different .h5 files.
       An example would be a double input model with two inputs: a loose timecut input (e.g. yzt-x) and a tight timecut input (also yzt-x).
       The important thing is that both files contain the same events, just with different cuts/projections!
       Another usecase would be a double input xyz-t + xyz-c model.

    The output (train_files, test_files) is structured as follows:
    1) train/test: [ ( [train/test_filepath]  , n_rows) ]. The outmost list has len 1 as well as the list in the tuple.
    2) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has arbitrary length (depends on number of files), but len 1 for the list in the tuple.
       The two train and test input .list files should be structured as follows:
            file_0 \n
            file_1 \n
            file_2 # finish, no \n at last line!

    3) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has arbitrary length, as well as the list inside the tuple.
       The two train and test input .list files should be structured as follows:
            file_0_dset_0 \n
            file_0_dset_1 \n
            \n
            file_1_dset_0 \n
            file_1_dset_1 # finish, no \n at last line!

    :return: list(([train_filepaths], train_filesize)) train_files: list of tuples that contains the list(trainfiles) and their number of rows.
    :return: list(([test_filepaths], test_filesize)) test_files: list of tuples that contains the list(testfiles) and their number of rows.
    :return: bool multiple_inputs: flag that specifies if the -m option of the parser has been used.
    """
    parser = argparse.ArgumentParser(description='E.g. < python run_cnn.py train_filepath test_filepath [...] > \n'
                                                 'Script that runs a CNN. \n'
                                                 'The input arguments are either single files for train- and testdata or \n'
                                                 'a .list file that contains the filepaths of the train-/testdata.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('train_file', metavar='train_file', type=str, nargs='?', help='the filepath of the traindata file.')
    parser.add_argument('test_file', metavar='test_file', type=str, nargs='?', help='the filepath of the testdata file.')
    parser.add_argument('-l', '--list', dest='listfile_train_and_test', type=str, nargs=2,
                        help='filepaths of two .list files (train / test) that contains all .h5 files that should be used for training/testing.')
    parser.add_argument('-m', '--multiple_files', dest='listfile_multiple', type=str, nargs=2,
                        help='filepaths of two .list files where each contains multiple input files '
                             ' that should be used for training/testing in double/triple/... input models that need multiple input files per batch. \n'
                             'The required structure of the .list files can be found in /utilities/input_utilities parse_input()')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    multiple_inputs = False

    if args.listfile_train_and_test:
        train_files, test_files = [], []

        for line in open(args.listfile_train_and_test[0]):
            line = line.rstrip('\n')
            train_files.append(([line], h5_get_number_of_rows(line)))

        for line in open(args.listfile_train_and_test[1]):
            line = line.rstrip('\n')
            test_files.append(([line], h5_get_number_of_rows(line)))

    elif args.listfile_multiple:
        multiple_inputs = True

        train_files, test_files = [], []
        train_files_temp, test_files_temp = [], []

        for line in open(args.listfile_multiple[0]):

            if line == '\n': # newline separator which specifies that one set of different cnn input files has been finished
                # append one tuple (len > 1) to the train_files list for a certain datasplit number (e.g. file_0)
                train_files.append((train_files_temp, h5_get_number_of_rows(train_files_temp[0])))
                train_files_temp = []
                continue

            line = line.rstrip('\n')
            train_files_temp.append(line)

        train_files.append((train_files_temp, h5_get_number_of_rows(train_files_temp[0])))

        for line in open(args.listfile_multiple[1]):

            if line == '\n':  # newline separator which specifies that one set of different cnn input files has been finished
                # append one tuple (len > 1) to the train_files list for a certain datasplit number (e.g. file_0)
                test_files.append((test_files_temp, h5_get_number_of_rows(test_files_temp[0])))
                test_files_temp = []
                continue

            line = line.rstrip('\n')
            test_files_temp.append(line)

        test_files.append((test_files_temp, h5_get_number_of_rows(test_files_temp[0]))) # save temp to test_files for the last dataset block

    else:
        train_files = [([args.train_file], h5_get_number_of_rows(args.train_file))]
        test_files = [([args.test_file], h5_get_number_of_rows(args.test_file))]

    return train_files, test_files, multiple_inputs


def h5_get_number_of_rows(h5_filepath):
    """
    Gets the total number of rows of the first dataset of a .h5 file. Hence, all datasets should have the same number of rows!
    :param string h5_filepath: filepath of the .h5 file.
    :return: int number_of_rows: number of rows of the .h5 file in the first dataset.
    """
    f = h5py.File(h5_filepath, 'r')
    number_of_rows = f[list(f.keys())[0]].shape[0]
    f.close()

    return number_of_rows


def use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=False):
    """
    Copies the test and train files to the node-local ssd scratch folder and returns the new filepaths of the train and test data.
    Speeds up I/O and reduces RRZE network load.
    :param list train_files: list that contains all train files in tuples (filepath, f_size).
    :param list test_files: list that contains all test files in tuples (filepath, f_size).
    :param bool multiple_inputs: specifies if the -m option in the parser has been chosen. This means that the list inside the train/test_files tuple has more than one element!
    :return: list train_files_ssd, test_files_ssd: new train/test list with updated SSD /scratch filepaths.
    """
    local_scratch_path = os.environ['TMPDIR']
    train_files_ssd, test_files_ssd = [], []
    print('Copying the input train/test data to the node-local SSD scratch folder')

    if multiple_inputs is True:
        # in the case that we need multiple input data files for each batch, e.g. double input model with two different timecuts
        f_paths_train_ssd_temp, f_paths_test_ssd_temp = [], []

        for file_tuple in train_files:
            input_filepaths, f_size = file_tuple[0], file_tuple[1]

            for f_path in input_filepaths:
                shutil.copy2(f_path, local_scratch_path)  # copy to /scratch node-local SSD
                f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
                f_paths_train_ssd_temp.append(f_path_ssd)

            train_files_ssd.append((f_paths_train_ssd_temp, f_size)) # f_size of all f_paths should be the same
            f_paths_train_ssd_temp = []

        for file_tuple in test_files:
            input_filepaths, f_size = file_tuple[0], file_tuple[1]

            for f_path in input_filepaths:
                shutil.copy2(f_path, local_scratch_path)  # copy to /scratch node-local SSD
                f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
                f_paths_test_ssd_temp.append(f_path_ssd)

            test_files_ssd.append((f_paths_test_ssd_temp, f_size)) # f_size of all f_paths should be the same
            f_paths_test_ssd_temp = []

    else:
        for file_tuple in train_files:
            input_filepath, f_size = file_tuple[0][0], file_tuple[1]

            shutil.copy2(input_filepath, local_scratch_path) # copy to /scratch node-local SSD
            input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
            train_files_ssd.append(([input_filepath_ssd], f_size))

        for file_tuple in test_files:
            input_filepath, f_size = file_tuple[0][0], file_tuple[1]

            shutil.copy2(input_filepath, local_scratch_path) # copy to /scratch node-local SSD
            input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
            test_files_ssd.append(([input_filepath_ssd], f_size))

    print('Finished copying the input train/test data to the node-local SSD scratch folder')
    return train_files_ssd, test_files_ssd