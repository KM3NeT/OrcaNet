#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD."""

import os
import shutil
import h5py
import toml

def config_test(file):
    config = toml.load(config_file)

def read_out_config_file(config_file):
    """
    Extract the properties of the model which will be built from a .toml file.

    Parameters
    ----------
    config_file : str
        Path and name of the .toml file that defines the properties of the model.
    Returns
    -------
    Many different inpput options handed directly to the function execute_nn.py in run_cnn.py.
    See there for documentation.
    """
    config = toml.load(config_file)
    # Adjustment of values (as toml has strict requirements for the format, e.g. lists can only contain
    #   variables of the same type
    # the losses are in lists like [name, metric, weight] in the toml file, all as strings
    losses = []
    for loss in config["losses"]:
        if len(loss)==3:
            losses.append( [loss[0], [loss[1], float(loss[2])]] )
        else:
            losses.append( [loss[0], [loss[1], 1.0]] )
    config["losses"] = dict(losses)
    if config["class_type"][0]=="None":
        config["class_type"][0] = None
    config["n_gpu"][0] = int(config["n_gpu"][0])

    n_bins = config["n_bins"]
    class_type = config["class_type"]
    nn_arch = config["nn_arch"]
    batchsize = config["batchsize"]
    epoch = config["epoch"]
    n_gpu = config["n_gpu"]
    mode = config["mode"]
    swap_4d_channels = config["swap_4d_channels"]
    use_scratch_ssd = config["use_scratch_ssd"]
    zero_center = config["zero_center"]
    shuffle=(False, None)
    str_ident = config["str_ident"]

    losses = config["losses"]
    loss_opt = (losses, None)

    return n_bins, class_type, nn_arch, batchsize, epoch, \
           n_gpu, mode, swap_4d_channels, use_scratch_ssd,\
            zero_center, shuffle, str_ident, loss_opt

def read_out_list_file(list_file):
    """
    Reads out input files for network training. The format of the list file should be as follows:
        Single empty line seperates train from test files, double empty line seperates different training and test sets which
        are given simultaneosly to a network.

    Parameters
    ----------
    list_file : str
        Path to a .list file containing the pathes to training and test files to be used during training.

    Returns
    -------
    train_files : list
        A list containing the paths to the different training files given in the list_file.
        Example format:
                [
                 [['path/to/train_file_1_dimx.h5', 'path/to/train_file_1_dimy.h5'], number_of_events_train_files_1],
                 [['path/to/train_file_2_dimx.h5', 'path/to/train_file_2_dimy.h5'], number_of_events_train_files_1]
                ]
    test_files : list
        Like the above but for test files.
    multiple_inputs : bool
        Whether seperate sets of input files were given (e.g. for networks taking data
        simulataneosly from different files).
    """
    lines=[]
    with open(list_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if line[:1]=="#":
                continue
            else:
                lines.append(line)
    multiple_inputs=False
    train_files_list, test_files_list = [], []
    train_files_in_dimension, test_files_in_dimension = [], []
    empty_lines_read = 0
    for line in lines:
        if line=="":
            empty_lines_read += 1
            continue

        if empty_lines_read==0:
            train_files_in_dimension.append(line)
        elif empty_lines_read==1:
            test_files_in_dimension.append(line)
        elif empty_lines_read==3:
            # Block for one dimension is finished
            train_files_list.append(train_files_in_dimension)
            test_files_list.append(test_files_in_dimension)
            train_files_in_dimension, test_files_in_dimension = [], []
            empty_lines_read = 0
            train_files_in_dimension.append(line)
            multiple_inputs=True
        elif empty_lines_read==2:
            raise ValueError("Check formating of the list file! (empty_lines_read counter is at {} during readoout of file {})".format(empty_lines_read, list_file))
    train_files_list.append(train_files_in_dimension)
    test_files_list.append(test_files_in_dimension)
    """
    Format of train_files_list at this point:
    [
     ['path/to/train_file_1_dimx.h5', 'path/to/train_file_2_dimx.h5'],
     ['path/to/train_file_1_dimy.h5', 'path/to/train_file_2_dimy.h5']
    ]
    Reformat to match the desired output format. Only look up dimension of first file (others should be the same).
    """
    train_files, test_files = [], []
    for set_number in range(len(train_files_list[0])):
        file_set = [dimension[set_number] for dimension in train_files_list]
        train_files.append([file_set, h5_get_number_of_rows(file_set[0])])
    for set_number in range(len(test_files_list[0])):
        file_set = [dimension[set_number] for dimension in test_files_list]
        test_files.append([file_set, h5_get_number_of_rows(file_set[0])])

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