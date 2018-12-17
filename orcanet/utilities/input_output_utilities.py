#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD."""

import os
import shutil
import h5py
import toml
from time import gmtime, strftime

def read_out_config_file(config_file):
    """
    Extract the properties of the model which will be built from a .toml file. These are handed to the execute_nn
    function in run_nn.py.

    Parameters
    ----------
    config_file : str
        Path and name of the .toml file that defines the properties of the model.
    Returns
    -------
        positional_arguments: tuple
            Positional arguments of the execute_nn function.
        config["keyword_arguments"] : dict
            Keyword arguments of the execute_nn function.
    """
    config = toml.load(config_file)

    loss_opt = (config["losses"], None)
    n_bins = config["positional_arguments"]["n_bins"]
    if config["positional_arguments"]["class_type"][0]=="None":
        config["positional_arguments"]["class_type"][0] = None
    class_type = config["positional_arguments"]["class_type"]
    nn_arch = config["positional_arguments"]["nn_arch"]
    epoch = config["positional_arguments"]["epoch"]
    mode = config["positional_arguments"]["mode"]
    if config["positional_arguments"]["swap_4d_channels"]=="None":
        config["positional_arguments"]["swap_4d_channels"] = None
    swap_4d_channels = config["positional_arguments"]["swap_4d_channels"]
    positional_arguments = (loss_opt, n_bins, class_type, nn_arch, epoch, mode, swap_4d_channels)

    if "n_gpu" in config["keyword_arguments"]:
        config["keyword_arguments"]["n_gpu"][0] = int(config["keyword_arguments"]["n_gpu"][0])

    return positional_arguments, config["keyword_arguments"]


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


def write_summary_logfile(train_files, batchsize, epoch, folder_name, model, history_train, history_test, lr):
    """
    Write to the summary.txt file in every trained model folder.

    Parameters
    ----------
    train_files : list(([train_filepaths], train_filesize))
        List of tuples with the filepaths and the filesizes of the train_files.
    batchsize : int
        Batchsize that is used in the evaluate_generator method.
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.
    folder_name : str
        Name of the folder in the cnns directory in which everything will be saved.
    model : ks.model.Model
        Keras model instance of a neural network.
    history_train : Keras history object
        History object containing the history of the training, averaged over files.
    history_test : list
        List of test losses for all the metrics, averaged over all test files.
    lr : float
        The current learning rate of the model.
    """
    # Save test log
    steps_per_total_epoch, steps_cum = 0, [0] # get this for the epoch_number_float in the logfile
    for f, f_size in train_files:
        steps_per_file = int(f_size / batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)

    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)
    logfile_fname = folder_name + '/summary.txt'
    with open(logfile_fname, 'a+') as logfile:
        # Write the headline
        if os.stat(logfile_fname).st_size == 0:
            logfile.write('#Epoch\tLR\t')
            for i, metric in enumerate(model.metrics_names):
                logfile.write("train_" + str(metric) + "\ttest_" + str(metric) + "\t")
            logfile.write('\n')
        # Write the content: Epoch, LR, train_1, test_1, ...
        logfile.write("{:.4g}\t".format(float(epoch_number_float)))
        logfile.write("{:.4g}\t".format(float(lr)))
        for i, metric_name in enumerate(model.metrics_names):
            logfile.write("{:.4g}\t".format(float(history_train.history[metric_name][0])))
            if history_test is None:
                logfile.write("nan\t")
            else:
                logfile.write("{:.4g}\t".format(float(history_test[i])))
        logfile.write('\n')


def write_full_logfile(model, history_train, history_test, lr, lr_decay, epoch, train_file,
                            test_files, batchsize, n_bins, class_type, swap_4d_channels, str_ident, folder_name):
    """
    Function for saving various information during training and testing to a .txt file.
    """
    logfile=folder_name + '/full_log.txt'
    with open(logfile, 'a+') as f_out:
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