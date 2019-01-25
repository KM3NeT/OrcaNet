#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD.

Reading toml files: There are three different keywords:
    "input" :   The input to networks.
    "config" :  The Configuration.
    "model" :   Options for auto-generated OrcaNet models.

"""

import os
import shutil
import h5py
import toml
import numpy as np
from datetime import datetime


def read_out_config_file(file):
    """
    Extract the variables of a model from the .toml file and convert them to a dict.

    Toml can not handle arrays with mixed types of variables, so some conversion are done.

    Parameters
    ----------
    file : str
        Path and name of the .toml file that defines the properties of the model.

    Returns
    -------
    keyword_arguments : dict
        Values for the OrcaNet scripts, as listed in the Configuration class.

    """
    file_content = toml.load(file)["config"]
    if "class_type" in file_content:
        if file_content["class_type"][0] == "None":
            file_content["class_type"][0] = None
    if "n_gpu" in file_content:
        file_content["n_gpu"][0] = int(file_content["n_gpu"][0])
    return file_content


def read_out_list_file(file):
    """
    Reads out a list file in .toml format containing the pathes to training
    and validation files and bring it into the proper format.

    Parameters
    ----------
    file : str
        Path to a .list file containing the paths to training and validation files to be used during training.

    Returns
    -------
    train_files : dict
        A dict containing the paths to the different training files given in the list_file.
        Example for the output format:
                {
                 "input_A" : ['path/to/set_A_train_file_1.h5', 'path/to/set_A_train_file_2.h5', ...]
                 "input_B" : ['path/to/set_B_train_file_1.h5', 'path/to/set_B_train_file_2.h5', ...]
                 ...
                }
    validation_files : dict
        Like the above but for validation files.
    n_train : int
        The number of train files.
    n_val : int
        The number of validation files.

    Raises
    -------
    AssertionError
        If different inputs have a different number of training or validation files.

    """
    # a dict with inputnames as keys and dicts with the lists of train/val files as values
    file_content = toml.load(file)

    train_files, validation_files = {}, {}
    # no of train/val fiels in each input set
    n_train, n_val = [], []
    for input_key in file_content:
        train_files[input_key] = file_content[input_key]["train_files"]
        validation_files[input_key] = file_content[input_key]["validation_files"]
        n_train.append(len(train_files[input_key]))
        n_val.append(len(validation_files[input_key]))

    if not n_train.count(n_train[0]) == len(n_train):
        raise AssertionError("The specified training inputs do not all have the same number of files")
    if not n_val.count(n_val[0]) == len(n_val):
        raise AssertionError("The specified validation inputs do not all have the same number of files")
    # TODO Maybe files have different number of events? Should give an error
    return train_files, validation_files, n_train, n_val


def read_out_model_file(file):
    """
    Read out parameters for creating models with OrcaNet from a toml file.

    Parameters
    ----------
    file : str
        Path to the toml file.

    Returns
    -------
    nn_arch : str
        Name of the architecture to be loaded.
    loss_opt : tuple
        The losses and weights of the model.
    file_content : dict
        Keyword arguments for the model generation.

    """
    file_content = toml.load(file)["model"]
    nn_arch = file_content.pop("nn_arch")
    losses = file_content.pop("losses")
    loss_opt = (losses, None)
    return nn_arch, loss_opt, file_content


def write_full_logfile_startup(cfg):
    """
    Whenever the orca_train function is run, this logs all the input parameters in the full log file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('----------------------------------'+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'---------------------------------------------------\n\n')
        f_out.write("New execution of the orca_train function started with the following options:\n\n")
        f_out.write("List file path:\t"+cfg.get_list_file()+"\n")
        f_out.write("Given trainfiles in the .list file:\n")
        f_out.write(str(cfg.get_train_files())+"\n")
        f_out.write("Given validation files in the .list file:\n")
        f_out.write(str(cfg.get_val_files()) + "\n")
        f_out.write("\nConfiguration used:\n")
        for key in vars(cfg):
            if not key.startswith("_"):
                f_out.write("   {}:\t{}\n".format(key, getattr(cfg, key)))
        f_out.write("\nPrivate attributes:\n")
        for key in vars(cfg):
            if key.startswith("_"):
                f_out.write("   {}:\t{}\n".format(key, getattr(cfg, key)))
        f_out.write("\n")


def write_full_logfile(cfg, model, history_train, history_val, lr, epoch, train_file):
    """
    Function for saving various information during training and validation to a .txt file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('---------------Epoch {} File {}-------------------------------------------------------------------------\n'.format(epoch[0], epoch[1]))
        f_out.write('\n')
        f_out.write('Current time: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n')
        f_out.write('Decayed learning rate to ' + str(lr) + ' before epoch ' + str(epoch[0]) +
                    ' and file ' + str(epoch[1]) + ')\n')
        f_out.write('Trained in epoch ' + str(epoch) + ' on file ' + str(epoch[1]) + ', ' + str(train_file) + '\n')
        if history_val is not None:
            f_out.write('Validated in epoch ' + str(epoch) + ', file ' + str(epoch[1]) + ' on val_files ' + str(cfg.get_val_files()) + '\n')
        f_out.write('History for training / validating: \n')
        f_out.write('Train: ' + str(history_train.history) + '\n')
        if history_val is not None:
            f_out.write('Validation: ' + str(history_val) + ' (' + str(model.metrics_names) + ')' + '\n')
        f_out.write('\n')
        # f_out.write('Additional Info:\n')
        # f_out.write('Batchsize=' + str(batchsize) + ', n_bins=' + str(n_bins) +
        #            ', class_type=' + str(class_type) + '\n' +
        #            'swap_4d_channels=' + str(swap_4d_channels) + ', str_ident=' + str_ident + '\n')
        # f_out.write('\n')


def write_summary_logfile(cfg, epoch, model, history_train, history_val, lr):
    """
    Write to the summary.txt file in every trained model folder.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    epoch : tuple(int, int)
        The number of the current epoch and the current filenumber.
    model : ks.model.Model
        Keras model instance of a neural network.
    history_train : Keras history object
        History object containing the history of the training, averaged over files.
    history_val : List
        List of validation losses for all the metrics, averaged over all validation files.
    lr : float
        The current learning rate of the model.

    """
    # get this for the epoch_number_float in the logfile
    steps_per_total_epoch, steps_cum = 0, [0]
    for f_size in cfg.get_train_file_sizes():
        steps_per_file = int(f_size / cfg.batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)
    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)

    # Write the logfile
    logfile_fname = cfg.main_folder + 'summary.txt'
    with open(logfile_fname, 'a+') as logfile:
        # Write the headline
        if os.stat(logfile_fname).st_size == 0:
            logfile.write('Epoch\tLR\t')
            for i, metric in enumerate(model.metrics_names):
                logfile.write("train_" + str(metric) + "\tval_" + str(metric))
                if i + 1 < len(model.metrics_names):
                    logfile.write("\t")
            logfile.write('\n')
        # Write the content: Epoch, LR, train_1, val_1, ...
        logfile.write("{:.4g}\t".format(float(epoch_number_float)))
        logfile.write("{:.4g}\t".format(float(lr)))
        for i, metric_name in enumerate(model.metrics_names):
            logfile.write("{:.4g}\t".format(float(history_train.history[metric_name][0])))
            if history_val is None:
                logfile.write("nan")
            else:
                logfile.write("{:.4g}".format(float(history_val[i])))
            if i + 1 < len(model.metrics_names):
                logfile.write("\t")
        logfile.write('\n')


def read_logfiles(summary_logfile):
    """
    Read out the data from the summary.txt file, and from all training log files in the log_train folder which
    is in the same directory as the summary.txt file.

    Parameters
    ----------
    summary_logfile : str
        Path of the summary.txt file in a model folder.

    Returns
    -------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.

    """
    summary_data = np.genfromtxt(summary_logfile, names=True, delimiter="\t")

    # list of all files in the log_train folder of this model
    log_train_folder = "/".join(summary_logfile.split("/")[:-1])+"/log_train/"
    files = os.listdir(log_train_folder)
    train_file_data = []
    for file in files:
        # file is something like "log_epoch_1_file_2.txt", extract the 1 and 2:
        epoch, file_no = [int(file.split(".")[0].split("_")[i]) for i in [2,4]]
        file_data = np.genfromtxt(log_train_folder+file, names=True, delimiter="\t")
        train_file_data.append([[epoch, file_no], file_data])
    train_file_data.sort()
    full_train_data = train_file_data[0][1]
    for [epoch, file_no], file_data in train_file_data[1:]:
        # file_data["Batch_float"]+=(epoch-1)
        full_train_data = np.append(full_train_data, file_data)
    return summary_data, full_train_data


def h5_get_number_of_rows(h5_filepath):
    """
    Gets the total number of rows of the first dataset of a .h5 file. Hence, all datasets should have the same number of rows!
    :param string h5_filepath: filepath of the .h5 file.
    :return: int number_of_rows: number of rows of the .h5 file in the first dataset.
    """
    f = h5py.File(h5_filepath, 'r')
    # remove any keys to pytables folders that may be in the file
    f_keys_stripped = [x for x in list(f.keys()) if '_i_' not in x]

    number_of_rows = f[f_keys_stripped[0]].shape[0]
    f.close()
    return number_of_rows


def h5_get_n_bins(train_files):
    """
    Get the number of bins from the training files. Only the first files are looked up, the others should be identical.
    CAREFUL: The name of the neural network 'images' dataset in the files MUST be 'x'.

    Parameters
    ----------
    train_files : List
        A list containing the paths to the different training files given in the list_file.
        Example format:
                [
                 [['path/to/train_file_1_dimx.h5', 'path/to/train_file_1_dimy.h5'], number_of_events_train_files_1],
                 [['path/to/train_file_2_dimx.h5', 'path/to/train_file_2_dimy.h5'], number_of_events_train_files_1]
                ]

    Returns
    -------
    n_bins : list

    """
    n_bins = []
    for dim_file in train_files[0][0]:
        f = h5py.File(dim_file, "r")
        n_bins.append(f['x'].shape[1:])
    return n_bins


def use_node_local_ssd_for_input(train_files, val_files):
    """
    Copies the test and train files to the node-local ssd scratch folder and returns the new filepaths of the train and test data.
    Speeds up I/O and reduces RRZE network load.

    Parameters
    ----------
    train_files : dict
        Dict containing the train file pathes.
    val_files
        Dict containing the val file pathes.

    Returns
    -------
    train_files_ssd : dict
        Train dict with updated SSD /scratch filepaths.
    test_files_ssd : dict
        Val dict with updated SSD /scratch filepaths.

    """
    local_scratch_path = os.environ['TMPDIR']
    train_files_ssd, val_files_ssd = {}, {}

    for input_key in train_files:
        old_pathes = train_files[input_key]
        new_pathes = []
        for f_path in old_pathes:
            # copy to /scratch node-local SSD
            f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
            print("Copying", f_path, "\nto", f_path_ssd)
            shutil.copy2(f_path, local_scratch_path)
            new_pathes.append(f_path_ssd)
        train_files_ssd[input_key] = new_pathes

    for input_key in val_files:
        old_pathes = val_files[input_key]
        new_pathes = []
        for f_path in old_pathes:
            # copy to /scratch node-local SSD
            f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
            print("Copying", f_path, "\nto", f_path_ssd)
            shutil.copy2(f_path, local_scratch_path)
            new_pathes.append(f_path_ssd)
        val_files_ssd[input_key] = new_pathes

    print('Finished copying the input train/test data to the node-local SSD scratch folder.')
    return train_files_ssd, val_files_ssd
