#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code regarding reading user input, and writing output like logfiles.
"""

import os
import shutil
import h5py
import toml
import numpy as np
from datetime import datetime
from collections import namedtuple


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
                 "input_A" : ('path/to/set_A_train_file_1.h5', 'path/to/set_A_train_file_2.h5', ...)
                 "input_B" : ('path/to/set_B_train_file_1.h5', 'path/to/set_B_train_file_2.h5', ...)
                 ...
                }
    validation_files : dict
        Like the above but for validation files.

    Raises
    -------
    AssertionError
        If different inputs have a different number of training or validation files.

    """
    # a dict with inputnames as keys and dicts with the lists of train/val files as values
    file_content = toml.load(file)

    train_files, validation_files = {}, {}
    # no of train/val files in each input set
    n_train, n_val = [], []
    for input_key, input_values in file_content.items():
        assert isinstance(input_values, dict) and len(input_values.keys()) == 2, \
            "Wrong input format in toml list file (input {}: {})".format(input_key, input_values)
        assert "train_files" in input_values.keys(), "No train files specified in toml list file"
        assert "validation_files" in input_values.keys(), "No validation files specified in toml list file"

        train_files[input_key] = tuple(input_values["train_files"])
        validation_files[input_key] = tuple(input_values["validation_files"])
        n_train.append(len(train_files[input_key]))
        n_val.append(len(validation_files[input_key]))

    if not n_train.count(n_train[0]) == len(n_train):
        raise AssertionError("The specified training inputs do not all have the same number of files!")
    if not n_val.count(n_val[0]) == len(n_val):
        raise AssertionError("The specified validation inputs do not all have the same number of files!")
    return train_files, validation_files


def read_out_model_file(file):
    """
    Read out parameters for creating models with OrcaNet from a toml file.

    Parameters
    ----------
    file : str
        Path to the toml file.

    Returns
    -------
    modeldata : namedtuple
        Infos for building a predefined model with OrcaNet.

    """
    file_content = toml.load(file)['model']
    nn_arch = file_content.pop('nn_arch')
    compile_opt = file_content.pop('compile_opt')

    class_type = ''
    str_ident = ''
    swap_4d_channels = None

    if 'class_type' in file_content:
        class_type = file_content.pop('class_type')
    if 'str_ident' in file_content:
        str_ident = file_content.pop('str_ident')
    if 'swap_4d_channels' in file_content:
        swap_4d_channels = file_content.pop('swap_4d_channels')

    ModelData = namedtuple('ModelData', 'nn_arch compile_opt class_type str_ident swap_4d_channels args')
    modeldata = ModelData(nn_arch, compile_opt, class_type, str_ident, swap_4d_channels, file_content)

    return modeldata


def write_full_logfile_startup(cfg):
    """
    Whenever the train function is run, this logs all the input parameters in the full log file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('----------------------------------'+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'---------------------------------------------------\n\n')
        f_out.write("New training started with the following options:\n\n")
        f_out.write("List file path:\t"+cfg.get_list_file()+"\n")

        f_out.write("Given trainfiles in the .list file:\n")
        for input_name, input_files in cfg.get_train_files().items():
            f_out.write(input_name + ":")
            [f_out.write("\t" + input_file + "\n") for input_file in input_files]

        f_out.write("Given validation files in the .list file:\n")
        for input_name, input_files in cfg.get_val_files().items():
            f_out.write(input_name + ":")
            [f_out.write("\t" + input_file + "\n") for input_file in input_files]

        f_out.write("\nConfiguration used:\n")
        for key in vars(cfg):
            if not key.startswith("_"):
                f_out.write("   {}:\t{}\n".format(key, getattr(cfg, key)))

        modeldata = cfg.get_modeldata()
        if modeldata is not None:
            f_out.write("Given modeldata:")
            for key, val in modeldata._asdict().items():
                f_out.write("\t{}:\t{}".format(key, val))

        f_out.write("\n")


def write_full_logfile(cfg, model, history_train, history_val, lr, epoch, files_dict):
    """
    Function for saving various information during training and validation to a .txt file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    model : Model
        The keras model.
    history_train : keras history object
        The history of the training.
    history_val : List or None
        The history of the validation.
    lr : float
        The current learning rate.
    epoch : tuple
        Current epoch and file number.
    files_dict : dict
        The name of every input as a key, the path to one of the training file, on which the model has just been trained, as values.

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('---------------Epoch {} File {}-------------------------------------------------------------------------\n'.format(epoch[0], epoch[1]))
        f_out.write('\n')
        f_out.write('Current time: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n')
        f_out.write('Decayed learning rate to ' + str(lr) + ' before epoch ' + str(epoch[0]) +
                    ' and file ' + str(epoch[1]) + ')\n')
        f_out.write('Trained in epoch ' + str(epoch) + ' file number ' + str(epoch[1]) + ', on files ' + str(files_dict) + '\n')
        if history_val is not None:
            f_out.write('Validated in epoch ' + str(epoch) + ', file ' + str(epoch[1]) + 'on the val files\n')
        f_out.write('History for training / validating: \n')
        f_out.write('Train: ' + str(history_train.history) + '\n')
        if history_val is not None:
            f_out.write('Validation: ' + str(history_val) + ' (' + str(model.metrics_names) + ')' + '\n')
        f_out.write('\n')


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
    for f_size in cfg.get_file_sizes("train"):
        steps_per_file = int(f_size / cfg.batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)
    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)

    # Get the widths of the columns. They depend on the widths of the metric names in the first line
    data = ["Epoch", "LR", ]
    for i, metric_name in enumerate(model.metrics_names):
        data.append("train_" + str(metric_name))
        data.append("val_" + str(metric_name))
    headline, widths = get_summary_log_line(data)

    logfile_fname = cfg.main_folder + 'summary.txt'
    with open(logfile_fname, 'a+') as logfile:
        # Write the two headlines if the file is empty
        if os.stat(logfile_fname).st_size == 0:
            logfile.write(headline + "\n")

            vline = ["-" * width for width in widths]
            vertical_line = get_summary_log_line(vline, widths, seperator="-+-")
            logfile.write(vertical_line + "\n")

        # Write the content: Epoch, LR, train_1, val_1, ...
        data = [epoch_number_float, lr]
        for i, metric_name in enumerate(model.metrics_names):
            data.append(history_train.history[metric_name][0])
            if history_val is None:
                data.append("nan")
            else:
                data.append(history_val[i])

        line = get_summary_log_line(data, widths)
        logfile.write(line + '\n')


def get_summary_log_line(data, widths=None, seperator=" | ", minimum_cell_width=9, float_precision=4):
    """
    Get a line in the proper format for the summary plot, consisting of multiple spaced and seperated cells.

    Parameters
    ----------
    data : List
        Strings or floats of what is in each cell.
    widths : List or None
        Optional: The width of every cell. If None, will set it automatically, depending on the data.
        If widths is given, but what is given in data is wider than the width, the cell will expand without notice.
    minimum_cell_width : int
        If widths is None, this defines the minimum width of the cells in characters.
    seperator : str
        String that seperates two adjacent cells.
    float_precision : int
        Precision to which floats are rounded if they appear in data.

    Returns
    -------
    line : str
        The line.
    new_widths : List
        Optional: If the input widths were None: The widths of the cells .

    """
    if widths is None:
        new_widths = []

    line = ""
    for i, entry in enumerate(data):
        # no seperator before the first entry
        if i == 0:
            sep = ""
        else:
            sep = seperator

        # If entry is a number, round to given precision and make it a string
        if not isinstance(entry, str):
            entry = format(float(entry), "."+str(float_precision)+"g")

        if widths is None:
            cell_width = max(minimum_cell_width, len(entry))
            new_widths.append(cell_width)
        else:
            cell_width = widths[i]

        cell_cont = format(entry, "<"+str(cell_width))

        line += "{seperator}{entry}".format(seperator=sep, entry=cell_cont,)

    if widths is None:
        return line, new_widths
    else:
        return line


def read_logfiles(cfg):
    """
    Read out the data from the summary.txt file, and from all training log files in the log_train folder which
    is in the same directory as the summary.txt file.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    Returns
    -------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.

    """
    summary_data = np.genfromtxt(cfg.main_folder + "/summary.txt", names=True, delimiter="|", autostrip=True, comments="--")

    # list of all files in the log_train folder of this model
    log_train_folder = cfg.get_subfolder("log_train")
    files = os.listdir(log_train_folder)
    train_file_data = []
    for file in files:
        if not (file.startswith("log_epoch_") and file.endswith(".txt")):
            continue
        # file is something like "log_epoch_1_file_2.txt", extract the 1 and 2:
        epoch, file_no = [int(file.split(".")[0].split("_")[i]) for i in [2, 4]]
        file_data = np.genfromtxt(log_train_folder+"/"+file, names=True, delimiter="\t")
        train_file_data.append([[epoch, file_no], file_data])

    train_file_data.sort()
    full_train_data = train_file_data[0][1]
    for [epoch, file_no], file_data in train_file_data[1:]:
        full_train_data = np.append(full_train_data, file_data)
    return summary_data, full_train_data


def h5_get_number_of_rows(h5_filepath, datasets):
    """
    Gets the total number of rows of of a .h5 file.

    Multiple dataset names can be given as a list to check if they all have the same number of rows (axis 0).

    Parameters
    ----------
    h5_filepath : str
        filepath of the .h5 file.
    datasets : list
        The names of datasets in the file to check.

    Returns
    -------
    number_of_rows: int
        number of rows of the .h5 file in the first dataset.

    Raises
    ------
    AssertionError
        If the given datasets do not have the same no of rows.

    """
    with h5py.File(h5_filepath, 'r') as f:
        number_of_rows = [f[dataset].shape[0] for dataset in datasets]
    if not number_of_rows.count(number_of_rows[0]) == len(number_of_rows):
        err_msg = "Datasets do not have the same number of samples in file " + h5_filepath
        for i, dataset in enumerate(datasets):
            err_msg += "\nDataset: {}\tSamples: {}".format(dataset, number_of_rows[i])
        raise AssertionError(err_msg)
    return number_of_rows[0]


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
