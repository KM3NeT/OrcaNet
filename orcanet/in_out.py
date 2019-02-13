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

from orcanet.utilities.nn_utilities import get_inputs, generate_batches_from_hdf5_file


class IOHandler(object):
    """
    Access info indirectly contained in the cfg object.

    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_latest_epoch(self):
        """
        Check all saved models in the ./saved_models folder and return the highest epoch / file_no pair.

        Will only consider files that end with .h5 as models.

        Returns
        -------
        latest_epoch : tuple
            The highest epoch, file_no pair. (0,0) if the folder is empty or does not exist yet.

        """
        if os.path.exists(self.cfg.output_folder + "saved_models"):
            files = []
            for file in os.listdir(self.cfg.output_folder + "saved_models"):
                if file.endswith('.h5'):
                    files.append(file)

            if len(files) == 0:
                latest_epoch = (0, 0)
            else:
                epochs = []
                for file in files:
                    epoch, file_no = file.split("model_epoch_")[-1].split(".h5")[0].split("_file_")
                    epochs.append((int(epoch), int(file_no)))
                latest_epoch = max(epochs)
        else:
            latest_epoch = (0, 0)
        return latest_epoch

    def get_next_epoch(self, epoch):
        """
        Return the next epoch / fileno tuple (depends on how many train files there are).

        Parameters
        ----------
        epoch : tuple
            Current epoch and file number.

        Returns
        -------
        next_epoch : tuple
            Next epoch and file number.

        """
        if epoch[0] == 0 and epoch[1] == 0:
            next_epoch = (1, 1)
        elif epoch[1] == self.get_no_of_files("train"):
            next_epoch = (epoch[0] + 1, 1)
        else:
            next_epoch = (epoch[0], epoch[1] + 1)
        return next_epoch

    def get_subfolder(self, name=None, create=False):
        """
        Get the path to one or all subfolders of the main folder.

        Parameters
        ----------
        name : str or None
            The name of the subfolder.
        create : bool
            If the subfolder should be created if it does not exist.

        Returns
        -------
        subfolder : str or tuple
            The path of the subfolder. If name is None, all subfolders will be returned as a tuple.

        """
        def get(fdr):
            subfdr = subfolders[fdr]
            if create and not os.path.exists(subfdr):
                print("Creating directory: " + subfdr)
                os.makedirs(subfdr)
            return subfdr

        subfolders = {"log_train": self.cfg.output_folder + "log_train",
                      "saved_models": self.cfg.output_folder + "saved_models",
                      "plots": self.cfg.output_folder + "plots",
                      "activations": self.cfg.output_folder + "plots/activations",
                      "predictions": self.cfg.output_folder + "predictions"}

        if name is None:
            subfolder = [get(name) for name in subfolders]
        else:
            subfolder = get(name)
        return subfolder

    def get_model_path(self, epoch, fileno):
        """
        Get the path to a model (which might not exist yet).

        Parameters
        ----------
        epoch : int
            The epoch.
        fileno : int
            The file number.

        Returns
        -------
        model_filename : str
            Path to a model.

        """
        model_filename = self.get_subfolder("saved_models") + '/model_epoch_' + str(epoch) + '_file_' + str(fileno) + '.h5'
        return model_filename

    def get_pred_path(self, epoch, fileno, list_name):
        """ Get the path to a saved prediction. """
        pred_filename = self.get_subfolder("predictions") + '/pred_model_epoch_{}_file_{}_on_{}_val_files.h5'.format(epoch, fileno, list_name)
        return pred_filename

    def get_n_bins(self):
        """
        Get the number of bins from the training files.

        Only the first files are looked up, the others should be identical.

        Returns
        -------
        n_bins : dict
            Toml-list input names as keys, list of the bins as values.

        """
        train_files = self.cfg.get_files("train")
        n_bins = {}
        for input_key in train_files:
            with h5py.File(train_files[input_key][0], "r") as f:
                n_bins[input_key] = f[self.cfg.key_samples].shape[1:]
        return n_bins

    def get_file_sizes(self, which):
        """
        Get the number of samples in each training or validation input file.

        Parameters
        ----------
        which : str
            Either train or val.

        Returns
        -------
        file_sizes : list
            Its length is equal to the number of files in each input set.

        Raises
        ------
        AssertionError
            If there is a different number of samples in any of the files of all inputs.

        """
        file_sizes_full, error_file_sizes, file_sizes = {}, [], []
        for n, file_no_set in enumerate(self.yield_files(which)):
            # the number of samples in the n-th file of all inputs
            file_sizes_full[n] = [h5_get_number_of_rows(file, datasets=[self.cfg.key_labels, self.cfg.key_samples])
                                  for file in file_no_set.values()]
            if not file_sizes_full[n].count(file_sizes_full[n][0]) == len(file_sizes_full[n]):
                error_file_sizes.append(n)
            else:
                file_sizes.append(file_sizes_full[n][0])

        if len(error_file_sizes) != 0:
            err_msg = "The files you gave for the different inputs of the model do not all have the same " \
                      "number of samples!\n"
            for n in error_file_sizes:
                err_msg += "File no {} in {} has the following files sizes for the different inputs: {}\n".format(n, which, file_sizes_full[n])
            raise AssertionError(err_msg)

        return file_sizes

    def get_no_of_files(self, which):
        """
        Return the number of training or validation files.

        Only looks up the no of files of one (random) list input, as equal length is checked during read in.

        Parameters
        ----------
        which : str
            Either train or val.

        Returns
        -------
        no_of_files : int
            The number of files.

        """
        files = self.cfg.get_files(which)
        no_of_files = len(list(files.values())[0])
        return no_of_files

    def yield_files(self, which):
        """
        Yield a training or validation file for every input.

        Parameters
        ----------
        which : str
            Either train or val.

        Yields
        ------
        files_dict : dict
            The name of every toml list input as a key, one of the filepaths as values.
            They will be yielded in the same order as they are given in the toml file.

        """
        files = self.cfg.get_files(which)
        for file_no in range(self.get_no_of_files(which)):
            files_dict = {key: files[key][file_no] for key in files}
            yield files_dict

    def check_connections(self, model):
        """
        Check if the names and shapes of the samples and labels in the given input files work with the model.

        Also takes into account the possibly present sample or label modifiers.

        Parameters
        ----------
        model : ks.model
            A keras model.

        Raises
        ------
        AssertionError
            If they dont work together.

        """
        def check_for_error(list_ns, layer_ns):
            """ Get the names of the layers which dont have a fitting counterpart from the toml list. """
            # Both inputs are dicts with  name: shape  of input/output layers/data
            err_names, err_shapes = [], []
            for layer_name in layer_ns:
                if layer_name not in list_ns.keys():
                    # no matching name
                    err_names.append(layer_name)
                elif list_ns[layer_name] != layer_ns[layer_name]:
                    # no matching shape
                    err_shapes.append(layer_name)
            return err_names, err_shapes

        print("\nInput check\n-----------")
        # Get a batch of data to investigate the given modifier functions
        xs, y_values = self.get_batch()
        layer_inputs = get_inputs(model)
        # keys: name of layers, values: shape of input
        layer_inp_shapes = {key: layer_inputs[key].input_shape[1:] for key in layer_inputs}
        list_inp_shapes = self.get_n_bins()

        print("The inputs in your toml list file have the following names and shapes:")
        for list_key in list_inp_shapes:
            print("\t{}\t{}".format(list_key, list_inp_shapes[list_key]))

        if self.cfg.sample_modifier is None:
            print("\nYou did not specify a sample modifier.")
        else:
            modified_xs = self.cfg.sample_modifier(xs)
            modified_shapes = {modi_key: modified_xs[modi_key].shape[1:] for modi_key in modified_xs}
            print("\nAfter applying your sample modifier, they have the following names and shapes:")
            for list_key in modified_shapes:
                print("\t{}\t{}".format(list_key, modified_shapes[list_key]))
            list_inp_shapes = modified_shapes

        print("\nYour model requires the following input names and shapes:")
        for layer_key in layer_inp_shapes:
            print("\t{}\t{}".format(layer_key, layer_inp_shapes[layer_key]))

        err_inp_names, err_inp_shapes = check_for_error(list_inp_shapes, layer_inp_shapes)

        err_msg_inp = ""
        if len(err_inp_names) == 0 and len(err_inp_shapes) == 0:
            print("\nInput check passed.\n")
        else:
            print("\nInput check failed!")
            if len(err_inp_names) != 0:
                err_msg_inp += "No matching input name from the list file for input layer(s): " \
                           + (", ".join(str(e) for e in err_inp_names) + "\n")
            if len(err_inp_shapes) != 0:
                err_msg_inp += "Shapes of layers and labels do not match for the following input layer(s): " \
                           + (", ".join(str(e) for e in err_inp_shapes) + "\n")
            print("Error:", err_msg_inp)

        # ----------------------------------
        print("\nOutput check\n------------")
        # tuple of strings
        mc_names = y_values.dtype.names
        print("The following {} label names are in your toml list file:".format(len(mc_names)))
        print("\t" + ", ".join(str(name) for name in mc_names), end="\n\n")

        if self.cfg.label_modifier is not None:
            label_names = tuple(self.cfg.label_modifier(y_values).keys())
            print("The following {} labels get produced from them by your label_modifier:".format(len(label_names)))
            print("\t" + ", ".join(str(name) for name in label_names), end="\n\n")
        else:
            label_names = mc_names
            print("You did not specify a label_modifier. The output layers will be provided with "
                  "labels that match their name from the above.\n\n")

        # tuple of strings
        loss_names = tuple(model.output_names)
        print("Your model has the following {} output layers:".format(len(loss_names)))
        print("\t" + ", ".join(str(name) for name in loss_names), end="\n\n")

        err_out_names = []
        for loss_name in loss_names:
            if loss_name not in label_names:
                err_out_names.append(loss_name)

        err_msg_out = ""
        if len(err_out_names) == 0:
            print("Output check passed.\n")
        else:
            print("Output check failed!")
            if len(err_out_names) != 0:
                err_msg_out += "No matching label name from the list file for output layer(s): " \
                           + (", ".join(str(e) for e in err_out_names) + "\n")
            print("Error:", err_msg_out)

        err_msg = err_msg_inp + err_msg_out
        if err_msg != "":
            raise AssertionError(err_msg)

    def get_generator(self, mc_info=False):
        """
        For testing purposes: Return a generator reading from the first training file.

        Returns
        -------
        generator : generator object
            Yields tuples with a batch of samples and labels each.

        """
        files_dict = next(self.yield_files("train"))
        generator = generate_batches_from_hdf5_file(self, files_dict, yield_mc_info=mc_info)
        return generator

    def get_batch(self):
        """ For testing purposes, return a batch of samples and mc_infos. """
        files_dict = next(self.yield_files("train"))
        xs = {}
        for i, inp_name in enumerate(files_dict):
            with h5py.File(files_dict[inp_name], "r") as f:
                xs[inp_name] = f[self.cfg.key_samples][:self.cfg.batchsize]
                if i == 0:
                    mc_info = f[self.cfg.key_labels][:self.cfg.batchsize]
        return xs, mc_info


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


def write_full_logfile_startup(orca):
    """
    Whenever the orca_train function is run, this logs all the input parameters in the full log file.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.

    """
    logfile = orca.cfg.output_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('----------------------------------'+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'---------------------------------------------------\n\n')
        f_out.write("New execution of the orca_train function started with the following options:\n\n")
        f_out.write("List file path:\t"+orca.cfg.get_list_file()+"\n")

        f_out.write("Given trainfiles in the .list file:\n")
        for input_name, input_files in orca.cfg.get_files("train").items():
            f_out.write(input_name + ":")
            [f_out.write("\t" + input_file + "\n") for input_file in input_files]

        f_out.write("Given validation files in the .list file:\n")
        for input_name, input_files in orca.cfg.get_files("val").items():
            f_out.write(input_name + ":")
            [f_out.write("\t" + input_file + "\n") for input_file in input_files]

        f_out.write("\nConfiguration used:\n")
        for key in vars(orca.cfg):
            if not key.startswith("_"):
                f_out.write("   {}:\t{}\n".format(key, getattr(orca.cfg, key)))

        f_out.write("\n")


def write_full_logfile(orca, model, history_train, history_val, lr, epoch, files_dict):
    """
    Function for saving various information during training and validation to a .txt file.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
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
    logfile = orca.cfg.output_folder + 'full_log.txt'
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


def write_summary_logfile(orca, epoch, model, history_train, history_val, lr):
    """
    Write to the summary.txt file in every trained model folder.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.
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
    for f_size in orca.io.get_file_sizes("train"):
        steps_per_file = int(f_size / orca.cfg.batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)
    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)

    # Get the widths of the columns. They depend on the widths of the metric names in the first line
    data = ["Epoch", "LR", ]
    for i, metric_name in enumerate(model.metrics_names):
        data.append("train_" + str(metric_name))
        data.append("val_" + str(metric_name))
    headline, widths = get_summary_log_line(data)

    logfile_fname = orca.cfg.output_folder + 'summary.txt'
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


def read_logfiles(orca):
    """
    Read out the data from the summary.txt file, and from all training log files in the log_train folder which
    is in the same directory as the summary.txt file.

    Parameters
    ----------
    orca : object OrcaHandler
        Contains all the configurable options in the OrcaNet scripts.

    Returns
    -------
    summary_data : numpy.ndarray
        Structured array containing the data from the summary.txt file.
    full_train_data : numpy.ndarray
        Structured array containing the data from all the training log files, merged into a single array.

    """
    summary_data = np.genfromtxt(orca.cfg.output_folder + "/summary.txt", names=True, delimiter="|", autostrip=True, comments="--")

    # list of all files in the log_train folder of this model
    log_train_folder = orca.io.get_subfolder("log_train")
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
