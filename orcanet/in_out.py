#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code regarding reading user input, and writing output like logfiles.
"""

import os
import shutil
import h5py
import numpy as np

from orcanet.utilities.nn_utilities import get_inputs
from orcanet.utilities.visualization import plot_history


class IOHandler(object):
    """
    Access info indirectly contained in the cfg object.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        self._used_ssd = False
        self._tmpdir_train_files = None
        self._tmpdir_val_files = None

    def get_latest_epoch(self):
        """
        Return the highest epoch/fileno pair of any saved model.

        Returns
        -------
        latest_epoch : tuple or None
            The highest epoch, file_no pair. None if the folder is
            empty or does not exist yet.

        """
        epochs = self.get_all_epochs()
        if len(epochs) == 0:
            latest_epoch = None
        else:
            latest_epoch = epochs[-1]

        return latest_epoch

    def get_all_epochs(self):
        """
        Get a sorted list of all existing epoch/fileno pairs.

        Returns
        -------
        epochs : List
            The (epoch, fileno) tuples. List is empty if none can be found.
        """
        saved_models_folder = self.cfg.output_folder + "saved_models"
        epochs = []

        if os.path.exists(saved_models_folder):
            files = []
            for file in os.listdir(saved_models_folder):
                if file.startswith("model_epoch_") and file.endswith('.h5'):
                    files.append(file)

            for file in files:
                # model_epoch_XX_file_YY
                file_base = os.path.splitext(file)[0]
                f_epoch, file_no = file_base.split(
                    "model_epoch_")[-1].split("_file_")
                epochs.append((int(f_epoch), int(file_no)))
            epochs.sort()

        return epochs

    def get_next_epoch(self, epoch):
        """
        Return the next epoch / fileno tuple.

        It depends on how many train files there are.

        Parameters
        ----------
        epoch : tuple or None
            Current epoch and file number.

        Returns
        -------
        next_epoch : tuple
            Next epoch and file number.

        """
        if epoch is None:
            next_epoch = (1, 1)
        elif epoch[1] == self.get_no_of_files("train"):
            next_epoch = (epoch[0] + 1, 1)
        else:
            next_epoch = (epoch[0], epoch[1] + 1)
        return next_epoch

    def get_previous_epoch(self, epoch):
        """ Return the previous epoch / fileno tuple. """
        if epoch[1] == 1:
            if epoch[0] == 1:
                raise ValueError("Can not get previous epoch of epoch {} file {}".format(*epoch))
            n_train_files = self.get_no_of_files("train")
            prev_epoch = (epoch[0]-1, n_train_files)

        else:
            prev_epoch = (epoch[0], epoch[1]-1)

        return prev_epoch

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
            The path of the subfolder. If name is None, all subfolders
            will be returned as a tuple.

        """
        subfolders = {
            "train_log": self.cfg.output_folder + "train_log",
            "saved_models": self.cfg.output_folder + "saved_models",
            "plots": self.cfg.output_folder + "plots",
            "activations": self.cfg.output_folder + "plots/activations",
            "predictions": self.cfg.output_folder + "predictions",
        }

        def get(fdr):
            subfdr = subfolders[fdr]
            if create and not os.path.exists(subfdr):
                print("Creating directory: " + subfdr)
                os.makedirs(subfdr)
            return subfdr

        if name is None:
            subfolder = [get(name) for name in subfolders]
        else:
            subfolder = get(name)
        return subfolder

    def get_model_path(self, epoch, fileno, local=False):
        """
        Get the path to a model (which might not exist yet).

        Parameters
        ----------
        epoch : int
            Its epoch.
        fileno : int
            Its file number.
        local : bool
            If True, will only return the path inside the output_folder,
            i.e. models/models_epochXX_file_YY.h5.

        Returns
        -------
        model_path : str
            The path to the model.
        """
        """ 
        
        """
        if epoch == -1 and fileno == -1:
            epoch, fileno = self.get_latest_epoch()
        if epoch < 1 or fileno < 1:
            raise ValueError("Invalid epoch/file number {}, {}: Must be "
                             "either (-1, -1) or both >0".format(epoch, fileno))

        subfolder = self.get_subfolder("saved_models")
        if local:
            subfolder = subfolder.split("/")[-1]
        file_name = 'model_epoch_{}_file_{}.h5'.format(epoch, fileno)

        model_path = subfolder + "/" + file_name
        return model_path

    def get_pred_path(self, epoch, fileno):
        """ Get the path to a saved prediction. """
        list_file = self.cfg.get_list_file()
        if list_file is None:
            raise ValueError("No tom list file specified. Can not look up "
                             "saved prediction")
        list_name = os.path.splitext(os.path.basename(list_file))[0]

        pred_filename = self.get_subfolder("predictions") + \
            '/pred_model_epoch_{}_file_{}_on_{}_val_files.h5'.format(
                epoch, fileno, list_name)

        return pred_filename

    def use_local_node(self):
        """
        Copies the test and val files to the node-local ssd scratch folder
        and sets the new filepaths of the train and val data.
        Speeds up I/O and reduces RRZE network load.
        """
        if not self._used_ssd:
            train_files_ssd, val_files_ssd = use_local_tmpdir(
                self.cfg.get_files("train"), self.cfg.get_files("val"))
            self._tmpdir_train_files = train_files_ssd
            self._tmpdir_val_files = val_files_ssd
            self._used_ssd = True

    def get_local_files(self, which):
        """
        Get the training or validation file paths for each list input set.

        Returns the path to the copy of the file on the local tmpdir, if
        it has been made.

        Returns
        -------
        dict
            A dict containing the paths to the training or validation files on
            which the model will be trained on. Example for the format for
            two input sets with two files each:
            {
             "input_A" : ('path/to/set_A_file_1.h5', 'path/to/set_A_file_2.h5'),
             "input_B" : ('path/to/set_B_file_1.h5', 'path/to/set_B_file_2.h5'),
            }

        """
        if which == "train":
            if not self._used_ssd:
                return self.cfg.get_files("train")
            else:
                return self._tmpdir_train_files

        elif which == "val":
            if not self._used_ssd:
                return self.cfg.get_files("val")
            else:
                return self._tmpdir_val_files
        else:
            raise NameError("Unknown fileset name ", which)

    def get_n_bins(self):
        """
        Get the number of bins from the training files.

        Only the first files are looked up, the others should be identical.

        Returns
        -------
        n_bins : dict
            Toml-list input names as keys, list of the bins as values.

        """
        # TODO check if bins are equal in all files?
        train_files = self.get_local_files("train")
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
        file_sizes : List
            Its length is equal to the number of files in each input set.

        Raises
        ------
        ValueError
            If there is a different number of samples in any of the
            files of all inputs.

        """
        file_sizes_full, error_file_sizes, file_sizes = {}, [], []
        for n, file_no_set in enumerate(self.yield_files(which)):
            # the number of samples in the n-th file of all inputs
            file_sizes_full[n] = [h5_get_number_of_rows(
                file, datasets=[self.cfg.key_mc_info, self.cfg.key_samples])
                for file in file_no_set.values()]
            if not file_sizes_full[n].count(file_sizes_full[n][0]) == \
                    len(file_sizes_full[n]):
                    error_file_sizes.append(n)
            else:
                file_sizes.append(file_sizes_full[n][0])

        if len(error_file_sizes) != 0:
            err_msg = "The files you gave for the different inputs of the model " \
                      "do not all have the same number of samples!\n"
            for n in error_file_sizes:
                err_msg += "File no {} in {} has the following files sizes " \
                           "for the different inputs: {}\n".format(
                            n, which, file_sizes_full[n])
            raise ValueError(err_msg)

        return file_sizes

    def get_no_of_files(self, which):
        """
        Return the number of training or validation files.

        Only looks up the no of files of one (random) list input, as equal
        length is checked during read in.

        Parameters
        ----------
        which : str
            Either train or val.

        Returns
        -------
        no_of_files : int
            The number of files.

        """
        files = self.get_local_files(which)
        no_of_files = len(list(files.values())[0])
        return no_of_files

    def yield_files(self, which):
        """
        Yield a training or validation filepaths for every input.

        They will be yielded in the same order as they are given in the
        toml file.

        Parameters
        ----------
        which : str
            Either train or val.

        Yields
        ------
        files_dict : dict
            Keys: The name of every toml list input.
            Values: One of the filepaths.

        """
        files = self.get_local_files(which)
        for file_no in range(self.get_no_of_files(which)):
            files_dict = {key: files[key][file_no] for key in files}
            yield files_dict

    def get_file(self, which, file_no):
        """ Get a dict with the n-th files. """
        files = self.get_local_files(which)
        files_dict = {key: files[key][file_no-1] for key in files}
        return files_dict

    def check_connections(self, model):
        """
        Check if the names and shapes of the samples and labels in the
        given input files work with the model.

        Also takes into account the possibly present sample or label modifiers.

        Parameters
        ----------
        model : ks.model
            A keras model.

        Raises
        ------
        ValueError
            If they dont work together.

        """
        print("\nInput check\n-----------")
        # Get a batch of data to investigate the given modifier functions
        xs, y_values = self.get_batch()
        layer_inputs = get_inputs(model)
        # keys: name of layers, values: shape of input
        layer_inp_shapes = {key: layer_inputs[key].input_shape[1:]
                            for key in layer_inputs}
        list_inp_shapes = self.get_n_bins()

        print("The inputs in your toml list file have the following "
              "names and shapes:")
        for list_key in list_inp_shapes:
            print("\t{}\t{}".format(list_key, list_inp_shapes[list_key]))

        if self.cfg.sample_modifier is None:
            print("\nYou did not specify a sample modifier.")
        else:
            modified_xs = self.cfg.sample_modifier(xs)
            modified_shapes = {modi_key: modified_xs[modi_key].shape[1:]
                               for modi_key in modified_xs}
            print("\nAfter applying your sample modifier, they have the "
                  "following names and shapes:")
            for list_key in modified_shapes:
                print("\t{}\t{}".format(list_key, modified_shapes[list_key]))
            list_inp_shapes = modified_shapes

        print("\nYour model requires the following input names and shapes:")
        for layer_key in layer_inp_shapes:
            print("\t{}\t{}".format(layer_key, layer_inp_shapes[layer_key]))

        # Both inputs are dicts with  name: shape  of input/output layers/data
        err_inp_names, err_inp_shapes = [], []
        for layer_name in layer_inp_shapes:
            if layer_name not in list_inp_shapes.keys():
                # no matching name
                err_inp_names.append(layer_name)
            elif list_inp_shapes[layer_name] != layer_inp_shapes[layer_name]:
                # no matching shape
                err_inp_shapes.append(layer_name)

        err_msg_inp = ""
        if len(err_inp_names) == 0 and len(err_inp_shapes) == 0:
            print("\nInput check passed.")
        else:
            print("\nInput check failed!")
            if len(err_inp_names) != 0:
                err_msg_inp += "No matching input name from the list file " \
                               "for input layer(s): " + (
                                ", ".join(str(e) for e in err_inp_names) + "\n")
            if len(err_inp_shapes) != 0:
                err_msg_inp += "Shapes of layers and labels do not match for " \
                               "the following input layer(s): " + (
                                ", ".join(str(e) for e in err_inp_shapes) + "\n")
            print("Error:", err_msg_inp)

        # ----------------------------------
        print("\nOutput check\n------------")
        # tuple of strings
        mc_names = y_values.dtype.names
        print("The following {} label names are in your toml list file:".format(
            len(mc_names)))
        print("\t" + ", ".join(str(name) for name in mc_names), end="\n\n")

        if self.cfg.label_modifier is not None:
            label_names = tuple(self.cfg.label_modifier(y_values).keys())
            print("The following {} labels get produced from them by your "
                  "label_modifier:".format(len(label_names)))
            print("\t" + ", ".join(str(name) for name in label_names), end="\n\n")
        else:
            label_names = mc_names
            print("You did not specify a label_modifier. The output layers "
                  "will be provided with labels that match their name from "
                  "the above.\n\n")

        # tuple of strings
        loss_names = tuple(model.output_names)
        print("Your model has the following {} output layers:".format(
            len(loss_names)))
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
                err_msg_out += "No matching label name from the list file " \
                               "for output layer(s): " + (
                                ", ".join(str(e) for e in err_out_names) + "\n")
            print("Error:", err_msg_out)

        err_msg = err_msg_inp + err_msg_out
        if err_msg != "":
            raise ValueError(err_msg)

    def get_batch(self):
        """
        For testing purposes, return a batch of samples and mc_infos.

        This will always be the first batchsize samples and mc_info from
        the first file, before any modifiers have been applied.

        Returns
        -------
        xs : dict
            Keys: Names of the input datasets from the list toml file.
            Values: ndarray, a batch of samples.
        mc_info : ndarray
            From the mc_info datagroup of the input files.

        """
        # TODO gets mc_info only from first train file
        files_dict = next(self.yield_files("train"))
        xs = {}
        for i, inp_name in enumerate(files_dict):
            with h5py.File(files_dict[inp_name], "r") as f:
                xs[inp_name] = f[self.cfg.key_samples][:self.cfg.batchsize]
                if i == 0:
                    mc_info = f[self.cfg.key_mc_info][:self.cfg.batchsize]
        return xs, mc_info

    def get_input_shapes(self):
        """
        Get the input names and shapes of the data after the modifier has
        been applied.

        Returns
        -------
        input_shapes : dict
            Keys: Name of the inputs of the model.
            Values: Their shape without the batchsize.

        """
        if self.cfg.sample_modifier is None:
            input_shapes = self.get_n_bins()
        else:
            xs, mc_info = self.get_batch()
            xs_mod = self.cfg.sample_modifier(xs)
            input_shapes = {input_name: input_xs.shape[1:]
                            for input_name, input_xs in xs_mod.items()}
        return input_shapes

    def print_log(self, lines, logging=True):
        """ Print and also log to the full log file. """
        if isinstance(lines, str):
            lines = [lines, ]

        if not logging:
            for line in lines:
                print(line)
        else:
            full_log_file = self.cfg.output_folder + 'log.txt'
            with open(full_log_file, 'a+') as f_out:
                for line in lines:
                    f_out.write(line + "\n")
                    print(line)

    def get_epoch_float(self, epoch, fileno):
        """ Make a float value out of epoch/fileno. """
        # calculate the fraction of samples per file compared to all files,
        # e.g. [100, 50, 50] --> [0.5, 0.75, 1]
        file_sizes = self.get_file_sizes("train")
        file_sizes_rltv = np.cumsum(file_sizes) / np.sum(file_sizes)

        epoch_float = epoch - 1 + file_sizes_rltv[fileno - 1]
        return epoch_float


class HistoryHandler:
    """
    For reading and plotting data from summary and train log files.

    """
    def __init__(self, summary_file, train_log_folder):
        self.summary_filename = summary_file
        self.train_log_folder = train_log_folder

    def plot_metric(self, metric_name, **kwargs):
        """
        Plot the training and validation history of a metric.

        This will read out data from the summary file, as well as
        all training log files, and plot them over the epoch.

        Parameters
        ----------
        metric_name : str
            Name of the metric to be plotted over the epoch. This name is what
            was written in the head line of the summary.txt file, except without
            the train_ or val_ prefix.
        kwargs
            Keyword arguments for the plot_history function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The plot.

        """
        summary_data = self.get_summary_data()
        full_train_data = self.get_train_data()
        summary_label = "val_" + metric_name

        if metric_name not in full_train_data.dtype.names:
            raise ValueError(
                "Train log metric name {} unknown, must be one of {}".format(
                    metric_name, self.get_metrics()))
        if summary_label not in summary_data.dtype.names:
            raise ValueError(
                "Summary metric name {} unknown, must be one of {}".format(
                    summary_label, self.get_metrics()))

        if summary_data["Epoch"].shape == (0,):
            # When no lines are present in the summary.txt file.
            val_data = None
        else:
            val_data = [summary_data["Epoch"], summary_data[summary_label]]

        if full_train_data["Batch_float"].shape == (0,):
            # When no lines are present
            raise ValueError("Can not make summary plot: Training log files "
                             "contain no data!")
        else:
            train_data = [full_train_data["Batch_float"],
                          full_train_data[metric_name]]

        # if no validation has been done yet
        if np.all(np.isnan(val_data)[1]):
            val_data = None

        if "y_label" not in kwargs:
            kwargs["y_label"] = metric_name

        fig = plot_history(train_data, val_data, **kwargs)
        return fig

    def plot_lr(self, **kwargs):
        """
        Plot the learning rate over the epochs.

        Parameters
        ----------
        kwargs
            Keyword arguments for the plot_history function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The plot.

        """
        summary_data = self.get_summary_data()

        epoch = summary_data["Epoch"]
        lr = summary_data["LR"]
        # plot learning rate like val data (connected dots)
        val_data = (epoch, lr)

        if "y_label" not in kwargs:
            kwargs["y_label"] = "Learning rate"
        if "legend" not in kwargs:
            kwargs["legend"] = False

        fig = plot_history(train_data=None, val_data=val_data, **kwargs)
        return fig

    def get_metrics(self):
        """
        Get the name of the metrics from the first line in the summary file.

        This will be the actual name of the metric, i.e. "loss" and not
        "train_loss" or "val_loss".

        Returns
        -------
        all_metrics : List
            A list of the metrics.

        """
        summary_data = self.get_summary_data()
        all_metrics = []
        for keyword in summary_data.dtype.names:
            if keyword == "Epoch" or keyword == "LR":
                continue
            if "train_" in keyword:
                keyword = keyword.split("train_")[-1]
            else:
                keyword = keyword.split("val_")[-1]
            if keyword not in all_metrics:
                all_metrics.append(keyword)
        return all_metrics

    def get_summary_data(self):
        """
        Read out the summary file in the output folder.

        Returns
        -------
        summary_data : ndarray
            Numpy structured array with the column names as datatypes.
            Its shape is the number of lines with data.

        """
        summary_data = self._load_txt(self.summary_filename)
        if summary_data.shape == ():
            # When only one line is present
            summary_data = summary_data.reshape(1,)
        return summary_data

    def get_column_names(self):
        """
        Get the str in the first line in each column.

        Returns
        -------
        tuple : column_names
            The names in the same order as they appear in the summary.txt.

        """
        summary_data = self.get_summary_data()
        column_names = summary_data.dtype.names
        return column_names

    def get_train_data(self):
        """
        Read out all training logfiles in the output folder.

        Read out the data from the summary.txt file, and from all training
        log files in the train_log folder, which is in the same directory
        as the summary.txt file.

        Returns
        -------
        summary_data : numpy.ndarray
            Structured array containing the data from the summary.txt file.
            Its shape is the number of lines with data.

        """
        # list of all files in the train_log folder of this model
        files = os.listdir(self.train_log_folder)
        train_file_data = []
        for file in files:
            if not (file.startswith("log_epoch_") and file.endswith(".txt")):
                continue
            # file is sth like "log_epoch_1_file_2.txt", extract the 1 and 2:
            epoch, file_no = [int(file.split(".")[0].split("_")[i]) for i in [2, 4]]
            file_data = self._load_txt(self.train_log_folder + "/" + file)
            train_file_data.append([[epoch, file_no], file_data])

        # sort so that earlier epochs come first
        train_file_data.sort()
        full_train_data = train_file_data[0][1]
        for [epoch, file_no], file_data in train_file_data[1:]:
            full_train_data = np.append(full_train_data, file_data)

        if full_train_data.shape == ():
            # When only one line is present
            full_train_data = full_train_data.reshape(1,)
        return full_train_data

    def get_state(self):
        """
        Get the state of a training.

        For every line in the summary logfile, get a dict with the epoch
        as a float, and is_trained and is_validated bools.

        Returns
        -------
        state_dicts : List
            List of dicts.

        """
        summary_data = self.get_summary_data()
        state_dicts = []
        names = summary_data.dtype.names

        for line in summary_data:
            val_losses, train_losses = {}, {}
            for name in names:
                if name.startswith("val_"):
                    val_losses[name] = line[name]
                elif name.startswith("train_"):
                    train_losses[name] = line[name]
                elif name not in ["Epoch", "LR"]:
                    raise NameError(
                        "Invalid summary file: Invalid column name {}: must be "
                        "either Epoch, LR, or start with val_ or train_".format(name))

            n_nans_val = np.count_nonzero(np.isnan(list(val_losses.values())))
            n_nans_train = np.count_nonzero(np.isnan(list(train_losses.values())))

            if n_nans_val == 0:
                is_val = True
            elif n_nans_val == len(val_losses):
                is_val = False
            else:
                raise ValueError(
                    "Invalid summary file: Expected val losses to be either only "
                    "nans, or no nans (got {})".format(val_losses))

            if n_nans_train == 0:
                is_trained = True
            elif n_nans_train == len(train_losses):
                is_trained = False
            else:
                raise ValueError(
                    "Invalid summary file: Expected train losses to be either only "
                    "nans, or no nans (got {})".format(train_losses))

            line_state = {"epoch": line["Epoch"],
                          "is_trained": is_trained,
                          "is_validated": is_val,}
            state_dicts.append(line_state)

        return state_dicts

    @staticmethod
    def _load_txt(filepath):
        file_data = np.genfromtxt(
            filepath,
            names=True,
            delimiter="|",
            autostrip=True,
            comments="--",
            missing_values="n/a",
            filling_values=np.nan
        )
        return file_data


def h5_get_number_of_rows(h5_filepath, datasets):
    """
    Gets the total number of rows of of a .h5 file.

    Multiple dataset names can be given as a list to check if they all
    have the same number of rows (axis 0).

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
        err_msg = "Datasets do not have the same number of samples " \
                  "in file " + h5_filepath
        for i, dataset in enumerate(datasets):
            err_msg += "\nDataset: {}\tSamples: {}".format(dataset,
                                                           number_of_rows[i])
        raise AssertionError(err_msg)
    return number_of_rows[0]


def use_local_tmpdir(train_files, val_files):
    """
    Copies the val and train files to the local temp folder.

    Returns the new filepaths of the train and val data.
    Speeds up I/O and reduces network load.

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
        train_files_ssd[input_key] = tuple(new_pathes)

    for input_key in val_files:
        old_pathes = val_files[input_key]
        new_pathes = []
        for f_path in old_pathes:
            # copy to /scratch node-local SSD
            f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
            print("Copying", f_path, "\nto", f_path_ssd)
            shutil.copy2(f_path, local_scratch_path)
            new_pathes.append(f_path_ssd)
        val_files_ssd[input_key] = tuple(new_pathes)

    print('Finished copying the input data to the local tmpdir folder.')
    return train_files_ssd, val_files_ssd
