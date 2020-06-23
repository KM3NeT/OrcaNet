#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code regarding user input.
"""

import os
import shutil
import h5py
import numpy as np
from inspect import signature
from orcanet.h5_generator import Hdf5BatchGenerator


def get_subfolder(main_folder, name=None, create=False):
    """
    Get the path to one or all subfolders of the main folder.

    Parameters
    ----------
    main_folder : str
        The main folder.
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
    if not main_folder[-1] == "/":
        main_folder += "/"

    subfolders = {
        "train_log": main_folder + "train_log",
        "saved_models": main_folder + "saved_models",
        "plots": main_folder + "plots",
        "activations": main_folder + "plots/activations",
        "predictions": main_folder + "predictions",
        "inference": main_folder + "predictions/inference"
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


def get_inputs(model):
    """ Get names and keras layers of the inputs of the model, as a dict.
    """
    return {name: model.get_layer(name) for name in model.input_names}


class IOHandler(object):
    """
    Access info indirectly contained in the cfg object.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # copies of files on local tmpdir
        self._tmpdir_files_dict = {
            "train": None,
            "val": None,
            "inference": None,
        }

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
        Get a sorted list of the epoch/fileno pairs of all saved models.

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
        subfolder = get_subfolder(self.cfg.output_folder, name, create)
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

    def get_latest_prediction_file_no(self, epoch, fileno):
        """
        Returns the file number of the latest currently predicted val file.

        Parameters
        ----------
        epoch : int
            Epoch of the model that has predicted.
        fileno : int
            Fileno of the model that has predicted.

        Returns
        -------
        latest_val_file_no : int or None
            File number of the prediction file with the highest val index.
            STARTS FROM 1, so this is whats in the file name.
            None if there is none.

        """
        prediction_folder = self.get_subfolder("predictions", create=True)

        val_file_nos = []

        for file in os.listdir(prediction_folder):
            # name e.g.: pred_model_epoch_6_file_1_on_list_val_file_1.h5
            if not (file.endswith(".h5") and file.startswith("pred_model")):
                continue

            f_epoch, f_fileno, val_file_no = split_name_of_predfile(file)
            if f_epoch == epoch and f_fileno == fileno:
                val_file_nos.append(val_file_no)

        if len(val_file_nos) == 0:
            latest_val_file_no = None
        else:
            latest_val_file_no = max(val_file_nos)

        return latest_val_file_no

    def get_pred_path(self, epoch, fileno, pred_file_no):
        """
        Gets the path of a prediction file. The ints all start from 1.

        Parameters
        ----------
        epoch : int
            Epoch of an already trained nn model.
        fileno : int
            File number train step of an already trained nn model.
        pred_file_no : int
            Val file no of the prediction files that are found in the
            prediction folder.

        Returns
        -------
        pred_filepath : str
            The path.

        """
        list_file = self.cfg.get_list_file()
        if list_file is None:
            raise ValueError("No toml list file specified. Can not look up "
                             "saved prediction")
        list_name = os.path.splitext(os.path.basename(list_file))[0]

        pred_filepath = self.get_subfolder("predictions") + \
            '/pred_model_epoch_{}_file_{}_on_{}_val_file_{}.h5'.format(
                epoch, fileno, list_name, pred_file_no)

        return pred_filepath

    def get_pred_files_list(self, epoch=None, fileno=None):
        """
        Returns a sorted list with all pred .h5 files in the prediction folder.
        Does not include the inference files.

        Parameters
        ----------
        epoch : int, optional
            Specific model epoch to look pred files up for.
        fileno : int, optional
            Specific model epoch to look pred files up for.

        Returns
        -------
        pred_files_list : List
            List with the full filepaths of all prediction results files.

        """
        prediction_folder = self.get_subfolder("predictions")

        pred_files_list = []
        for file in os.listdir(prediction_folder):
            if not (file.startswith("pred_model_epoch") and file.endswith(".h5")):
                continue
            pred_file = os.path.join(prediction_folder, file)
            p_epoch, p_file_no, p_val_file_no = split_name_of_predfile(pred_file)
            if epoch is not None and epoch != p_epoch:
                continue
            if fileno is not None and fileno != p_file_no:
                continue
            pred_files_list.append(pred_file)

        pred_files_list.sort()  # sort predicted val files from 1 ... n
        return pred_files_list

    @staticmethod
    def get_cumulative_number_of_rows(file_list):
        """
        Returns the cumulative number of rows (axis_0) in a list
        based on the specified input .h5 files.

        The folders in each file are expected to have the same number of rows,
        such that the info can be taken from the first folder.

        Parameters
        ----------
        file_list : list
            List that contains the filepaths of the h5 files.

        Returns
        -------
        cum_number_of_rows : List
            List that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows).

        """
        total_number_of_rows = 0
        cum_number_of_rows = [0]

        for file_name in file_list:
            f = h5py.File(file_name, 'r')

            # get number of rows from the first folder of the file -> each folder needs to have the same number of rows
            f_keys = list(f.keys())
            total_number_of_rows += f[f_keys[0]].shape[0]
            cum_number_of_rows.append(total_number_of_rows)

            f.close()

        return cum_number_of_rows

    def concatenate_pred_files(self, concatenated_folder):
        """ Concatenates prediction files to a single one. """
        if not os.path.isdir(concatenated_folder):
            os.mkdir(concatenated_folder)

        pred_files_list = self.get_pred_files_list()

        # get cumulative number of rows in the pred files
        cum_number_of_rows = self.get_cumulative_number_of_rows(pred_files_list)

        basename = os.path.basename(pred_files_list[-1])  # take one pred filepath, doesnt matter which one so use -1
        pred_filepath_conc = concatenated_folder + '/' + basename.split('_val_file_')[0] + '_all_val_files.h5'

        f_conc = h5py.File(pred_filepath_conc, 'w')
        for i, pred_file in enumerate(pred_files_list):
            print('Concatenating file ' + str(i) + ' of ' + str(len(pred_files_list)))

            f = h5py.File(pred_file, 'r')
            for folder_name in f:
                folder_data = f[folder_name]

                if i == 0:
                    # first file; create the dataset with no max shape
                    maxshape = (None,) + folder_data.shape[1:]  # change shape of axis zero to None
                    chunks = True
                    compression, compression_opts = folder_data.compression, folder_data.compression_opts

                    output_dataset = f_conc.create_dataset(folder_name, data=folder_data, maxshape=maxshape, chunks=chunks,
                                                           compression=compression, compression_opts=compression_opts)

                    output_dataset.resize(cum_number_of_rows[-1], axis=0)

                else:
                    f_conc[folder_name][cum_number_of_rows[i]:cum_number_of_rows[i + 1]] = folder_data

            f_conc.flush()

        f_conc.close()

        return pred_filepath_conc

    def get_local_files(self, which):
        """
        Get the training or validation file paths for each list input set.

        Returns the path to the copy of the file on the local tmpdir, which
        it will generate if called for the first time.

        Parameters
        ----------
        which : str
            Either "train", "val", or "inference".

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
        if which not in self._tmpdir_files_dict.keys():
            raise NameError("Unknown fileset name ", which)

        files = self.cfg.get_files(which)
        if self.cfg.use_scratch_ssd:
            if self._tmpdir_files_dict[which] is None:
                self._tmpdir_files_dict[which] = use_local_tmpdir(files)
            return self._tmpdir_files_dict[which]
        else:
            return files

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
                n_bins[input_key] = f[self.cfg.key_x_values].shape[1:]
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
                file, datasets=[self.cfg.key_y_values, self.cfg.key_x_values])
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
        info_blob = self.get_batch()
        y_values = info_blob["y_values"]
        layer_inputs = get_inputs(model)
        # keys: name of layers, values: shape of input
        layer_inp_shapes = {key: layer_inputs[key].input_shape[0][1:]
                            for key in layer_inputs}
        list_inp_shapes = self.get_n_bins()

        print("The data in the files of the toml list have the following "
              "names and shapes:")
        for list_key in list_inp_shapes:
            print("\t{}\t{}".format(list_key, list_inp_shapes[list_key]))

        if self.cfg.sample_modifier is None:
            print("\nYou did not specify a sample modifier.")
            info_blob["xs"] = info_blob["x_values"]
        else:
            modified_xs = self.cfg.sample_modifier(info_blob)
            modified_shapes = {modi_key: modified_xs[modi_key].shape[1:]
                               for modi_key in modified_xs}
            print("\nAfter applying your sample modifier, they have the "
                  "following names and shapes:")
            for list_key in modified_shapes:
                print("\t{}\t{}".format(list_key, modified_shapes[list_key]))
            list_inp_shapes = modified_shapes
            info_blob["xs"] = modified_xs

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
                err_msg_inp += "No matching input name from the input files " \
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
        print("The following {} label names are in the first file of the "
              "toml list:".format(len(mc_names)))
        print("\t" + ", ".join(str(name) for name in mc_names), end="\n\n")

        if self.cfg.label_modifier is not None:
            label_names = tuple(self.cfg.label_modifier(info_blob).keys())
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
                err_msg_out += "No matching label name from the input files " \
                               "for output layer(s): " + (
                                ", ".join(str(e) for e in err_out_names) + "\n")
            print("Error:", err_msg_out)

        err_msg = err_msg_inp + err_msg_out
        if err_msg != "":
            raise ValueError(err_msg)

    def get_batch(self):
        """
        For testing purposes, return a batch of x_values and y_values.

        This will always be the first batchsize samples and y_values from
        the first file, before any modifiers have been applied.

        Returns
        -------
        info_blob : dict
            X- and y-values from the files. Has the following entries:
            x_values : dict
                Keys: Names of the input datasets from the list toml file.
                Values: ndarray, a batch of samples.
            y_values : ndarray
                From the y_values datagroup of the input files.

        """
        gen = Hdf5BatchGenerator(
            next(self.yield_files("train")),
            batchsize=self.cfg.batchsize,
            key_x_values=self.cfg.key_x_values,
            key_y_values=self.cfg.key_y_values,
            keras_mode=False,
        )
        info_blob = gen[0]
        info_blob.pop("xs")
        info_blob.pop("ys")
        return info_blob

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
            info_blob = self.get_batch()
            xs_mod = self.cfg.sample_modifier(info_blob)
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

    def get_learning_rate(self, epoch):
        """
        Get the learning rate for a given epoch and file number.

        The user learning rate (cfg.learning_rate) can be None, a float,
        a tuple, or a function.

        Parameters
        ----------
        epoch : tuple
            Epoch and file number. Both start at 1, i.e. the start of the
            training is (1, 1), the next file is (1, 2), ...
            This is also in the filename of the saved models.

        Returns
        -------
        lr : float
            The learning rate that will be used for the given epoch/fileno.

        """
        error_msg = "The learning rate must be either a float, a tuple of " \
                    "two floats or a function."
        no_train_files = self.get_no_of_files("train")
        user_lr = self.cfg.learning_rate

        if isinstance(user_lr, str):
            # read lr from a csv file in the main folder, which must have
            # 3 columns (Epoch, fileno, lr)
            lr_file = os.path.join(self.cfg.output_folder, user_lr)
            lr_table = np.genfromtxt(lr_file)

            if len(lr_table.shape) == 1:
                lr_table = lr_table.reshape((1, ) + lr_table.shape)

            if len(lr_table.shape) != 2 or lr_table.shape[1] != 3:
                raise ValueError("Invalid lr.csv format")
            lr_table = [[tuple(lrt[0:2]), lrt[2]] for lrt in lr_table]
            lr_table.sort()

            lr = None
            # get lr from the table, one line before where the table is bigger
            for table_epoch in lr_table:
                if table_epoch[0] > tuple(epoch):
                    break
                else:
                    lr = table_epoch[1]
            if lr is None:
                raise ValueError(
                    "csv learning rate not specified for epoch {}".format(epoch))
            return lr

        try:
            # Float => Constant LR
            lr = float(user_lr)
            return lr
        except (ValueError, TypeError):
            pass

        try:
            # List => Exponentially decaying LR
            length = len(user_lr)
            lr_init = float(user_lr[0])
            lr_decay = float(user_lr[1])
            if length != 2:
                raise LookupError(
                    "{} (Your tuple has length {})".format(error_msg,
                                                           len(user_lr)))

            lr = lr_init * (1 - lr_decay) ** (
                        (epoch[1] - 1) + (epoch[0] - 1) * no_train_files)
            return lr
        except (ValueError, TypeError):
            pass

        try:
            # Callable => User defined function
            n_params = len(signature(user_lr).parameters)
            if n_params != 2:
                raise TypeError("A custom learning rate function must have two "
                                "input parameters: The epoch and the file number. "
                                "(yours has {})".format(n_params))
            lr = user_lr(epoch[0], epoch[1])
            return lr
        except (ValueError, TypeError):
            raise TypeError("{} (You gave {} of type {}) ".format(
                error_msg, user_lr, type(user_lr)))


def split_name_of_predfile(file):
    """
    Get epoch, fileno, cal fileno from the name of a predfile.

    Parameters
    ----------
    file : str
        Like this: model_epoch_XX_file_YY_on_USERLIST_val_file_ZZ.h5

    Returns
    -------
    epoch , file_no, val_file_no : tuple(int)
        As integers.

    """
    file_base = os.path.splitext(file)[0]
    rest, val_file_no = file_base.split("_val_file_")
    rest, file_no = rest.split("_on_")[0].split("_file_")
    epoch = rest.split("_epoch_")[-1]

    epoch, file_no, val_file_no = map(int, [epoch, file_no, val_file_no])

    return epoch, file_no, val_file_no


def h5_get_number_of_rows(h5_filepath, datasets=None):
    """
    Gets the total number of rows of of a .h5 file.

    Multiple dataset names can be given as a list to check if they all
    have the same number of rows (axis 0).

    Parameters
    ----------
    h5_filepath : str
        filepath of the .h5 file.
    datasets : list
        Optional, The names of datasets in the file to check.

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
        if datasets is None:
            datasets = [x for x in list(f.keys())]

        number_of_rows = [f[dataset].shape[0] for dataset in datasets]
    if not number_of_rows.count(number_of_rows[0]) == len(number_of_rows):
        err_msg = "Datasets do not have the same number of samples " \
                  "in file " + h5_filepath
        for i, dataset in enumerate(datasets):
            err_msg += "\nDataset: {}\tSamples: {}".format(dataset,
                                                           number_of_rows[i])
        raise AssertionError(err_msg)
    return number_of_rows[0]


def use_local_tmpdir(files):
    """
    Copies given files to the local temp folder.

    Parameters
    ----------
    files : dict
        Dict containing the file pathes.

    Returns
    -------
    files_ssd : dict
        Dict with updated SSD/scratch filepaths.

    """
    local_scratch_path = os.environ['TMPDIR']
    files_ssd = {}

    for input_key in files:
        old_pathes = files[input_key]
        new_pathes = []
        for f_path in old_pathes:
            # copy to /scratch node-local SSD
            f_path_ssd = os.path.join(local_scratch_path,
                                      os.path.basename(f_path))
            print("Copying", f_path, "\nto", f_path_ssd)
            shutil.copy2(f_path, local_scratch_path)
            new_pathes.append(f_path_ssd)
        files_ssd[input_key] = tuple(new_pathes)

    print('Finished copying to local tmpdir folder.')
    return files_ssd
