#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code like parsing the command line input or
technical stuff like copying the files to the node-local SSD.

Reading toml files: There are three different keywords:
    "input" :   The input to networks.
    "config" :  The Settings.
    "model" :   Options for auto-generated OrcaNet models.

"""

import os
import shutil
import h5py
import toml
from time import gmtime, strftime
import numpy as np
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
        Values for the OrcaNet scripts, as listed in the Settings class.

    """
    file_content = toml.load(file)["config"]
    if "class_type" in file_content:
        if file_content["class_type"][0] == "None":
            file_content["class_type"][0] = None
    if "n_gpu" in file_content:
        file_content["n_gpu"][0] = int(file_content["n_gpu"][0])
    return file_content


def list_get_number_of_files(file_content, keyword):
    """
    Get the number of training or validation files from the content of a toml list.

    Parameters
    ----------
    file_content : dict
        From a list file by toml.load().
    keyword : str
        Keyword in the file content dictionary to look up, e.g. "train_files" or "validation_files".

    Returns
    -------
    number_of_files : int
        The number of files.

    Raises
    -------
    ValueError
        If different inputs have a different number of files.

    """
    number_of_files = 0
    for dataset_no in range(len(file_content)):
        current_number_of_files = len(file_content[dataset_no][keyword])
        if dataset_no == 0:
            number_of_files = current_number_of_files
        elif current_number_of_files != number_of_files:
            raise ValueError("Error: The specified inputs do not all have the same number of files ("+keyword+")")
    return number_of_files


def list_restructure(number_of_files, keyword, file_content):
    """ Arrange the given files to the desired format. """
    files = []
    for file_no in range(number_of_files):
        file_set = []
        for input_data in file_content:
            file_set.append(input_data[keyword][file_no])
        files.append([file_set, h5_get_number_of_rows(file_set[0])])
        # TODO Maybe files have different number of events? Should give an error
    return files


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
    train_files : list
        A list containing the paths to the different training files given in the list_file.
        Example for the output format:
                [
                 [['path/to/train_file_1_dimx.h5', 'path/to/train_file_1_dimy.h5'], number_of_events_train_files_1],
                 [['path/to/train_file_2_dimx.h5', 'path/to/train_file_2_dimy.h5'], number_of_events_train_files_2],
                 ...
                ]
    validation_files : list
        Like the above but for validation files.
    multiple_inputs : bool
        Whether seperate sets of input files were given (e.g. for networks taking data
        simulataneosly from different files).

    """
    file_content = toml.load(file)["input"]
    number_of_train_files = list_get_number_of_files(file_content, "train_files")
    number_of_val_files = list_get_number_of_files(file_content, "validation_files")
    train_files = list_restructure(number_of_train_files, "train_files", file_content)
    validation_files = list_restructure(number_of_val_files, "validation_files", file_content)
    multiple_inputs = len(file_content) > 1

    return train_files, validation_files, multiple_inputs


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
    cfg : class Settings
        ...

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('--------------------------------------------------------------------------------------------------------\n\n\n')
        f_out.write("New execution of the orca_train function started with the following options:\n")
        f_out.write("List file path:\t"+cfg.list_file+"\n")
        f_out.write("Given trainfiles in the .list file:\n")
        for train_file in cfg.get_train_files():
            f_out.write("   " + str(train_file)+"\n")
        f_out.write("\nGiven validation files in the .list file:\n")
        for val_file in cfg.get_val_files():
            f_out.write("   " + str(val_file) + "\n")
        f_out.write("\nGiven options in the .toml config:\n")
        for key in vars(cfg):
            f_out.write("   {}:\t{}\n".format(key, cfg[key]))
        f_out.write("\n")


def write_full_logfile(cfg, model, history_train, history_val, lr, lr_decay, epoch):
    """
    Function for saving various information during training and validation to a .txt file.

    """
    logfile = cfg.main_folder + 'full_log.txt'
    with open(logfile, 'a+') as f_out:
        f_out.write('---------------Epoch {} File {}-------------------------------------------------------------------------\n'.format(epoch[0], epoch[1]))
        f_out.write('\n')
        f_out.write('Current time: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        f_out.write('Decayed learning rate to ' + str(lr) + ' before epoch ' + str(epoch[0]) +
                    ' and file ' + str(epoch[1]) + ' (minus ' + str(lr_decay) + ')\n')
        f_out.write('Trained in epoch ' + str(epoch) + ' on file ' + str(epoch[1]) + ', ' + str(cfg.train_file) + '\n')
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
    cfg : class Settings
        ...
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
    # Save val log
    steps_per_total_epoch, steps_cum = 0, [0]  # get this for the epoch_number_float in the logfile
    for f, f_size in cfg.get_train_files():
        steps_per_file = int(f_size / cfg.batchsize)
        steps_per_total_epoch += steps_per_file
        steps_cum.append(steps_cum[-1] + steps_per_file)

    epoch_number_float = epoch[0] - (steps_per_total_epoch - steps_cum[epoch[1]]) / float(steps_per_total_epoch)
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
    number_of_rows = f[list(f.keys())[0]].shape[0]
    f.close()
    return number_of_rows


def h5_get_n_bins(train_files):
    """
    Get the number of bins from the training files. Only the first files are looked up, the others should be identical.

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
        n_bins.append(f[list(f.keys())[0]].shape[1:])
    return n_bins


def use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=False):
    """
    Copies the test and train files to the node-local ssd scratch folder and returns the new filepaths of the train and test data.
    Speeds up I/O and reduces RRZE network load.
    :param List train_files: list that contains all train files in tuples (filepath, f_size).
    :param List test_files: list that contains all test files in tuples (filepath, f_size).
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

            train_files_ssd.append((f_paths_train_ssd_temp, f_size))  # f_size of all f_paths should be the same
            f_paths_train_ssd_temp = []

        for file_tuple in test_files:
            input_filepaths, f_size = file_tuple[0], file_tuple[1]

            for f_path in input_filepaths:
                shutil.copy2(f_path, local_scratch_path)  # copy to /scratch node-local SSD
                f_path_ssd = local_scratch_path + '/' + os.path.basename(f_path)
                f_paths_test_ssd_temp.append(f_path_ssd)

            test_files_ssd.append((f_paths_test_ssd_temp, f_size))  # f_size of all f_paths should be the same
            f_paths_test_ssd_temp = []

    else:
        for file_tuple in train_files:
            input_filepath, f_size = file_tuple[0][0], file_tuple[1]

            shutil.copy2(input_filepath, local_scratch_path)  # copy to /scratch node-local SSD
            input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
            train_files_ssd.append(([input_filepath_ssd], f_size))

        for file_tuple in test_files:
            input_filepath, f_size = file_tuple[0][0], file_tuple[1]

            shutil.copy2(input_filepath, local_scratch_path)  # copy to /scratch node-local SSD
            input_filepath_ssd = local_scratch_path + '/' + os.path.basename(input_filepath)
            test_files_ssd.append(([input_filepath_ssd], f_size))

    print('Finished copying the input train/test data to the node-local SSD scratch folder')
    return train_files_ssd, test_files_ssd


class Settings(object):
    """
    Container object for all the configurable options in the OrcaNet scripts.

    Attributes
    ----------
    main_folder : str
        Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
    modeldata : namedtuple
        Optional info only required for building a predefined model with OrcaNet.
        It is set via self.load_from_model_file. [default: None]

        modeldata.nn_arch : str
            Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
        modeldata.loss_opt : tuple(dict, dict/str/None,)
            Tuple that contains 1) the loss_functions and loss_weights as dicts (this is the losses table from the toml file)
            and 2) the metrics.
        modeldata.args : dict
            Keyword arguments for the model generation.

    _train_files : list
        A list containing the paths to the different training files given in the list_file.
        Example for the output format:
                [
                 [['path/to/train_file_1_dimx.h5', 'path/to/train_file_1_dimy.h5'], number_of_events_train_files_1],
                 [['path/to/train_file_2_dimx.h5', 'path/to/train_file_2_dimy.h5'], number_of_events_train_files_2],
                 ...
                ]
    _val_files : list
        Like train_files but for the validation files.
    _multiple_inputs : bool
        Whether seperate sets of input files were given (e.g. for networks taking data
        simulataneosly from different files).

    batchsize : int
        Batchsize that should be used for the training / inferencing of the cnn.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower')
    epochs_to_train : int
        How many new epochs should be trained by running this function. -1 for infinite.
    initial_epoch : int
        The epoch at which the training is supposed to start. 0 means start a new training, >0 mean resume training.
        Can also give -1 to automatically load the most recent epoch found in the main folder.
    initial_fileno : int
        When using multiple files, define the file number at which the training is supposed to start, e.g.
        1 for the first file. If both epoch and fileno are -1, automatically load the most recent file found
        in the main folder.
    n_events : None/int
        For testing purposes. If not the whole .h5 file should be used for training, define the number of events.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    shuffle : tuple(bool, None/int)
        Declares if the training data should be shuffled before the next training epoch [0].
        If the train dataset is too large to be shuffled in place, one can preshuffle them n times before running
        OrcaNet, the number n should then be put into [1].
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    swap_4d_channels : None/str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X, TODO add multi input options
    train_logger_display : int
        How many batches should be averaged for one line in the training log files.
    train_logger_flush : int
        After how many lines the training log file should be flushed. -1 for flush at the end of the epoch only.
    train_verbose : int
        verbose option of keras.model.fit_generator.
    use_scratch_ssd : bool
        Declares if the input files should be copied to the node-local SSD scratch space (only working at Erlangen CC).
    zero_center : bool
        Declares if the input images ('xs') should be zero-centered before training.

    """
    def __init__(self, main_folder, list_file=None, config_file=None):
        """
        Set the attributes of the object.

        Values are loaded from the given files, if provided. Otherwise, default values are used.

        Parameters
        ----------
        main_folder : str
            Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
        list_file : str
            Path to a list file with pathes to all the h5 files that should be used for training and validation.
        config_file : str
            Path to the config file with attributes that are used instead of the default ones.

        """
        # Default settings:
        self.batchsize = 64
        self.class_type = ['None', 'energy_dir_bjorken-y_vtx_errors']
        self.epochs_to_train = -1
        self.initial_epoch = -1
        self.initial_fileno = -1
        self.n_events = None
        self.n_gpu = (1, 'avolkov')
        self.shuffle = (False, None)
        self.str_ident = ''
        self.swap_4d_channels = None
        self.train_logger_display = 100
        self.train_logger_flush = -1
        self.train_verbose = 2
        self.use_scratch_ssd = False
        self.zero_center = False

        self._default_values = dict(self.__dict__)
        self.modeldata = None

        # IO attributes:
        if main_folder[-1] == "/":
            self.main_folder = main_folder
        else:
            self.main_folder = main_folder+"/"

        self._list_file = None
        self._train_files = None
        self._val_files = None
        self._multiple_inputs = None

        if list_file is not None:
            self.set_from_list_file(list_file)

        self._config_file = None
        if config_file is not None:
            self.set_from_config_file(config_file)

    def set_from_list_file(self, list_file):
        """ Set filepaths to the ones given in a list file. """
        if self._list_file is None:
            self._train_files, self._val_files, self._multiple_inputs = read_out_list_file(list_file)
            # Save internally which path was used to load the info
            self._list_file = list_file
        else:
            raise ValueError("You tried to load filepathes from a list file, but pathes have already been loaded \
            for this object. (From the file " + self._list_file + ")\nYou should not use \
            two different list files for one Settings object!")
        # self.n_bins = h5_get_n_bins(self.train_files)

    def set_from_config_file(self, config_file):
        """ Overwrite default attribute values with values from a config file. """
        if self._config_file is None:
            user_values = read_out_config_file(config_file)
            for key in user_values:
                if hasattr(self, key):
                    setattr(self, key, user_values[key])
                else:
                    raise AttributeError("You tried to set the attribute "+str(key)+" in your config file\n"
                                         + config_file + "\n, but this attribute is not provided. Check \
                                         the possible attributes in the definition of the Settings class.")
            # Save internally which path was used to load the info
            self._config_file = config_file
        else:
            raise ValueError("You tried to load settings from a config file, but they have already been loaded \
            for this object! (From the file " + self._config_file + ")\nYou should not use \
            two different config files for one Settings object!")

    def set_from_model_file(self, model_file):
        """ Set attributes for generating models with OrcaNet. """
        nn_arch, loss_opt, args = read_out_model_file(model_file)
        ModelData = namedtuple("ModelData", "nn_arch loss_opt args")
        data = ModelData(nn_arch, loss_opt, args)
        self.modeldata = data

    def get_default_values(self):
        """ Return default values of common settings. """
        return self._default_values

    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_multiple_inputs(self):
        # TODO Remove this attribute and make it a function instead
        return self._multiple_inputs

    def get_latest_epoch(self):
        """
        Check all saved models in the ./saved_models folder and return the highest epoch / file_no pair.

        Returns
        -------
        latest_epoch : tuple
            The highest epoch, file_no pair. (0,1) if the folder is empty.

        """
        files = os.listdir(self.main_folder + "saved_models")
        if len(files) == 0:
            latest_epoch = (0, 1)
        else:
            epochs = []
            for file in files:
                epoch, file_no = file.split("model_epoch_")[-1].split(".h5")[0].split("_file_")
                epochs.append((int(epoch), int(file_no)))
            latest_epoch = max(epochs)
        return latest_epoch

