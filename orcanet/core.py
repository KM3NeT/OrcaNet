#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import warnings
from collections import namedtuple
import keras as ks

from orcanet.run_nn import train_and_validate_model
from orcanet.eval_nn import predict_and_investigate_model_performance, get_modelname
from orcanet.utilities.input_output_utilities import read_out_list_file, read_out_config_file, read_out_model_file, use_node_local_ssd_for_input, h5_get_n_bins, write_full_logfile_startup
from orcanet.utilities.nn_utilities import load_zero_center_data
from orcanet.utilities.losses import get_all_loss_functions


class Configuration(object):
    """
    Container object for all the configurable options in the OrcaNet scripts.

    Sensible default values were chosen for the settings.
    You can change the all of these public attributes (the ones without a leading underscore _) either directly or with a
    .toml config file via the method set_from_config_file().

    Attributes
    ----------
    main_folder : str
        Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
        Has a '/' at the end.
    batchsize : int
        Batchsize that should be used for the training / inferencing of the cnn.
    class_type : tuple(int, str)
        Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
        I.e. (2, 'track-shower')
    filter_out_tf_garbage : bool
        If true, surpresses the tensorflow info logs which usually spam the terminal.
    epochs_to_train : int
        How many new epochs should be trained by running this function. -1 for infinite.
    initial_epoch : int
        The epoch of the model with which the training is supposed to start, e.g. 1 means load the saved model from
        epoch 1 and continue training. 0 means start a new training (initial_fileno also has to be 0 for this).
        Can also give -1 to automatically load the most recent epoch found in the main folder, if present, or make
        a new model otherwise.
    initial_fileno : int
        When using multiple files, define the file number with which the training is supposed to start, e.g.
        1 for load the model trained on the first file. If both epoch and fileno are -1, automatically set to the most
        recent file found in the main folder.
    learning_rate : float or tuple or function
        The learning rate for the training.
        If it is a float, the learning rate will be constantly this value.
        If it is a tuple of two floats, the first float gives the learning rate in epoch 1 file 1, and the second
        float gives the decrease of the learning rate per file (e.g. 0.1 for 10% decrease per file).
        You can also give an arbitrary function, which takes as an input the epoch, the file number and the
        Configuration object (in this order), and returns the learning rate.
    n_events : None or int
        For testing purposes. If not the whole .h5 file should be used for training, define the number of events.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    str_ident : str
        Optional string identifier that gets appended to the modelname. Useful when training models which would have
        the same modelname. Also used for defining models and projections!
    swap_4d_channels : None or str
        For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
        Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X, TODO add multi input options
    train_logger_display : int
        How many batches should be averaged for one line in the training log files.
    train_logger_flush : int
        After how many lines the training log file should be flushed (updated on the disk).
        -1 for flush at the end of the file only.
    use_scratch_ssd : bool
        Declares if the input files should be copied to the node-local SSD scratch space (only working at Erlangen CC).
    validate_after_n_train_files : int
        Validate the model after this many training files have been trained on in an epoch, starting from the first.
        E.g. if validate_after_n_train_files == 3, validation will happen after file 1,4,7,...
    verbose_train : int
        verbose option of keras.model.fit_generator.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
    verbose_val : int
        verbose option of evaluate_generator.
        0 = silent, 1 = progress bar.
    zero_center_folder : None or str
        Path to a folder in which zero centering images are stored. [default: None]
        If this path is set, zero centering images for the given dataset will either be calculated and saved
        automatically at the start of the training, or loaded if they have been saved before.

    Private attributes
    ------------------
    _train_files : list or None
        A list containing the paths to the different training files on which the model will be trained on.
        Example for the output format:
                [
                 [['path/to/train_file_1_dimx.h5', 'path/to/train_file_1_dimy.h5'], number_of_events_train_files_1],
                 [['path/to/train_file_2_dimx.h5', 'path/to/train_file_2_dimy.h5'], number_of_events_train_files_2],
                 ...
                ]
    _val_files : list or None
        Like train_files but for the validation files.
    _multiple_inputs : bool or None
        Whether seperate sets of input files were given (e.g. for networks taking data
        simulataneosly from different files).
    _list_file : str or None
        Path to the list file that was used to set the training and validation files. Is None if no list file
        has been used yet.
    _modeldata : namedtuple or None
        Optional info only required for building a predefined model with OrcaNet. [default: None]
        It is not needed for executing orcatrain. It is set via self.load_from_model_file.

        modeldata.nn_arch : str
            Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
        modeldata.loss_opt : tuple(dict, dict/str/None,)
            Tuple that contains 1) the loss_functions and loss_weights as dicts (this is the losses table from the toml file)
            and 2) the metrics.
        modeldata.args : dict
            Keyword arguments for the model generation.

    """
    def __init__(self, main_folder, list_file=None, config_file=None):
        """
        Set the attributes of the Configuration object.

        Values are loaded from the given files, if provided. Otherwise, default values are used.

        Parameters
        ----------
        main_folder : str
            Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
        list_file : str or None
            Path to a list file with pathes to all the h5 files that should be used for training and validation.
        config_file : str or None
            Path to the config file with attributes that are used instead of the default ones.

        """
        # Configuration:
        self.batchsize = 64
        self.class_type = ['None', 'energy_dir_bjorken-y_vtx_errors']
        self.epochs_to_train = -1
        self.filter_out_tf_garbage = True
        self.initial_epoch = -1
        self.initial_fileno = -1
        self.learning_rate = 0.001
        self.n_events = None
        self.n_gpu = (1, 'avolkov')
        self.str_ident = ''
        self.swap_4d_channels = None
        self.train_logger_display = 100
        self.train_logger_flush = -1
        self.use_scratch_ssd = False
        self.validate_after_n_train_files = 2
        self.verbose_train = 2
        self.verbose_val = 1
        self.zero_center_folder = None

        self._default_values = dict(self.__dict__)

        # Main folder:
        if main_folder[-1] == "/":
            self.main_folder = main_folder
        else:
            self.main_folder = main_folder+"/"

        # Private attributes:
        self._train_files = None
        self._val_files = None
        self._multiple_inputs = None
        self._list_file = None
        self._modeldata = None

        # Load the optionally given list and config files.
        if list_file is not None:
            self.set_from_list_file(list_file)
        if config_file is not None:
            self.set_from_config_file(config_file)

    def set_from_list_file(self, list_file):
        """ Set filepaths to the ones given in a list file. """
        if self._list_file is None:
            self._train_files, self._val_files, self._multiple_inputs = read_out_list_file(list_file)
            # Save internally which path was used to load the info
            self._list_file = list_file
        else:
            raise AssertionError("You tried to load filepathes from a list file, but pathes have already been loaded \
            for this object. (From the file " + self._list_file + ")\nYou should not use \
            two different list files for one Configuration object!")

    def set_from_config_file(self, config_file):
        """ Overwrite default attribute values with values from a config file. """
        user_values = read_out_config_file(config_file)
        for key in user_values:
            if hasattr(self, key):
                setattr(self, key, user_values[key])
            else:
                raise AssertionError("You tried to set the attribute "+str(key)+" in your config file\n"
                                     + config_file + "\n, but this attribute is not provided. Check \
                                     the possible attributes in the definition of the Configuration class.")

    def set_from_model_file(self, model_file):
        """ Set attributes for generating models with OrcaNet. """
        nn_arch, loss_opt, args = read_out_model_file(model_file)
        ModelData = namedtuple("ModelData", "nn_arch loss_opt args")
        data = ModelData(nn_arch, loss_opt, args)
        self._modeldata = data

    def get_latest_epoch(self):
        """
        Check all saved models in the ./saved_models folder and return the highest epoch / file_no pair.

        Returns
        -------
        latest_epoch : tuple
            The highest epoch, file_no pair. (0,0) if the folder is empty or does not exist yet.

        """
        if os.path.exists(self.main_folder + "saved_models"):
            files = os.listdir(self.main_folder + "saved_models")
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
        elif epoch[1] == len(self.get_train_files()):
            next_epoch = (epoch[0] + 1, 1)
        else:
            next_epoch = (epoch[0], epoch[1] + 1)
        return next_epoch

    def use_local_node(self):
        """
        Copies the test and val files to the node-local ssd scratch folder and sets the new filepaths of the train and val data.
        Speeds up I/O and reduces RRZE network load.
        """
        train_files_ssd, test_files_ssd = use_node_local_ssd_for_input(self.get_train_files(), self.get_val_files(), self.get_multiple_inputs())
        self._train_files = train_files_ssd
        self._val_files = test_files_ssd

    def make_folder_structure(self):
        """
        Make subfolders for a specific model if they don't exist already. These subfolders will contain e.g. saved models,
        logfiles, etc.

        """
        main_folder = self.main_folder
        folders_to_create = [main_folder + "log_train", main_folder + "saved_models",
                             main_folder + "plots/activations", main_folder + "predictions"]
        for directory in folders_to_create:
            if not os.path.exists(directory):
                print("Creating directory: " + directory)
                os.makedirs(directory)

    def get_n_bins(self):
        return h5_get_n_bins(self._train_files)

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

    def get_modeldata(self):
        return self._modeldata

    def get_list_file(self):
        return self._list_file


def orca_train(cfg, initial_model=None):
    """
    Core code that trains a neural network.

    Set up everything for the training (like the folder structure and potentially loading in a saved model)
    and train for the given number of epochs.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.
    initial_model : ks.models.Model or None
        Compiled keras model to use for training and validation. Only required for the first epoch of training, as
        the most recent saved model will be loaded otherwise.

    """
    if cfg.filter_out_tf_garbage:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg.make_folder_structure()
    write_full_logfile_startup(cfg)
    # The epoch that will be incremented during the scripts:
    epoch = (cfg.initial_epoch, cfg.initial_fileno)
    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))
    # Epoch here is the epoch of the currently existing model (or 0,0 if there is none)
    if epoch[0] == 0 and epoch[1] == 0:
        assert initial_model is not None, "You need to provide a compiled keras model for the start of the training! (You gave None)"
        model = initial_model
    else:
        # Load an existing model
        if initial_model is not None:
            warnings.warn("You provided a model even though this is not the start of the training. Provided model is ignored!")
        path_of_model = cfg.main_folder + 'saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
        print("Loading saved model: "+path_of_model)
        model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    model.summary()
    if cfg.use_scratch_ssd:
        cfg.use_local_node()

    trained_epochs = 0
    while trained_epochs < cfg.epochs_to_train or cfg.epochs_to_train == -1:
        # Set epoch to the next file
        epoch = cfg.get_next_epoch(epoch)
        train_and_validate_model(cfg, model, epoch)
        trained_epochs += 1


def orca_eval(cfg):
    """
    Core code that evaluates a neural network. The input parameters are the same as for orca_train, so that it is compatible
    with the .toml file.
    TODO Should be directly callable on a saved model, so that less arguments are required, and maybe no .toml is needed?

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    """
    folder_name = cfg.main_folder
    test_files = cfg.get_val_files()
    n_bins = cfg.get_n_bins()
    class_type = cfg.class_type
    swap_4d_channels = cfg.swap_4d_channels
    batchsize = cfg.batchsize
    str_ident = cfg.str_ident
    list_name = os.path.basename(cfg.get_list_file()).split(".")[0]
    nn_arch = cfg.get_modeldata().nn_arch
    epoch = (cfg.initial_epoch, cfg.initial_fileno)

    if cfg.filter_out_tf_garbage:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))

    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None

    if cfg.use_scratch_ssd:
        cfg.use_local_node()

    path_of_model = folder_name + 'saved_models/model_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5'
    model = ks.models.load_model(path_of_model, custom_objects=get_all_loss_functions())
    modelname = get_modelname(n_bins, class_type, nn_arch, swap_4d_channels, str_ident)
    arr_filename = folder_name + 'predictions/pred_model_epoch_{}_file_{}_on_{}_val_files.npy'.format(str(epoch[0]), str(epoch[1]), list_name)

    predict_and_investigate_model_performance(cfg, model, test_files, n_bins, batchsize, class_type, swap_4d_channels,
                                              str_ident, modelname, xs_mean, arr_filename, folder_name)
