#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import keras as ks
import toml
import warnings

from orcanet.backend import train_and_validate_model, make_model_prediction
from orcanet.in_out import IOHandler, HistoryHandler
from orcanet.utilities.nn_utilities import load_zero_center_data, get_auto_label_modifier
from orcanet.logging import log_start_training


class Organizer:
    """
    Core class for working with networks in OrcaNet.

    Attributes
    ----------
    cfg : Configuration
        Contains all configurable options.
    io : orcanet.in_out.IOHandler
        Utility functions for accessing the info in cfg.
    history : orcanet.in_out.HistoryHandler
        For reading and plotting data from the log files created
        during training.

    """

    def __init__(self, output_folder, list_file=None, config_file=None):
        """
        Set the attributes of the Configuration object.

        Instead of using a config_file, the attributes of orga.cfg can
        also be changed directly, e.g. by calling orga.cfg.batchsize.

        Parameters
        ----------
        output_folder : str
            Name of the folder of this model in which everything will be saved,
            e.g., the summary.txt log file is located in here.
            Will be used to load saved files or to save new ones.
        list_file : str or None
            Path to a toml list file with pathes to all the h5 files that should
            be used for training and validation.
            Will be used to extract of samples or labels.
        config_file : str or None
            Path to a toml config file with settings that are used instead of
            the default ones.

        """
        self.cfg = Configuration(output_folder, list_file, config_file)
        self.io = IOHandler(self.cfg)
        self.history = HistoryHandler(self.cfg.output_folder + "summary.txt",
                                      self.io.get_subfolder("train_log"))

        self._xs_mean = None
        self._auto_label_modifier = None

    def train(self, model=None, force_model=False, epochs=None):
        """
        Train and validate a model.

        The various settings of this process can be controlled with the
        attributes of orca.cfg.
        The model will be trained on the given data, saved and validated.
        Logfiles of the training are saved in the output folder.
        Plots showing the training and validation history, as well as
        the weights and activations of the network are generated in
        the plots subfolder after every validation.
        The training can be resumed by executing this function again.

        Parameters
        ----------
        model : ks.models.Model or None
            Compiled keras model to use for training and validation. Required
            only for the start of training. After the model was saved for
            the first time, the most recent model will be loaded automatically
            to continue the training.
        force_model : bool
            If true, the given model will be used to continue the training,
            instead of loading the most recent saved one.
        epochs : int or None
            How many epochs should be trained by running this function.
            None for infinite.

        Returns
        -------
        model : ks.models.Model
            The trained keras model.

        """
        if self.cfg.get_list_file() is None:
            raise ValueError("No files specified. You need to load a toml "
                             "list file with your files before training")

        if self.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        self.io.get_subfolder(create=True)

        # the epoch of the currently existing model (or 0,0 if there is none)
        latest_epoch = self.io.get_latest_epoch()
        print(
            "Set to epoch {} file {}.".format(latest_epoch[0], latest_epoch[1]))

        if self.io.is_new():
            if model is None:
                raise ValueError("You need to provide a compiled keras model "
                                 "for the start of the training! (You gave None)")
            try:
                ks.utils.plot_model(
                    model, self.io.get_subfolder("plots") +
                           "/model_epoch_{}_file_{}.png".format(*latest_epoch))
            except OSError as e:
                warnings.warn("Can not plot model: " + str(e))

        elif force_model is True:
            if model is None:
                raise ValueError('You set "force_model" to True, but didnt '
                                 'provide a model in "model" that should be used!')
            print('Continuing training with user model (force_model=True)')
        else:
            # Load an existing model
            if model is not None:
                raise ValueError("You provided a model even though this is not "
                                 "the start of the training.")
            path_of_model = self.io.get_model_path(latest_epoch[0],
                                                   latest_epoch[1])
            print("Continuing training with saved model: " + path_of_model)
            model = ks.models.load_model(path_of_model,
                                         custom_objects=self.cfg.custom_objects)

        if self.cfg.label_modifier is None:
            self._auto_label_modifier = get_auto_label_modifier(model)

        self.io.check_connections(model)

        if self.cfg.use_scratch_ssd:
            self.io.use_local_node()

        # Set epoch to the next file (the one we are about to train)
        next_epoch = self.io.get_next_epoch(latest_epoch)

        log_start_training(self)

        if self.cfg.zero_center_folder is not None:
            # Make sure the xs_mean is calculated
            self.get_xs_mean(logging=True)

        trained_epochs = 0
        while epochs is None or trained_epochs < epochs:
            # Train on remaining files of an epoch
            train_and_validate_model(self, model, next_epoch)
            next_epoch = (next_epoch[0] + 1, 1)
            trained_epochs += 1

        return model

    def predict(self, epoch=-1, fileno=-1):
        """
        Make a prediction if it does not exist yet, and return its filepath.

        Load a model, let it predict on all samples of the validation set
        in the toml list, and save this prediction together with all the
        mc_info as a h5 file in the predictions subfolder.

        Parameters
        ----------
        epoch : int
            The epoch of the model to load for prediction
            Can also give -1 to automatically load the most recent epoch
            found in the main folder.
        fileno : int
            When using multiple files, define the file number for the
            prediction, e.g. 1 for load the model trained on the first file.
            Can also give -1 to automatically load the most recent fileno
            from the given epoch found in the main folder.

        Returns
        -------
        pred_filename : str
            The path to the created prediction file.

        """
        if self.cfg.get_list_file() is None:
            raise ValueError("No files specified. You need to load a toml list "
                             "file before predicting or loading a prediction")

        if self.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        if fileno == -1:
            epoch, fileno = self.io.get_latest_epoch(epoch)
            print("Automatically set epoch to epoch {} file {}.".format(epoch,
                                                                        fileno))

        list_name = os.path.splitext(
            os.path.basename(self.cfg.get_list_file()))[0]
        pred_filename = self.io.get_pred_path(epoch, fileno, list_name)

        if os.path.isfile(pred_filename):
            print("Prediction has already been done.")
        else:
            if self.cfg.zero_center_folder is not None:
                self.get_xs_mean()

            if self.cfg.use_scratch_ssd:
                self.io.use_local_node()

            model = ks.models.load_model(
                self.io.get_model_path(epoch, fileno),
                custom_objects=self.cfg.custom_objects)

            make_model_prediction(self, model, pred_filename, samples=None)

        return pred_filename

    def get_xs_mean(self, logging=False):
        """
        Set and return the zero center image for each list input.

        Requires the cfg.zero_center_folder to be set. If no existing
        image for the given input files is found in the folder, it will
        be calculated and saved by averaging over all samples in the
        train dataset.

        Parameters
        ----------
        logging : bool
            If true, the execution of this function will be logged into the
            full summary in the output folder if called for the first time.

        Returns
        -------
        dict
            Dict of numpy arrays that contains the mean_image of the x dataset
            (1 array per list input).
            Example format:
            { "input_A" : ndarray, "input_B" : ndarray }

        """
        if self.cfg.zero_center_folder is None:
            raise ValueError("Can not calculate zero center: "
                             "No zero center folder given")
        if self._xs_mean is None:
            self._xs_mean = load_zero_center_data(self, logging=logging)
        return self._xs_mean


class Configuration(object):
    """
    Contains all the configurable options in the OrcaNet scripts.

    All of these public attributes (the ones without a
    leading underscore) can be changed either directly or with a
    .toml config file via the method update_config().

    Attributes
    ----------
    batchsize : int
        Batchsize that should be used for the training and validation of
        the network.
    callback_train : keras callback or list or None
        Callback or list of callbacks to use during training.
    custom_objects : dict or None
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered by keras during deserialization of models.
    dataset_modifier : function or None
        For orga.pred: Function that determines which datasets get created
        in the resulting h5 file. If none, every output layer will get one
        dataset each for both the label and the prediction, and one dataset
        containing the mc_info from the validation files.
    filter_out_tf_garbage : bool
        If true, surpresses the tensorflow info logs which usually spam
        the terminal.
    key_samples : str
        The name of the datagroup in your h5 input files which contains
        the samples for the network.
    key_labels : str
        The name of the datagroup in your h5 input files which contains
        the labels for the network.
    label_modifier : function or None
        Operation to be performed on batches of labels read from the input files
        before they are fed into the model. If None is given, all labels with
        the same name as the output layers will be passed to the model as a dict,
        with the keys being the dtype names.
    learning_rate : float or tuple or function
        The learning rate for the training.
        If it is a float, the learning rate will be constantly this value.
        If it is a tuple of two floats, the first float gives the learning rate
        in epoch 1 file 1, and the second float gives the decrease of the
        learning rate per file (e.g. 0.1 for 10% decrease per file).
        You can also give a function, which takes as an input the epoch and the
        file number (in this order), and returns the learning rate.
    max_queue_size : int
        max_queue_size option of the keras training and evaluation generator
        methods. How many batches get preloaded
        from the generator.
    n_events : None or int
        For testing purposes. If not the whole .h5 file should be used for
        training, define the number of samples.
    sample_modifier : function or None
        Operation to be performed on batches of samples read from the input
        files before they are fed into the model.
    shuffle_train : bool
        If true, the order in which batches are read out from the files during
        training are randomized each time they are read out.
    train_logger_display : int
        How many batches should be averaged for one line in the training log files.
    train_logger_flush : int
        After how many lines the training log file should be flushed (updated on
        the disk). -1 for flush at the end of the file only.
    output_folder : str
        Name of the folder of this model in which everything will be saved,
        e.g., the summary.txt log file is located in here.
    use_scratch_ssd : bool
        Only working at HPC Erlangen: Declares if the input files should be
        copied to the node-local SSD scratch space.
    validate_interval : int or None
        Validate the model after this many training files have been trained on
        in an epoch. There will always be a validation at the end of an epoch.
        None for only validate at the end of an epoch.
        Example: validate_interval=3 --> Validate after file 3, 6, 9, ...
    verbose_train : int
        verbose option of keras.model.fit_generator.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
    verbose_val : int
        verbose option of evaluate_generator.
        0 = silent, 1 = progress bar.
    zero_center_folder : None or str
        Path to a folder in which zero centering images are stored.
        If this path is set, zero centering images for the given dataset will
        either be calculated and saved automatically at the start of the
        training, or loaded if they have been saved before.

    """
    # TODO add a clober script that properly deletes models + logfiles
    def __init__(self, output_folder, list_file, config_file):
        """
        Set the attributes of the Configuration object.

        Values are loaded from the given files, if provided. Otherwise, default
        values are used.

        Parameters
        ----------
        output_folder : str
            Name of the folder of this model in which everything will be saved,
            e.g., the summary.txt log file is located in here.
        list_file : str or None
            Path to a toml list file with pathes to all the h5 files that should
            be used for training and validation.
        config_file : str or None
            Path to a toml config file with attributes that are used instead of
            the default ones.

        """
        self.batchsize = 64
        self.learning_rate = 0.001

        self.zero_center_folder = None
        self.validate_interval = None

        self.sample_modifier = None
        self.dataset_modifier = None
        self.label_modifier = None

        self.key_samples = "x"
        self.key_labels = "y"
        self.custom_objects = None
        self.shuffle_train = False

        self.callback_train = None
        self.use_scratch_ssd = False
        self.verbose_train = 1
        self.verbose_val = 0

        self.n_events = None
        self.filter_out_tf_garbage = True
        self.max_queue_size = 10
        self.train_logger_display = 100
        self.train_logger_flush = -1

        self._default_values = dict(self.__dict__)

        # Main folder:
        if output_folder[-1] == "/":
            self.output_folder = output_folder
        else:
            self.output_folder = output_folder+"/"

        # Private attributes:
        self._train_files = None
        self._val_files = None
        self._list_file = None

        # Load the optionally given list and config files.
        if list_file is not None:
            self.import_list_file(list_file)
        if config_file is not None:
            self.update_config(config_file)

    def import_list_file(self, list_file):
        """
        Import the filepaths of the training and validation files from a toml
        list file.

        Parameters
        ----------
        list_file : str
            Path to the toml list file.

        """
        if self._list_file is not None:
            raise ValueError("Can not load list file: Has already been loaded! "
                             "({})".format(self._list_file))

        file_content = toml.load(list_file)
        train_files, validation_files = {}, {}

        # no of train/val files in each input set
        n_train, n_val = [], []
        for input_key, input_values in file_content.items():
            if not len(input_values.keys()) == 2:
                raise NameError("Wrong input format in toml list file (input {}:"
                                " {})".format(input_key, input_values))
            if "train_files" not in input_values.keys():
                raise NameError("No train files specified in toml list file")
            if "validation_files" not in input_values.keys():
                raise NameError("No validation files specified in toml list file")

            train_files[input_key] = tuple(input_values["train_files"])
            validation_files[input_key] = tuple(input_values["validation_files"])
            n_train.append(len(train_files[input_key]))
            n_val.append(len(validation_files[input_key]))

        if not n_train.count(n_train[0]) == len(n_train):
            raise ValueError("The specified training inputs do not "
                             "all have the same number of files!")
        if not n_val.count(n_val[0]) == len(n_val):
            raise ValueError("The specified validation inputs do not "
                             "all have the same number of files!")

        self._train_files = train_files
        self._val_files = validation_files
        self._list_file = list_file

    def update_config(self, config_file):
        """
        Update the default cfg parameters with values from a toml config file.

        Parameters
        ----------
        config_file : str
            Path to a toml config file.

        """
        user_values = toml.load(config_file)["config"]
        for key in user_values:
            if hasattr(self, key):
                setattr(self, key, user_values[key])
            else:
                raise AttributeError(
                    "Unknown attribute {} in config file ".format(key))

    def get_list_file(self):
        """
        Returns the path to the list file that was used to set the training
        and validation files. None if no list file has been used.

        """
        return self._list_file

    def get_files(self, which):
        """
        Get the training or validation file paths for each list input set.

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
            if self._train_files is None:
                raise AttributeError("No train files have been specified!")
            return self._train_files
        elif which == "val":
            if self._val_files is None:
                raise AttributeError("No validation files have been specified!")
            return self._val_files
        else:
            raise NameError("Unknown fileset name ", which)

    def get_defaults(self):
        """ Get the default values for all settings. """
        return self._default_values
