#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import warnings
import keras as ks
import toml

from orcanet.backend import train_and_validate_model, make_model_prediction
from orcanet.in_out import read_out_list_file, use_node_local_ssd_for_input, write_full_logfile_startup, IOHandler
from orcanet.utilities.nn_utilities import load_zero_center_data, get_auto_label_modifier


class Configuration(object):
    """
    Container object for all the configurable options in the OrcaNet scripts.

    TODO add a clober script that properly deletes models + logfiles
    Sensible default values were chosen for the settings.
    You can change the all of these public attributes (the ones without a leading underscore _) either directly or with a
    .toml config file via the method update_config().

    Attributes
    ----------
    batchsize : int
        Batchsize that should be used for the training and validation of the network.
    custom_objects : dict or None
        Optional dictionary mapping names (strings) to custom classes or functions to be considered by keras
        during deserialization of models.
    dataset_modifier : function or None
        For orca_eval: Function that determines which datasets get created in the resulting h5 file.
        If none, every output layer will get one dataset each for both the label and the prediction, and one dataset
        containing the mc_info from the validation files.
    filter_out_tf_garbage : bool
        If true, surpresses the tensorflow info logs which usually spam the terminal.
    epochs_to_train : int or None
        How many new epochs should be trained by running this function. None for infinite.
    key_samples : str
        The name of the datagroup in your h5 input files which contains the samples for the network.
    key_labels : str
        The name of the datagroup in your h5 input files which contains the labels for the network.
    label_modifier : function or None
        Operation to be performed on batches of labels read from the input files before they are fed into the model.
        If None is given, all labels with the same name as the output layers will be passed to the model as a dict,
        with the keys being the dtype names.
    learning_rate : float or tuple or function
        The learning rate for the training.
        If it is a float, the learning rate will be constantly this value.
        If it is a tuple of two floats, the first float gives the learning rate in epoch 1 file 1, and the second
        float gives the decrease of the learning rate per file (e.g. 0.1 for 10% decrease per file).
        You can also give an arbitrary function, which takes as an input the epoch, the file number and the
        Configuration object (in this order), and returns the learning rate.
    max_queue_size : int
        max_queue_size option of the keras training and evaluation generator methods. How many batches get preloaded
        from the generator.
    n_events : None or int
        For testing purposes. If not the whole .h5 file should be used for training, define the number of samples.
    sample_modifier : function or None
        Operation to be performed on batches of samples read from the input files before they are fed into the model.
    shuffle_train : bool
        If true, the order in which batches are read out from the files during training are randomized each time they
        are read out.
    train_logger_display : int
        How many batches should be averaged for one line in the training log files.
    train_logger_flush : int
        After how many lines the training log file should be flushed (updated on the disk).
        -1 for flush at the end of the file only.
    output_folder : str
        Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
    use_scratch_ssd : bool
        Only working at HPC Erlangen: Declares if the input files should be copied to the node-local SSD scratch space.
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

    """
    def __init__(self, output_folder, list_file, config_file):
        """
        Set the attributes of the Configuration object.

        Values are loaded from the given files, if provided. Otherwise, default values are used.

        Parameters
        ----------
        output_folder : str
            Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
        list_file : str or None
            Path to a toml list file with pathes to all the h5 files that should be used for training and validation.
        config_file : str or None
            Path to a toml config file with attributes that are used instead of the default ones.

        """
        self.batchsize = 64
        self.custom_objects = None
        self.dataset_modifier = None
        self.epochs_to_train = None
        self.filter_out_tf_garbage = True
        self.key_samples = "x"
        self.key_labels = "y"
        self.label_modifier = None
        self.learning_rate = 0.001
        self.max_queue_size = 10
        self.n_events = None
        self.sample_modifier = None
        self.shuffle_train = False
        self.train_logger_display = 100
        self.train_logger_flush = -1
        self.use_scratch_ssd = False
        self.validate_after_n_train_files = 2
        self.verbose_train = 2
        self.verbose_val = 1
        self.zero_center_folder = None

        self._default_values = dict(self.__dict__)

        # Main folder:
        if output_folder[-1] == "/":
            self.output_folder = output_folder
        else:
            self.output_folder = output_folder+"/"

        # Private attributes:
        self._auto_label_modifier = None
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
        Set filepaths to the ones given in a list file.

        Parameters
        ----------
        list_file : str or None
            Path to a toml list file with pathes to all the h5 files that should be used for training and validation.

        """
        assert self._list_file is None, "You tried to load filepathes from a list file, but pathes have already " \
                                            "been loaded for this object. (From the file " + self._list_file \
                                            + ")\nYou can not use two different list files at once!"
        self._train_files, self._val_files = read_out_list_file(list_file)
        self._list_file = list_file

    def update_config(self, config_file):
        """
        Update the default configuration with values from a config file.

        Parameters
        ----------
        config_file : str
            Path to a toml config file with attribute values to use instead of the default ones.

        """
        user_values = toml.load(config_file)["config"]
        for key in user_values:
            assert hasattr(self, key), "Unknown attribute "+str(key)+" in config file " + config_file
            setattr(self, key, user_values[key])

    def get_list_file(self):
        """
        Returns the path to the list file that was used to set the training and validation files.
        None if no list file has been used.
        """
        return self._list_file

    def get_files(self, which):
        """
        Get the training file paths.

        Returns
        -------
        dict
            A dict containing the paths to the training or validation files on which the model will be trained on.
            Example for the format for two input sets with two files each:
                    {
                     "input_A" : ('path/to/set_A_file_1.h5', 'path/to/set_A_file_2.h5'),
                     "input_B" : ('path/to/set_B_file_1.h5', 'path/to/set_B_file_2.h5'),
                    }
        """
        if which == "train":
            assert self._train_files is not None, "No train files have been specified!"
            return self._train_files
        elif which == "val":
            assert self._val_files is not None, "No validation files have been specified!"
            return self._val_files
        else:
            raise NameError("Unknown fileset name ", which)

    def use_local_node(self):
        """
        Copies the test and val files to the node-local ssd scratch folder and sets the new filepaths of the train and val data.
        Speeds up I/O and reduces RRZE network load.
        """
        train_files_ssd, val_files_ssd = use_node_local_ssd_for_input(self.get_files("train"), self.get_files("val"))
        self._train_files = train_files_ssd
        self._val_files = val_files_ssd


class OrcaHandler:
    def __init__(self, output_folder, list_file=None, config_file=None):
        self.cfg = Configuration(output_folder, list_file, config_file)
        self.io = IOHandler(self.cfg)

    def train(self, model=None, force_model=False):
        """
        Core code that trains a neural network.

        Set up everything for the training (like the folder structure and potentially loading in a saved model)
        and train for the given number of epochs.

        Parameters
        ----------
        model : ks.models.Model or None
            Compiled keras model to use for training and validation. Required for the first epoch of training, as
            the most recent saved model will be loaded otherwise. Also required, if you use the force_model parameter.
        force_model : bool
            Use the keras model specified in the "model" parameter and don't load the most recently trained model
            from the disk.

        """
        if self.cfg.get_list_file() is None:
            raise ValueError("No files specified. You need to load a toml list file with your files before training")

        if self.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        self.io.get_subfolder(create=True)
        write_full_logfile_startup(self)
        # the epoch of the currently existing model (or 0,0 if there is none)
        epoch = self.io.get_latest_epoch()
        print("Set to epoch {} file {}.".format(epoch[0], epoch[1]))

        if epoch[0] == 0 and epoch[1] == 0:
            if model is None:
                raise ValueError("You need to provide a compiled keras model for the start of the training! (You gave None)")
            model = model
        elif force_model is True:
            if model is None:
                raise ValueError('You set "force_model" to True, but didnt provide a model in "model" that should be used!')
            print('You forced the OrcaHandler.train() method to use a keras model specified with the parameter "forced_model"!')
            model = model
        else:
            # Load an existing model
            if model is not None:
                raise ValueError("You provided a model even though this is not the start of the training. Provided model is ignored!")
            path_of_model = self.io.get_model_path(epoch[0], epoch[1])
            print("Loading saved model: " + path_of_model)
            model = ks.models.load_model(path_of_model, custom_objects=self.cfg.custom_objects)

        if self.cfg.label_modifier is None:
            self.cfg._auto_label_modifier = get_auto_label_modifier(model)

        self.io.check_connections(model)
        # model.summary()
        if self.cfg.use_scratch_ssd:
            self.cfg.use_local_node()

        trained_epochs = 0
        while self.cfg.epochs_to_train is None or trained_epochs < self.cfg.epochs_to_train:
            # Set epoch to the next file
            epoch = self.io.get_next_epoch(epoch)
            train_and_validate_model(self, model, epoch)
            trained_epochs += 1

    def predict(self, epoch=-1, fileno=-1):
        """
        Make a prediction if it does not exist yet, and return its filepath.

        Load a model, let it predict on all samples of the validation set in the toml list, and save it as a h5 file.
        The cfg.eval_epoch and cfg.eval_fileno parameters define which model is loaded.

        Parameters
        ----------
        epoch : int
            The epoch of the model to load for prediction
            Can also give -1 to automatically load the most recent epoch found in the main folder.
        fileno : int
            When using multiple files, define the file number for the prediction, e.g.
            1 for load the model trained on the first file. If both epoch and fileno are -1, automatically set to the most
            recent file found in the main folder.

        Returns
        -------
        pred_filename : str
            The path to the created prediction file.

        """
        assert self.cfg.get_list_file() is not None, "No files specified. You need to load a toml list file " \
                                                     "with your files before predicting or loading a prediction"

        if self.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        if epoch == -1 and fileno == -1:
            epoch, fileno = self.io.get_latest_epoch()
            print("Automatically set epoch to epoch {} file {}.".format(epoch, fileno))

        list_name = os.path.splitext(os.path.basename(self.cfg.get_list_file()))[0]
        pred_filename = self.io.get_pred_path(epoch, fileno, list_name)

        if os.path.isfile(pred_filename):
            print("Prediction has already been done.")

        else:
            if self.cfg.zero_center_folder is not None:
                xs_mean = load_zero_center_data(self)
            else:
                xs_mean = None

            if self.cfg.use_scratch_ssd:
                self.cfg.use_local_node()

            model = ks.models.load_model(self.io.get_model_path(epoch, fileno),
                                         custom_objects=self.cfg.custom_objects)

            make_model_prediction(self, model, xs_mean, pred_filename, samples=None)

        return pred_filename
