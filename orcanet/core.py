#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import toml
import warnings
from keras.models import load_model
from keras.utils import plot_model
import keras.backend as kb
import time
from datetime import timedelta

from orcanet.backend import make_model_prediction, save_actv_wghts_plot, train_model, validate_model
from orcanet.utilities.visualization import update_summary_plot
from orcanet.in_out import IOHandler, HistoryHandler
from orcanet.utilities.nn_utilities import load_zero_center_data, get_auto_label_modifier
from orcanet.logging import log_start_training, SummaryLogger, log_start_validation


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
        self.history = HistoryHandler(output_folder)

        self.xs_mean = None
        self._auto_label_modifier = None
        self._stored_model = None

    def train_and_validate(self, model=None, epochs=None):
        """
        Train a model and validate according to schedule.

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
            for the first epoch (the start of training).
            Afterwards, if model is None, the most recent model will be
            loaded automatically to continue the training.
        epochs : int or None
            How many epochs should be trained by running this function.
            None for infinite.

        Returns
        -------
        model : ks.models.Model
            The trained keras model.

        """
        latest_epoch = self.io.get_latest_epoch()

        model = self._get_model(model, logging=True)
        self._stored_model = model

        # check if the validation is missing for the latest fileno
        if latest_epoch is not None:
            state = self.history.get_state()[-1]
            if state["is_validated"] is False and self._val_is_due(latest_epoch):
                    self.validate()

        next_epoch = self.io.get_next_epoch(latest_epoch)
        n_train_files = self.io.get_no_of_files("train")

        trained_epochs = 0
        while epochs is None or trained_epochs < epochs:
            # Train on remaining files
            for file_no in range(next_epoch[1], n_train_files + 1):
                curr_epoch = (next_epoch[0], file_no)
                self.train(model)
                if self._val_is_due(curr_epoch):
                    self.validate()

            next_epoch = (next_epoch[0] + 1, 1)
            trained_epochs += 1

        self._stored_model = None
        return model

    def train(self, model=None):
        """
        Trains a model on the next file.

        The progress of the training is also logged and plotted.

        Parameters
        ----------
        model : ks.models.Model or None
            Compiled keras model to use for training. Required for the first
            epoch (the start of training).
            Afterwards, if model is None, the most recent model will be
            loaded automatically to continue the training.

        Returns
        -------
        history : dict
            The history of the training on this file. A record of training
            loss values and metrics values.

        """
        # Create folder structure
        self.io.get_subfolder(create=True)
        latest_epoch = self.io.get_latest_epoch()

        model = self._get_model(model, logging=True)

        self._set_up(model, logging=True)

        # epoch about to be trained
        next_epoch = self.io.get_next_epoch(latest_epoch)
        next_epoch_float = self.io.get_epoch_float(*next_epoch)

        if latest_epoch is None:
            self.io.check_connections(model)
            log_start_training(self)

        model_path = self.io.get_model_path(*next_epoch)
        model_path_local = self.io.get_model_path(*next_epoch, local=True)
        if os.path.isfile(model_path):
            raise FileExistsError(
                "Can not train model in epoch {} file {}, this model has "
                "already been saved!".format(*next_epoch))

        smry_logger = SummaryLogger(self, model)

        lr = self.io.get_learning_rate(next_epoch)
        kb.set_value(model.optimizer.lr, lr)

        files_dict = self.io.get_file("train", next_epoch[1])

        line = "Training in epoch {} on file {}/{}".format(
            next_epoch[0], next_epoch[1], self.io.get_no_of_files("train"))
        self.io.print_log(line)
        self.io.print_log("-" * len(line))
        self.io.print_log("Learning rate is at {}".format(
            kb.get_value(model.optimizer.lr)))
        self.io.print_log('Inputs and files:')
        for input_name, input_file in files_dict.items():
            self.io.print_log("   {}: \t{}".format(input_name,
                                                   os.path.basename(
                                                       input_file)))

        start_time = time.time()
        history = train_model(self, model, next_epoch, batch_logger=True)
        elapsed_s = int(time.time() - start_time)

        model.save(model_path)
        smry_logger.write_line(next_epoch_float, lr, history_train=history)

        self.io.print_log('Training results:')
        for metric_name, loss in history.items():
            self.io.print_log("   {}: \t{}".format(metric_name, loss))
        self.io.print_log("Elapsed time: {}".format(timedelta(seconds=elapsed_s)))
        self.io.print_log("Saved model to: {}\n".format(model_path_local))

        update_summary_plot(self)

        return history

    def validate(self):
        """
        Validate the most recent saved model on all validation files.

        Will also log the progress, as well as update the summary plot and
        plot weights and activations of the model.

        Returns
        -------
        history : dict
            The history of the validation on all files. A record of validation
            loss values and metrics values.

        """
        latest_epoch = self.io.get_latest_epoch()
        if latest_epoch is None:
            raise ValueError("Can not validate: No saved model found")
        if self.history.get_state()[-1]["is_validated"] is True:
            raise ValueError("Can not validate in epoch {} file {}: "
                             "Has already been validated".format(*latest_epoch))

        if self._stored_model is None:
            model = self.load_saved_model(*latest_epoch)
        else:
            model = self._stored_model

        self._set_up(model, logging=True)

        epoch_float = self.io.get_epoch_float(*latest_epoch)
        smry_logger = SummaryLogger(self, model)

        log_start_validation(self)

        start_time = time.time()
        history = validate_model(self, model)
        elapsed_s = int(time.time() - start_time)

        self.io.print_log('Validation results:')
        for metric_name, loss in history.items():
            self.io.print_log("   {}: \t{}".format(metric_name, loss))
        self.io.print_log("Elapsed time: {}\n".format(timedelta(seconds=elapsed_s)))
        smry_logger.write_line(epoch_float, "n/a", history_val=history)

        update_summary_plot(self)
        save_actv_wghts_plot(self, model, latest_epoch, samples=self.cfg.batchsize)

        return history

    def predict(self, epoch=-1, fileno=-1, concatenate=False):
        """
        Make a prediction if it does not exist yet, and return its filepath.

        Load a model, let it predict on all samples of the validation set
        in the toml list, and save this prediction together with all the
        mc_info as a h5 file in the predictions subfolder.

        Setting epoch, fileno = -1, -1 will load the most recent epoch
        found in the main folder.

        Parameters
        ----------
        epoch : int
            The epoch of the model to load for prediction
        fileno : int
            The file number of the model to load for prediction.
        concatenate : bool
            Whether the prediction files should also be concatenated.

        Returns
        -------
        pred_filename : list
            List to the paths of all created prediction file.
            If concatenate = True, the list only contains the
            path to the concatenated prediction file.

        """
        if fileno == -1 and epoch == -1:
            latest_epoch = self.io.get_latest_epoch()
            if latest_epoch is None:
                raise FileNotFoundError("Can not look up most recent model: "
                                        "No models found for {}".format(self.cfg.output_folder))
            epoch, fileno = latest_epoch
            print("Automatically set epoch to epoch {} file {}.".format(epoch, fileno))

        is_pred_done = self._check_if_pred_already_done(epoch, fileno)
        if is_pred_done:
            print("Prediction has already been done.")
            pred_filepaths = self.io.get_pred_files_list()

        else:
            model = self.load_saved_model(epoch, fileno, logging=False)
            self._set_up(model)

            start_time = time.time()
            make_model_prediction(self, model, epoch, fileno, samples=None)
            elapsed_s = int(time.time() - start_time)
            print('Finished predicting on all validation files.')
            print("Elapsed time: {}\n".format(timedelta(seconds=elapsed_s)))

            pred_filepaths = self.io.get_pred_files_list()

        # concatenate all prediction files if wished
        concatenated_folder = self.io.get_subfolder("predictions") + '/concatenated'
        n_val_files = self.io.get_no_of_files("val")
        if concatenate is True and n_val_files > 1:
            if not os.path.isdir(concatenated_folder):
                print('Concatenating all prediction files to a single one.')
                pred_filename_conc = self.io.concatenate_pred_files(concatenated_folder)
                pred_filepaths = [pred_filename_conc]
            else:
                # omit directories if there are any in the concatenated folder
                fname_conc_file_list = list(file for file in os.listdir(concatenated_folder)
                                        if os.path.isfile(os.path.join(concatenated_folder, file)))
                pred_filepaths = [concatenated_folder + '/' + fname_conc_file_list[0]]

        return pred_filepaths

    def _check_if_pred_already_done(self, epoch, fileno):
        """
        Checks if the prediction has already been done before.
        (-> predicted on all validation files)

        Returns
        -------
        pred_done : bool
            Boolean flag to specify if the prediction has
            already been fully done or not.

        """
        latest_pred_file_no = self.io.get_latest_prediction_file_no(epoch, fileno)
        total_no_of_val_files = self.io.get_no_of_files('val')

        if latest_pred_file_no is None:
            pred_done = False
        elif latest_pred_file_no == total_no_of_val_files - 1:  # val_file_nos start with 0
            return True
        else:
            pred_done = False

        return pred_done

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
        if self.xs_mean is None:
            if self.cfg.zero_center_folder is None:
                raise ValueError("Can not calculate zero center: "
                                 "No zero center folder given")
            self.xs_mean = load_zero_center_data(self, logging=logging)
        return self.xs_mean

    def load_saved_model(self, epoch, fileno, logging=False):
        """
        Load a saved model.

        Parameters
        ----------
        epoch : int
            Epoch of the saved model.
        fileno : int
            Fileno of the saved model.
        logging : bool
            If True, will log this function call into the log.txt file.

        Returns
        -------
        model : keras model

        """
        path_of_model = self.io.get_model_path(epoch, fileno)
        path_loc = self.io.get_model_path(epoch, fileno, local=True)
        self.io.print_log("Loading saved model: " + path_loc, logging=logging)
        model = load_model(path_of_model, custom_objects=self.cfg.custom_objects)
        return model

    def _get_model(self, model, logging=False):
        """ Load most recent saved model or use user model. """
        latest_epoch = self.io.get_latest_epoch()

        if latest_epoch is None:
            # new training
            if model is None:
                raise ValueError("You need to provide a compiled keras model "
                                 "for the start of the training! (You gave None)")
            self._save_as_json(model)

            try:
                plots_folder = self.io.get_subfolder("plots", create=True)
                plot_model(model, plots_folder + "/model_plot.png")
            except OSError as e:
                warnings.warn("Can not plot model: " + str(e))

        else:
            if model is None:
                model = self.load_saved_model(*latest_epoch, logging=logging)

        return model

    def _save_as_json(self, model):
        """ Save the architecture of a model as json to fixed path. """
        json_filename = "model_arch.json"

        json_string = model.to_json(indent=1)
        model_folder = self.io.get_subfolder("saved_models", create=True)
        with open(os.path.join(model_folder, json_filename), "w") as f:
            f.write(json_string)

    def _set_up(self, model, logging=False):
        """ Necessary setup for training, validating and predicting. """
        if self.cfg.filter_out_tf_garbage:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        if self.cfg.get_list_file() is None:
            raise ValueError("No files specified. You need to load a toml "
                             "list file with your files before training")

        if self.cfg.label_modifier is None:
            self._auto_label_modifier = get_auto_label_modifier(model)

        if self.cfg.use_scratch_ssd:
            self.io.use_local_node()

        if self.cfg.zero_center_folder is not None:
            self.get_xs_mean(logging)

    def _val_is_due(self, epoch):
        """ True if validation is due on given epoch according to schedule. """
        n_train_files = self.io.get_no_of_files("train")
        val_sched = (epoch[1] == n_train_files) or \
                    (self.cfg.validate_interval is not None and
                     epoch[1] % self.cfg.validate_interval == 0)
        return val_sched


class Configuration(object):
    """
    Contains all the configurable options in the OrcaNet scripts.

    All of these public attributes (the ones without a
    leading underscore) can be changed either directly or with a
    .toml config file via the method update_config().

    Attributes
    ----------
    batchsize : int
        Batchsize that will be used for the training and validation of
        the network.
    callback_train : keras callback or list or None
        Callback or list of callbacks to use during training.
    custom_objects : dict or None
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered by keras during deserialization of models.
    dataset_modifier : function or None
        For orga.predict: Function that determines which datasets get created
        in the resulting h5 file. If none, every output layer will get one
        dataset each for both the label and the prediction, and one dataset
        containing the mc_info from the validation files.
    filter_out_tf_garbage : bool
        If true, surpresses the tensorflow info logs which usually spam
        the terminal.
    key_samples : str
        The name of the datagroup in the h5 input files which contains
        the samples for the network.
    key_mc_info : str
        The name of the datagroup in the h5 input files which contains
        the info for the labels.
    label_modifier : function or None
        Operation to be performed on batches of labels read from the input files
        before they are fed into the model. If None is given, all labels with
        the same name as the output layers will be passed to the model as a dict,
        with the keys being the dtype names.
    learning_rate : float, tuple, function or str
        The learning rate for the training.
        If it is a float: The learning rate will be constantly this value.
        If it is a tuple of two floats: The first float gives the learning rate
        in epoch 1 file 1, and the second float gives the decrease of the
        learning rate per file (e.g. 0.1 for 10% decrease per file).
        If it is a function: Takes as an input the epoch and the
        file number (in this order), and returns the learning rate.
        If it is a str: Path to a csv file inside the main folder, containing
        3 columns with the epoch, fileno, and the value the lr will be set
        to when reaching this epoch/fileno.
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
        self.key_mc_info = "y"
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
                raise ValueError("Wrong input format in toml list file (input {}:"
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

        Parameters
        ----------
        which : str
            Either "train" or "val".

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

    @property
    def default_values(self):
        """ The default values for all settings. """
        return self._default_values
