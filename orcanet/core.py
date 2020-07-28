#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import toml
import warnings
import time
from datetime import timedelta
import tensorflow as tf
import tensorflow.keras as ks

import orcanet.backend as backend
from orcanet.utilities.visualization import update_summary_plot
from orcanet.in_out import IOHandler
from orcanet.history import HistoryHandler
from orcanet.utilities.nn_utilities import load_zero_center_data, get_auto_label_modifier
import orcanet.logging as olog
import medgeconv


class Organizer:
    """
    Core class for working with networks in OrcaNet.

    Attributes
    ----------
    cfg : orcanet.core.Configuration
        Contains all configurable options.
    io : orcanet.in_out.IOHandler
        Utility functions for accessing the info in cfg.
    history : orcanet.in_out.HistoryHandler
        For reading and plotting data from the log files created
        during training.

    """
    def __init__(self, output_folder,
                 list_file=None,
                 config_file=None,
                 tf_log_level=None):
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
        list_file : str, optional
            Path to a toml list file with pathes to all the h5 files that should
            be used for training and validation.
            Will be used to extract samples and labels.
        config_file : str, optional
            Path to a toml config file with settings that are used instead of
            the default ones.
        tf_log_level : int/str
            Sets the TensorFlow CPP_MIN_LOG_LEVEL environment variable.
            0 = all messages are logged (default behavior).
            1 = INFO messages are not printed.
            2 = INFO and WARNING messages are not printed.
            3 = INFO, WARNING, and ERROR messages are not printed.

        """
        if tf_log_level is not None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_log_level)

        self.cfg = Configuration(output_folder, list_file, config_file)
        self.io = IOHandler(self.cfg)
        self.history = HistoryHandler(output_folder)

        self.xs_mean = None
        self._auto_label_modifier = None
        self._stored_model = None

    def train_and_validate(self, model=None, epochs=None, to_epoch=None):
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
        model : ks.models.Model or str, optional
            Compiled keras model to use for training. Required for the first
            epoch (the start of training).
            Can also be the path to a saved keras model, which will be laoded.
            If model is None, the most recent saved model will be
            loaded automatically to continue the training.
        epochs : int, optional
            How many epochs should be trained by running this function.
            None for infinite. This includes the current epoch in case it
            is not finished yet, i.e. 1 means complete the epoch if there
            are files left, otherwise do the next epoch.
        to_epoch : int, optional
            Train up to and including this epoch. Can not be used together with
            epochs.

        Returns
        -------
        model : ks.models.Model
            The trained keras model.

        """
        latest_epoch = self.io.get_latest_epoch()

        model = self._get_model(model, logging=False)
        self._stored_model = model

        # check if the validation is missing for the latest fileno
        if latest_epoch is not None:
            state = self.history.get_state()[-1]
            if state["is_validated"] is False and self.val_is_due(latest_epoch):
                self.validate()

        next_epoch = self.io.get_next_epoch(latest_epoch)
        n_train_files = self.io.get_no_of_files("train")

        if to_epoch is None:
            epochs_left = epochs
        else:
            if epochs is not None:
                raise ValueError("Can not give both 'epochs' and 'to_epoch'")
            if latest_epoch is None:
                epochs_left = to_epoch
            else:
                epochs_left = max(
                    0, to_epoch - self.io.get_next_epoch(latest_epoch)[0] + 1)

        trained_epochs = 0
        while epochs_left is None or trained_epochs < epochs_left:
            # Train on remaining files
            for file_no in range(next_epoch[1], n_train_files + 1):
                curr_epoch = (next_epoch[0], file_no)
                self.train(model)
                if self.val_is_due(curr_epoch):
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
        model : ks.models.Model or str, optional
            Compiled keras model to use for training. Required for the first
            epoch (the start of training).
            Can also be the path to a saved keras model, which will be laoded.
            If model is None, the most recent saved model will be
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
            olog.log_start_training(self)

        model_path = self.io.get_model_path(*next_epoch)
        model_path_local = self.io.get_model_path(*next_epoch, local=True)
        if os.path.isfile(model_path):
            raise FileExistsError(
                "Can not train model in epoch {} file {}, this model has "
                "already been saved!".format(*next_epoch))

        smry_logger = olog.SummaryLogger(self, model)

        if self.cfg.learning_rate is not None:
            ks.backend.set_value(
                model.optimizer.lr, self.io.get_learning_rate(next_epoch)
            )

        files_dict = self.io.get_file("train", next_epoch[1])

        line = "Training in epoch {} on file {}/{}".format(
            next_epoch[0], next_epoch[1], self.io.get_no_of_files("train"))
        self.io.print_log(line)
        self.io.print_log("-" * len(line))
        self.io.print_log("Learning rate is at {}".format(
            ks.backend.get_value(model.optimizer.lr)))
        self.io.print_log('Inputs and files:')
        for input_name, input_file in files_dict.items():
            self.io.print_log("   {}: \t{}".format(input_name,
                                                   os.path.basename(
                                                       input_file)))

        start_time = time.time()
        history = backend.train_model(self, model, next_epoch, batch_logger=True)
        elapsed_s = int(time.time() - start_time)

        model.save(model_path)
        smry_logger.write_line(
            next_epoch_float,
            ks.backend.get_value(model.optimizer.lr),
            history_train=history,
        )

        self.io.print_log('Training results:')
        for metric_name, loss in history.items():
            self.io.print_log(f"   {metric_name}: \t{loss}")
        self.io.print_log(f"Elapsed time: {timedelta(seconds=elapsed_s)}")
        self.io.print_log(f"Saved model to: {model_path_local}\n")

        update_summary_plot(self)
        if self.cfg.cleanup_models:
            self.cleanup_models()

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
        smry_logger = olog.SummaryLogger(self, model)

        olog.log_start_validation(self)

        start_time = time.time()
        history = backend.validate_model(self, model)
        elapsed_s = int(time.time() - start_time)

        self.io.print_log('Validation results:')
        for metric_name, loss in history.items():
            self.io.print_log(f"   {metric_name}: \t{loss}")
        self.io.print_log(f"Elapsed time: {timedelta(seconds=elapsed_s)}\n")
        smry_logger.write_line(epoch_float, "n/a", history_val=history)

        update_summary_plot(self)

        if self.cfg.cleanup_models:
            self.cleanup_models()

        return history

    def predict(self, epoch=None, fileno=None, concatenate=False, samples=None):
        """
        Make a prediction if it does not exist yet, and return its filepath.

        Load the model with the lowest validation loss, let it predict on
        all samples of the validation set
        in the toml list, and save this prediction together with all the
        y_values as a h5 file in the predictions subfolder.

        Parameters
        ----------
        epoch : int, optional
            Epoch of a model to load.
        fileno : int, optional
            File number of a model to load.
        concatenate : bool
            Whether the prediction files should also be concatenated.
        samples : int, optional
            Don't use the full validation files, but just the given number
            of samples.

        Returns
        -------
        pred_filename : List
            List to the paths of all created prediction files.
            If concatenate = True, the list always only contains the
            path to the concatenated prediction file.

        """
        if fileno is None and epoch is None:
            epoch, fileno = self.history.get_best_epoch_fileno()
            print(f"Automatically set epoch to epoch {epoch} file {fileno}.")
        elif fileno is None or epoch is None:
            raise ValueError(
                "Either both or none of epoch and fileno must be None")

        is_pred_done = self._check_if_pred_already_done(epoch, fileno)
        if is_pred_done:
            print("Prediction has already been done.")
            pred_filepaths = self.io.get_pred_files_list(epoch, fileno)

        else:
            if self._stored_model is None:
                model = self.load_saved_model(epoch, fileno, logging=False)
            else:
                model = self._stored_model
            self._set_up(model)

            start_time = time.time()
            backend.make_model_prediction(
                self, model, epoch, fileno, samples=samples)
            elapsed_s = int(time.time() - start_time)
            print('Finished predicting on all validation files.')
            print("Elapsed time: {}\n".format(timedelta(seconds=elapsed_s)))

            pred_filepaths = self.io.get_pred_files_list(epoch, fileno)

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
                                            if os.path.isfile(os.path.join(concatenated_folder,
                                                                           file)))
                pred_filepaths = [concatenated_folder + '/' + fname_conc_file_list[0]]

        return pred_filepaths

    def inference(self, epoch=None, fileno=None):
        """
        Make an inference and return the filepaths.

        Load the model with the lowest validation loss, let
        it predict on all samples of the inference set
        in the toml list, and save this prediction as a h5 file in the
        predictions subfolder. y values will only be added if they are in
        the input file, so this can be used on un-labelled data as well.

        Parameters
        ----------
        epoch : int, optional
            Epoch of a model to load [default: lowest val loss].
        fileno : int, optional
            File number of a model to load [default: lowest val loss].

        Returns
        -------
        filenames : list
            List to the paths of all created output files.

        """
        if fileno is None and epoch is None:
            epoch, fileno = self.history.get_best_epoch_fileno()
            print("Automatically set epoch to epoch {} file {}.".format(epoch, fileno))
        elif fileno is None or epoch is None:
            raise ValueError(
                "Either both or none of epoch and fileno must be None")

        if self._stored_model is None:
            model = self.load_saved_model(epoch, fileno, logging=False)
        else:
            model = self._stored_model
        self._set_up(model)

        filenames = []
        for files_dict in self.io.yield_files("inference"):
            # output filename is based on name of file in first input
            first_filename = os.path.basename(list(files_dict.values())[0])
            output_filename = "model_epoch_{}_file_{}_on_{}".format(
                epoch, fileno, first_filename)

            output_path = os.path.join(self.io.get_subfolder("inference"),
                                       output_filename)
            filenames.append(output_path)
            if os.path.exists(output_path):
                warnings.warn("Warning: {} exists already, skipping "
                              "file".format(output_filename))
                continue

            print(f'Working on file {first_filename}')
            start_time = time.time()
            backend.h5_inference(
                self, model, files_dict, output_path, use_def_label=False)
            elapsed_s = int(time.time() - start_time)
            print(f'Finished on file {first_filename} in {elapsed_s/60} min')

        return filenames

    def cleanup_models(self):
        """
        Delete all models except for the the most recent one (to continue
        training), and the ones with the highest and lowest loss/metrics.

        """
        all_epochs = self.io.get_all_epochs()
        epochs_to_keep = {self.io.get_latest_epoch(), }
        try:
            for metric in self.history.get_metrics():
                epochs_to_keep.add(
                    self.history.get_best_epoch_fileno(
                        metric=f"val_{metric}", mini=True))
                epochs_to_keep.add(
                    self.history.get_best_epoch_fileno(
                        metric=f"val_{metric}", mini=False))
        except ValueError:
            # no best epoch exists
            pass

        for epoch in epochs_to_keep:
            if epoch not in all_epochs:
                warnings.warn(
                    f"ERROR: keeping_epoch {epoch} not in available epochs {all_epochs}, "
                    f"skipping clean-up of models!")
                return

        print("\nClean-up saved models:")
        for epoch in all_epochs:
            model_path = self.io.get_model_path(epoch[0], epoch[1])
            model_name = os.path.basename(model_path)
            if epoch in epochs_to_keep:
                print("Keeping model {}".format(model_name))
            else:
                print("Deleting model {}".format(model_name))
                os.remove(model_path)

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
        elif latest_pred_file_no == total_no_of_val_files:
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
            Epoch of the saved model. If both this and fileno are -1,
            load the most recent model.
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
        return self._load_model(path_of_model)

    def _get_model(self, model, logging=False):
        """ Load most recent saved model or use user model. """
        latest_epoch = self.io.get_latest_epoch()

        if latest_epoch is None:
            # new training, log info about model
            if model is None:
                raise ValueError("You need to provide a compiled keras model "
                                 "for the start of the training! (You gave None)")

            elif isinstance(model, str):
                # path to a saved model
                self.io.print_log("Loading model from " + model, logging=logging)
                model = self._load_model(model)

            if logging:
                self._save_as_json(model)
                model.summary(print_fn=self.io.print_log)

                try:
                    plots_folder = self.io.get_subfolder("plots", create=True)
                    ks.utils.plot_model(
                        model, plots_folder + "/model_plot.png", show_shapes=True)
                except (ImportError, AttributeError) as e:
                    # TODO remove AttributeError once https://github.com/tensorflow/tensorflow/issues/38988 is fixed
                    warnings.warn("Can not plot model: " + str(e))

        else:
            # resuming training, load model if it is not given
            if model is None:
                model = self.load_saved_model(*latest_epoch, logging=logging)

            elif isinstance(model, str):
                # path to a saved model
                self.io.print_log("Loading model from " + model, logging=logging)
                model = self._load_model(model)

        return model

    def _load_model(self, filepath):
        """ Load from path, with custom objects and parallized. """
        def ks_load():
            return ks.models.load_model(
                filepath, custom_objects=self.cfg.get_custom_objects())

        if self.cfg.multi_gpu and len(
                tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f'Number of GPUs: {strategy.num_replicas_in_sync}')
            with strategy.scope():
                model = ks_load()
        else:
            model = ks_load()
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
        if self.cfg.get_list_file() is None:
            raise ValueError("No files specified. Need to load a toml "
                             "list file with pathes to h5 files first.")

        if self.cfg.label_modifier is None:
            self._auto_label_modifier = get_auto_label_modifier(model)

        if self.cfg.zero_center_folder is not None:
            self.get_xs_mean(logging)

    def val_is_due(self, epoch=None):
        """
        True if validation is due on given epoch according to schedule.
        Does not check if it has been done already.

        """
        if epoch is None:
            epoch = self.io.get_latest_epoch()
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
    class_weight : dict or None
        Optional dictionary mapping class indices (integers) to a weight
        (float) value, used for weighting the loss function (during
        training only). This can be useful to tell the model to
        "pay more attention" to samples from an under-represented class.
    cleanup_models : bool
        If true, will only keep the best (in terms of val loss) and the most
        recent from all saved models in order to save disk space.
    custom_objects : dict, optional
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered by keras during deserialization of models.
    dataset_modifier : function or None
        For orga.predict: Function that determines which datasets get created
        in the resulting h5 file. If none, every output layer will get one
        dataset each for both the label and the prediction, and one dataset
        containing the y_values from the validation files.
    fixed_batchsize : bool
        The last batch in the file might be smaller then the batchsize.
        Usually, this is no problem, but set to True to skip this batch
        [default: False].
    key_x_values : str
        The name of the datagroup in the h5 input files which contains
        the samples for the network.
    key_y_values : str
        The name of the datagroup in the h5 input files which contains
        the info for the labels.
    label_modifier : function or None
        Operation to be performed on batches of y_values read from the input
        files before they are fed into the model as labels. If None is given,
        all y_values with the same name as the output layers will be passed
        to the model as a dict, with the keys being the dtype names.
    learning_rate : float, tuple, function, str (optional)
        The learning rate for the training.
        If None is given, don't change the learning rate at all.
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
    multi_gpu : bool
        Use all availble GPUs (distributed training if theres more then one).
    n_events : None or int
        For testing purposes. If not the whole .h5 file should be used for
        training, define the number of samples.
    sample_modifier : function or None
        Operation to be performed on batches of x_values read from the input
        files before they are fed into the model as samples.
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
    def __init__(self, output_folder, list_file=None, config_file=None, **kwargs):
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
        kwargs
            Overwrites the values given in the config file.

        """
        self.batchsize = 64
        self.learning_rate = None

        self.zero_center_folder = None
        self.validate_interval = None
        self.cleanup_models = False
        self.class_weight = None

        self.sample_modifier = None
        self.dataset_modifier = None
        self.label_modifier = None

        self.key_x_values = "x"
        self.key_y_values = "y"
        self.custom_objects = {}
        self.shuffle_train = False

        self.fixed_batchsize = False
        self.callback_train = []
        self.use_scratch_ssd = False
        self.verbose_train = 1
        self.verbose_val = 0
        self.make_weight_plots = False  # Removed in v0.11.1
        self.n_events = None
        self.max_queue_size = 10
        self.train_logger_display = 100
        self.train_logger_flush = -1
        self.multi_gpu = True

        self._default_values = dict(self.__dict__)

        # Main folder:
        if output_folder[-1] == "/":
            self.output_folder = output_folder
        else:
            self.output_folder = output_folder+"/"

        # Private attributes:
        self._files_dict = {
            "train": None,
            "val": None,
            "inference": None,
        }
        self._list_file = None

        # Load the optionally given list and config files.
        if list_file is not None:
            self.import_list_file(list_file)
        if config_file is not None:
            self.update_config(config_file)

        # set given kwargs:
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise AttributeError(
                    "Unknown attribute {}".format(key))

        # deprecation warning TODO remove in the future
        if self.make_weight_plots:
            warnings.warn("make_weight_plots was removed in version v0.11.1")

    def import_list_file(self, list_file):
        """
        Import the filepaths of the h5 files from a toml list file.

        Parameters
        ----------
        list_file : str
            Path to the toml list file.

        """
        if self._list_file is not None:
            raise ValueError("Can not load list file: Has already been loaded! "
                             "({})".format(self._list_file))

        file_content = toml.load(list_file)

        name_mapping = {
            "train_files": "train",
            "validation_files": "val",
            "inference_files": "inference",
        }

        for toml_name, files_dict_name in name_mapping.items():
            files = _extract_filepaths(file_content, toml_name)
            self._files_dict[files_dict_name] = files or None

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
            Either "train", "val" or "inference".

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
        if which not in self._files_dict.keys():
            raise NameError("Unknown fileset name ", which)
        if self._files_dict[which] is None:
            raise AttributeError("No {} files have been specified!".format(which))
        return self._files_dict[which]

    def get_custom_objects(self):
        """ Get user custom objects + orcanet internal ones. """
        orcanet_co = medgeconv.custom_objects
        return {**orcanet_co, **self.custom_objects}


def _get_h5_files(folder):
    h5files = []
    for f in os.listdir(folder):
        if f.endswith(".h5"):
            h5files.append(os.path.join(folder, f))
    h5files.sort()
    if not h5files:
        warnings.warn(f"No .h5 files in dir {folder}!")
    return h5files


def _extract_filepaths(file_content, which):
    """
    Get train, val or inf filepaths of all inputs from a toml readout.
    Makes sure that all input have the same number of files.

    """
    # alternative names to write in the toml file
    aliases = {
        "train_files": ("training_files", "train", "training"),
        "validation_files": ("val_files", "val", "validation"),
        "inference_files": ("inf_files", "inf", "inference"),
    }
    assert which in aliases.keys(), f"{which} not in {list(aliases.keys())}"

    def get_alias(ident):
        for k, v in aliases.items():
            if ident == k or ident in v:
                return k
        else:
            raise NameError(
                f"Unknown argument '{ident}' in toml file: "
                f"Must be either of {list(aliases.keys())}"
            )

    files = {}
    n_files = []
    for input_name, input_files in file_content.items():
        for filetype, filetyp_files in input_files.items():
            if get_alias(filetype) != which:
                continue
            # if a dir is given as a filepath, use all h5 files in that dir instead
            expanded_files = []
            for path in filetyp_files:
                if os.path.isdir(path):
                    expanded_files.extend(_get_h5_files(path))
                else:
                    expanded_files.append(path)
            files[input_name] = tuple(expanded_files)
            # store number of files for this output
            n_files.append(len(expanded_files))

    if n_files and n_files.count(n_files[0]) != len(n_files):
        raise ValueError(
            "Input with different number of {} in toml list".format(which))

    return files
