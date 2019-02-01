#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core scripts for the OrcaNet package.
"""

import os
import warnings
import keras as ks
import h5py

from orcanet.backend import train_and_validate_model, make_model_evaluation
from orcanet.in_out import read_out_list_file, read_out_config_file, read_out_model_file, use_node_local_ssd_for_input, write_full_logfile_startup, h5_get_number_of_rows
from orcanet.utilities.nn_utilities import load_zero_center_data, get_inputs, generate_batches_from_hdf5_file, get_auto_label_modifier


class Configuration(object):
    """
    Container object for all the configurable options in the OrcaNet scripts. TODO custom loss functions

    Sensible default values were chosen for the settings.
    You can change the all of these public attributes (the ones without a leading underscore _) either directly or with a
    .toml config file via the method set_from_config_file().

    Attributes
    ----------
    main_folder : str
        Name of the folder of this model in which everything will be saved, e.g., the summary.txt log file is located in here.
        Has a '/' at the end.
    batchsize : int
        Batchsize that should be used for the training and validation of the network.
    custom_objects : list or None
        Optional dictionary mapping names (strings) to custom classes or functions to be considered by keras, e.g.
        during deserialization of models.
    dataset_modifier : function or None
        For orca_eval: Function that determines which datasets get created in the resulting h5 file.
        If none, every output layer will get one dataset each for both the label and the prediction, and one dataset
        containing the mc_info from the validation files. TODO online doc
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
    key_samples : str
        The name of the datagroup in your h5 input files which contains the samples to the network.
    key_labels : str
        The name of the datagroup in your h5 input files which contains the labels to the network.
    label_modifier : function or None
        Operation to be performed on batches of labels read from the input files before they are fed into the model.
        If None is given, all labels with the same name as the output layers will be passed to the model as a dict,
        with the keys being the dtype names.
        TODO online doc on how to do this
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
        For testing purposes. If not the whole .h5 file should be used for training, define the number of events.
    n_gpu : tuple(int, str)
        Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov') [1].
    sample_modifier : function or None
        Operation to be performed on batches of samples read from the input files before they are fed into the model.
        TODO online doc on how to do this
    shuffle_train : bool
        If true, the order at which batches are read out from the files during training are randomized each time.
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
    _auto_label_modifier : None or function
        If no label modifier has been specified by the user, use an auto one instead.
    _train_files : dict or None
        A dict containing the paths to the different training files on which the model will be trained on.
        Example for the format for two input sets with two files each:
                {
                 "input_A" : ('path/to/set_A_train_file_1.h5', 'path/to/set_A_train_file_2.h5'),
                 "input_B" : ('path/to/set_B_train_file_1.h5', 'path/to/set_B_train_file_2.h5'),
                }
    _val_files : dict or None
        Like train_files but for the validation files.
    _list_file : str or None
        Path to the list file that was used to set the training and validation files. Is None if no list file
        has been used.
    _modeldata : namedtuple or None
        Optional info only required for building a predefined model with OrcaNet. [default: None]
        It is not needed for executing orcatrain. It is set via self.load_from_model_file.

        modeldata.nn_arch : str
            Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
        modeldata.loss_opt : tuple(dict, dict/str/None,)
            Tuple that contains 1) the loss_functions and loss_weights as dicts (this is the losses table from the toml file)
            and 2) the metrics.
        modeldata.class_type : str
            Declares the number of output classes / regression variables and a string identifier to specify the exact output classes.
            I.e. (2, 'track-shower')
        modeldata.str_ident : str
            Optional string identifier that gets appended to the modelname. Useful when training models which would have
            the same modelname. Also used for defining models and projections!
        modeldata.swap_4d_channels : None or str
            For 4D data input (3.5D models). Specifies, if the channels of the 3.5D net should be swapped.
            Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X, TODO add multi input options
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
            Path to a toml list file with pathes to all the h5 files that should be used for training and validation.
        config_file : str or None
            Path to a toml config file with attributes that are used instead of the default ones.

        """
        # Configuration:
        self.batchsize = 64
        self.custom_objects = None
        self.dataset_modifier = None
        self.epochs_to_train = -1
        self.filter_out_tf_garbage = True
        self.initial_epoch = -1
        self.initial_fileno = -1
        self.key_samples = "x"
        self.key_labels = "y"
        self.label_modifier = None
        self.learning_rate = 0.001
        self.max_queue_size = 10
        self.n_events = None
        self.n_gpu = (1, 'avolkov')
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
        if main_folder[-1] == "/":
            self.main_folder = main_folder
        else:
            self.main_folder = main_folder+"/"

        # Private attributes:
        self._auto_label_modifier = None
        self._train_files = None
        self._val_files = None
        self._list_file = None
        self._modeldata = None

        # Load the optionally given list and config files.
        if list_file is not None:
            self.set_from_list_file(list_file)
        if config_file is not None:
            self.set_from_config_file(config_file)

    def set_from_list_file(self, list_file):
        """
        Set filepaths to the ones given in a list file.

        Parameters
        ----------
        list_file : str or None
            Path to a toml list file with pathes to all the h5 files that should be used for training and validation.

        """
        if self._list_file is None:
            self._train_files, self._val_files = read_out_list_file(list_file)
            # Save internally which path was used to load the info
            self._list_file = list_file
        else:
            raise AssertionError("You tried to load filepathes from a list file, but pathes have already been loaded \
            for this object. (From the file " + self._list_file + ")\nYou should not use \
            two different list files for one Configuration object!")

    def set_from_config_file(self, config_file):
        """
        Overwrite default attribute values with values from a config file.

        Parameters
        ----------
        config_file : str or None
            Path to a toml config file with attributes that are used instead of the default ones.

        """
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
        self._modeldata = read_out_model_file(model_file)

    def get_latest_epoch(self):
        """
        Check all saved models in the ./saved_models folder and return the highest epoch / file_no pair.

        Will only consider files that end with .h5 as models.

        Returns
        -------
        latest_epoch : tuple
            The highest epoch, file_no pair. (0,0) if the folder is empty or does not exist yet.

        """
        if os.path.exists(self.main_folder + "saved_models"):
            files = []
            for file in os.listdir(self.main_folder + "saved_models"):
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
        elif epoch[1] == self.get_no_of_train_files():
            next_epoch = (epoch[0] + 1, 1)
        else:
            next_epoch = (epoch[0], epoch[1] + 1)
        return next_epoch

    def get_model_path(self, epoch, fileno):
        """
        Get the path to a model (which might not exist yet). TODO make so that -1-1 will give latest?

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
        model_filename = self.main_folder + 'saved_models/model_epoch_' + str(epoch) + '_file_' + str(fileno) + '.h5'
        return model_filename

    def get_eval_path(self, epoch, fileno, list_name):
        """ Get the path to a saved evaluation. """
        eval_filename = self.main_folder \
                        + 'evaluations/pred_model_epoch_{}_file_{}_on_{}_val_files.h5'.format(epoch, fileno, list_name)
        return eval_filename

    def use_local_node(self):
        """
        Copies the test and val files to the node-local ssd scratch folder and sets the new filepaths of the train and val data.
        Speeds up I/O and reduces RRZE network load.
        """
        train_files_ssd, test_files_ssd = use_node_local_ssd_for_input(self.get_train_files(), self.get_val_files())
        self._train_files = train_files_ssd
        self._val_files = test_files_ssd

    def make_folder_structure(self):
        """
        Make subfolders for a specific model if they don't exist already. These subfolders will contain e.g. saved models,
        logfiles, etc.

        """
        main_folder = self.main_folder
        folders_to_create = [main_folder + "log_train", main_folder + "saved_models",
                             main_folder + "plots/activations", main_folder + "evaluations"]
        for directory in folders_to_create:
            if not os.path.exists(directory):
                print("Creating directory: " + directory)
                os.makedirs(directory)

    def get_train_files(self):
        assert self._train_files is not None, "No train files have been specified!"
        return self._train_files

    def get_val_files(self):
        assert self._val_files is not None, "No validation files have been specified!"
        return self._val_files

    def get_n_bins(self):
        """
        Get the number of bins from the training files.

        Only the first files are looked up, the others should be identical.

        Returns
        -------
        n_bins : dict
            Toml-list input names as keys, list of the bins as values.

        """
        train_files = self.get_train_files()
        n_bins = {}
        for input_key in train_files:
            with h5py.File(train_files[input_key][0], "r") as f:
                n_bins[input_key] = f[self.key_samples].shape[1:]
        return n_bins

    def get_multiple_inputs(self):
        """
        Return True when seperate sets of input files were given (e.g. for networks taking data
        simulataneosly from different files).

        """
        train_files = self.get_train_files()
        return len(train_files) > 1

    def get_modeldata(self):
        return self._modeldata

    def get_list_file(self):
        return self._list_file

    def get_train_file_sizes(self):
        """
        Get the number of samples in each input file.
        # TODO only uses the no of samples of the first input! check if the others are the same % batchsize

        Returns
        -------
        file_sizes : list
            Its length is equal to the number of files in each input set.

        """
        train_files = self.get_train_files()
        file_sizes = []
        for file in train_files[list(train_files.keys())[0]]:
            file_sizes.append(h5_get_number_of_rows(file))
        return file_sizes

    def get_val_file_sizes(self):
        """
        Get the number of samples in each input file.
        # TODO only uses the no of samples of the first input! check if the others are the same % batchsize

        Returns
        -------
        file_sizes : list
            Its length is equal to the number of files in each input set.

        """
        val_files = self.get_val_files()
        file_sizes = []
        for file in val_files[list(val_files.keys())[0]]:
            file_sizes.append(h5_get_number_of_rows(file))
        return file_sizes

    def get_no_of_train_files(self):
        """
        Return the number of train files.

        Only looks up the no of files of one (random) list input, as equal length is checked during read in.

        Returns
        -------
        no_of_files : int
            The number of files.

        """
        train_files = self.get_train_files()
        no_of_files = len(list(train_files.values())[0])
        return no_of_files

    def get_no_of_val_files(self):
        """
        Return the number of val files.

        Only looks up the no of files of one (random) list input, as equal length is checked during read in.

        Returns
        -------
        no_of_files : int
            The number of files.

        """
        val_files = self.get_val_files()
        no_of_files = len(list(val_files.values())[0])
        return no_of_files

    def yield_train_files(self):
        """
        Yield a train file for every input.

        Yields
        ------
        files_dict : dict
            The name of every toml list input as a key, one of the filepaths as values.
            They will be yielded in the same order as they are given in the toml file.

        """
        train_files = self.get_train_files()
        for file_no in range(self.get_no_of_train_files()):
            files_dict = {key: train_files[key][file_no] for key in train_files}
            yield files_dict

    def yield_val_files(self):
        """
        Yield a validation file for every input.

        Yields
        ------
        files_dict : dict
            The name of every toml list input as a key, one of the filepaths as values.
            They will be yielded in the same order as they are given in the toml file.

        """
        val_files = self.get_val_files()
        for file_no in range(self.get_no_of_val_files()):
            files_dict = {key: val_files[key][file_no] for key in val_files}
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

        if self.sample_modifier is not None:
            modified_shapes = {modi_key: xs[modi_key].shape[1:] for modi_key in xs}
            print("After applying your sample modifier, they have the following names and shapes:")
            for list_key in modified_shapes:
                print("\t{}\t{}".format(list_key, modified_shapes[list_key]))
            list_inp_shapes = modified_shapes

        print("Your model requires the following input names and shapes:")
        for layer_key in layer_inp_shapes:
            print("\t{}\t{}".format(layer_key, layer_inp_shapes[layer_key]))

        err_inp_names, err_inp_shapes = check_for_error(list_inp_shapes, layer_inp_shapes)

        err_msg = ""
        if len(err_inp_names) == 0 and len(err_inp_shapes) == 0:
            print("Check passed.\n")
        else:
            if len(err_inp_names) != 0:
                err_msg += "No matching input name from the list file for input layer(s): " \
                           + (", ".join(str(e) for e in err_inp_names) + "\n")
            if len(err_inp_shapes) != 0:
                err_msg += "Shapes of layers and labels do not match for the following input layer(s): " \
                           + (", ".join(str(e) for e in err_inp_shapes) + "\n")
            print("Error:", err_msg)

        # ----------------------------------
        print("\nOutput check\n------------")
        # tuple of strings
        mc_names = y_values.dtype.names
        print("The following {} label names are in your toml list file:".format(len(mc_names)))
        print("\t" + ", ".join(str(name) for name in mc_names), end="\n\n")

        if self.label_modifier is not None:
            label_names = tuple(self.label_modifier(y_values).keys())
            print("The following {} labels get produced from them by your label_modifier:".format(len(label_names)))
            print("\t" + ", ".join(str(name) for name in label_names), end="\n\n")
        else:
            label_names = mc_names
            print("Since you did not specify a label_modifier, the output layers will be provided with "
                  "labels that match their name from the above.\n\n")

        # tuple of strings
        loss_names = tuple(model.loss.keys())
        print("Your model has the following {} output layers with loss functions:".format(len(loss_names)))
        print("\t" + ", ".join(str(name) for name in loss_names), end="\n\n")

        err_out_names = []
        for loss_name in loss_names:
            if loss_name not in label_names:
                err_out_names.append(loss_name)

        if len(err_out_names) == 0:
            print("Check passed.\n")
        else:
            if len(err_out_names) != 0:
                err_msg += "No matching label name from the list file for output layer(s): " \
                           + (", ".join(str(e) for e in err_out_names) + "\n")
            print("Error:", err_msg)

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
        files_dict = next(self.yield_train_files())
        generator = generate_batches_from_hdf5_file(self, files_dict, yield_mc_info=mc_info)
        return generator

    def get_batch(self):
        """ For testing purposes, return a batch of samples and mc_infos. """
        files_dict = next(self.yield_train_files())
        xs = {}
        for i, inp_name in enumerate(files_dict):
            with h5py.File(files_dict[inp_name]) as f:
                xs[inp_name] = f[self.key_samples][:self.batchsize]
                if i == 0:
                    mc_info = f[self.key_labels][:self.batchsize]
        return xs, mc_info


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

    # Epoch at this point is the epoch of the currently existing model (or 0,0 if there is none)
    if epoch[0] == 0 and epoch[1] == 0:
        assert initial_model is not None, "You need to provide a compiled keras model for the start of the training! (You gave None)"
        model = initial_model
    else:
        # Load an existing model
        if initial_model is not None:
            warnings.warn("You provided a model even though this is not the start of the training. Provided model is ignored!")
        path_of_model = cfg.get_model_path(epoch[0], epoch[1])
        print("Loading saved model: "+path_of_model)
        model = ks.models.load_model(path_of_model, custom_objects=cfg.custom_objects)  # get_all_loss_functions()

    if cfg.label_modifier is None:
        cfg._auto_label_modifier = get_auto_label_modifier(model)
    cfg.check_connections(model)
    # model.summary()
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
    Evaluate a model on all samples of the validation set in the toml list, and save it as a h5 file.

    The cfg.initial_epoch and cfg.initial_fileno parameters define which model is loaded.

    Parameters
    ----------
    cfg : object Configuration
        Configuration object containing all the configurable options in the OrcaNet scripts.

    """
    if cfg.filter_out_tf_garbage:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    list_name = os.path.basename(cfg.get_list_file()).split(".")[0]  # TODO make foolproof
    epoch = (cfg.initial_epoch, cfg.initial_fileno)

    if epoch[0] == -1 and epoch[1] == -1:
        epoch = cfg.get_latest_epoch()
        print("Automatically set epoch to epoch {} file {}.".format(epoch[0], epoch[1]))

    if cfg.zero_center_folder is not None:
        xs_mean = load_zero_center_data(cfg)
    else:
        xs_mean = None

    if cfg.use_scratch_ssd:
        cfg.use_local_node()

    path_of_model = cfg.get_model_path(epoch[0], epoch[1])
    model = ks.models.load_model(path_of_model, custom_objects=cfg.custom_objects)

    eval_filename = cfg.get_eval_path(epoch[0], epoch[1], list_name)
    make_model_evaluation(cfg, model, xs_mean, eval_filename, samples=None)
