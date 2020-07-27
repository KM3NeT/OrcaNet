import h5py
import time
import numpy as np
import tensorflow.keras as ks


class Hdf5BatchGenerator(ks.utils.Sequence):
    def __init__(self, files_dict,
                 batchsize=64,
                 key_x_values="x",
                 key_y_values="y",
                 sample_modifier=None,
                 label_modifier=None,
                 fixed_batchsize=False,
                 phase="training",
                 xs_mean=None,
                 f_size=None,
                 keras_mode=True,
                 shuffle=False,
                 class_weights=None):
        """
        Yields batches of input data from h5 files.

        This will go through one file, or multiple files in parallel, and yield
        one batch of data, which can then be used as an input to a model.
        Since multiple filepaths can be given to read out in parallel,
        this can also be used for models with multiple inputs.

        Parameters
        ----------
        files_dict : dict
            Pathes of the files to train on.
            Keys: The name of every input (from the toml list file, can be multiple).
            Values: The filepath of a single h5py file to read data from.
        batchsize : int
            Batchsize that will be used for reading data from the files.
        key_x_values : str
            The name of the datagroup in the h5 input files which contains
            the samples for the network.
        key_y_values : str
            The name of the datagroup in the h5 input files which contains
            the info for the labels. If this name is not in the file,
            y_values will be set to None.
        sample_modifier : function or None
            Operation to be performed on batches of samples read from the input
            files before they are fed into the model.
        label_modifier : function or None
            Operation to be performed on batches of labels read from the input files
            before they are fed into the model.
        fixed_batchsize : bool
            The last batch in the file might be smaller then the batchsize.
            Usually, this is no problem, but set to True to skip this batch.
        xs_mean : ndarray or None
            Zero center image to be subtracted from data as preprocessing.
        f_size : int or None
            Specifies the number of samples to be read from the .h5 file.
            If none, the whole .h5 file will be used.
        keras_mode : bool
            If true, yield xs and ys (samples and labels) for the keras fit
            generator function.
            If false, yield the info_blob containing the full sample and label
            info, both before and after the modifiers have been applied.
        shuffle : bool
            Randomize the order in which batches are read from the file
            (once during init). Can reduce read out speed.

        """
        if phase not in ("training", "validation", "inference"):
            raise ValueError("Invalid phase")
        self.files_dict = files_dict
        self.batchsize = batchsize
        self.key_x_values = key_x_values
        self.key_y_values = key_y_values
        self.sample_modifier = sample_modifier
        self.label_modifier = label_modifier
        self.fixed_batchsize = fixed_batchsize
        self.phase = phase
        self.xs_mean = xs_mean
        self.f_size = f_size
        self.keras_mode = keras_mode
        self.shuffle = shuffle
        self.class_weights = class_weights

        # a dict with the names of list inputs as keys, and the opened
        # h5 files as values
        self._files = {}
        # start index of each batch in the file
        self._sample_pos = None
        # total number of samples per file
        self._total_f_size = None

        # for keeping track of the readout speed
        self._total_time = 0.
        self._total_batches = 0

        self.open()

    def __len__(self):
        """ Number of batches in the Sequence (includes queue). """
        return len(self._sample_pos)

    def __getitem__(self, index):
        """
        Gets batch number `index`.

        Returns
        -------
        xs : dict
            Samples for the model train on.
            Keys : str
                The name(s) of the input layer(s) of the model.
            Values : ndarray
                A batch of samples for the corresponding input.
        ys : dict or None
            Labels for the model to train on. Will be None if there are
            no labels in the file.
            Keys : str
                The name(s) of the output layer(s) of the model.
            Values : ndarray
                A batch of labels for the corresponding output.

        If class_weights is not None, will return aditionally:
        sample_weights : dict
            Maps output names to weights for each sample in the batch as a
            np.array.

        If keras_mode is False, will return instead:
        info_blob : dict
            Blob containing the x_values, y_values, xs and ys, and optionally
            the sample_weights.

        """
        start_time = time.time()
        file_index = self._sample_pos[index]
        info_blob = {"phase": self.phase}
        info_blob["x_values"] = self.get_x_values(file_index)
        info_blob["y_values"] = self.get_y_values(file_index)

        # Modify the samples
        if self.sample_modifier is not None:
            xs = self.sample_modifier(info_blob)
        else:
            xs = info_blob["x_values"]
        info_blob["xs"] = xs

        # Modify the labels
        if info_blob["y_values"] is not None and self.label_modifier is not None:
            ys = self.label_modifier(info_blob)
        else:
            ys = None
        info_blob["ys"] = ys

        if self.fixed_batchsize:
            self.pad_to_size(info_blob)

        if self.class_weights is not None:
            info_blob["sample_weights"] = _get_sample_weights(ys, self.class_weights)

        self._total_time += time.time() - start_time
        self._total_batches += 1
        if self.keras_mode:
            if info_blob.get("sample_weights"):
                return xs, ys, info_blob["sample_weights"]
            else:
                return xs, ys
        else:
            return info_blob

    def pad_to_size(self, info_blob):
        """ Pad the batch to have a fixed batchsize. """
        org_batchsize = len(next(iter(info_blob["xs"].values())))
        if org_batchsize == self.batchsize:
            return
        info_blob["org_batchsize"] = org_batchsize
        for input_key, x in info_blob["xs"].items():
            info_blob["xs"][input_key] = _pad_to_size(x, self.batchsize)
        if info_blob.get("ys") is not None:
            for output_key, y in info_blob["ys"].items():
                info_blob["ys"][output_key] = _pad_to_size(y, self.batchsize)

    def open(self):
        """ Open all files and prepare for read out. """
        for input_key, file in self.files_dict.items():
            self._files[input_key] = h5py.File(file, 'r')
        self._store_file_length()
        self._store_batch_indices()

    def close(self):
        """ Close all files again. """
        for f in list(self._files.values()):
            f.close()

    def get_x_values(self, start_index):
        """
        Read one batch of samples from the files and zero center.

        Parameters
        ----------
        start_index : int
            The start index in the h5 files at which the batch will be read.
            The end index will be the start index + the batch size.

        Returns
        -------
        x_values : dict
            One batch of data for each input file.

        """
        x_values = {}
        for input_key, file in self._files.items():
            x_values[input_key] = file[self.key_x_values][
                      start_index: start_index + self._batchsize]
            if self.xs_mean is not None:
                x_values[input_key] = np.subtract(
                    x_values[input_key], self.xs_mean[input_key])

        return x_values

    def get_y_values(self, start_index):
        """
        Get y_values for the nn. Since the y_values are hopefully the same
        for all the files, use the ones from the first. TODO add check

        Parameters
        ----------
        start_index : int
            The start index in the h5 files at which the batch will be read.
            The end index will be the start index + the batch size.

        Returns
        -------
        y_values : ndarray
            The y_values, right from the files.

        """
        first_file = list(self._files.values())[0]
        try:
            y_values = first_file[self.key_y_values][
                       start_index:start_index + self._batchsize]
        except KeyError:
            # can not look up y_values, lets hope we dont need them
            y_values = None
        return y_values

    def print_timestats(self, print_func=None):
        """ Print stats about how long it took to read batches. """
        if print_func is None:
            print_func = print
        print_func("Statistics of data readout:")
        print_func(f"\tTotal time:\t{self._total_time/60:.2f} min")
        if self._total_batches != 0:
            print_func(
                f"\tPer batch:\t"
                f"{1000 * self._total_time/self._total_batches:.5} ms"
            )

    @property
    def _size(self):
        """ Size of the files that will be read in. Can be smaller than the actual
        file size if defined by user. """
        if self.f_size is None:
            return self._total_f_size
        else:
            return self.f_size

    @property
    def _batchsize(self):
        """
        Return the effective batchsize. Can be smaller than the user defined
        one if it would be larger than the size of the file.
        """
        if self._size < self.batchsize:
            return self._size
        else:
            return self.batchsize

    def _store_file_length(self):
        """
        Make sure all files have the same length and store this length.
        """
        lengths = []
        for f in list(self._files.values()):
            lengths.append(len(f[self.key_x_values]))

        if not lengths.count(lengths[0]) == len(lengths):
            self.close()
            raise ValueError("All data files must have the same length! "
                             "Given were:\n " + str(lengths))

        self._total_f_size = lengths[0]

    def _store_batch_indices(self):
        """
        Define the start indices of each batch in the h5 file and store this.
        """
        if self.fixed_batchsize and self.phase != "inference":
            total_no_of_batches = np.floor(self._size / self._batchsize)
        else:
            total_no_of_batches = np.ceil(self._size / self._batchsize)

        sample_pos = np.arange(int(total_no_of_batches)) * self._batchsize
        if self.shuffle:
            np.random.shuffle(sample_pos)

        self._sample_pos = sample_pos


def _get_sample_weights(ys, class_weights):
    """
    Produce a weight for each sample given the weight for each class.

    Parameters
    ----------
    ys : dict
        Maps output names to categorical one-hot labels as np.arrays.
        Expected to be 2D (n_samples, n_classes).
    class_weights : dict
        Maps output neuron numbers to weights as floats.

    Returns
    -------
    sample_weights : dict
        Maps output names to weights for each sample in the batch as a
        np.array.

    """
    sample_weights = {}
    for output_name, labels in ys.items():
        class_weights_arr = np.ones(labels.shape[1])
        for k, v in class_weights.items():
            class_weights_arr[int(k)] = v
        labels_class = np.argmax(labels, axis=-1)
        sample_weights[output_name] = class_weights_arr[labels_class]
    return sample_weights


def get_h5_generator(orga, files_dict, f_size=None, zero_center=False,
                     keras_mode=True, shuffle=False, use_def_label=True,
                     phase="training"):
    """
    Initialize the hdf5_batch_generator_base with the paramters in orga.cfg.

    Parameters
    ----------
    orga : orcanet.core.Organizer
        Contains all the configurable options in the OrcaNet scripts.
    files_dict : dict
        Pathes of the files to train on.
        Keys: The name of every input (from the toml list file, can be multiple).
        Values: The filepath of a single h5py file to read samples from.
    f_size : int or None
        Specifies the number of samples to be read from the .h5 file.
        If none, the whole .h5 file will be used.
    zero_center : bool
        Whether to use zero centering.
        Requires orga.zero_center_folder to be set.
    keras_mode : bool
        Specifies if mc-infos (y_values) should be yielded as well. The
        mc-infos are used for evaluation after training and testing is finished.
    shuffle : bool
        Randomize the order in which batches are read from the file.
        Significantly reduces read out speed.
    use_def_label : bool
        If True and no label modifier is given by user, use the default
        label modifier instead of none.

    Yields
    ------
    xs : dict
        Data for the model train on.
            Keys : str  The name(s) of the input layer(s) of the model.
            Values : ndarray    A batch of samples for the corresponding input.
    ys : dict or None
        Labels for the model to train on.
            Keys : str  The name(s) of the output layer(s) of the model.
            Values : ndarray    A batch of labels for the corresponding output.
        Will be None if there are no labels in the file.
    y_values : ndarray, optional
        Y values from the file. Only yielded if yield_mc_info is True.

    """
    if orga.cfg.label_modifier is not None:
        label_modifier = orga.cfg.label_modifier
    elif use_def_label:
        assert orga._auto_label_modifier is not None, \
            "Auto label modifier has not been set up (can be done with " \
            "nn_utilities.get_auto_label_modifier)"
        label_modifier = orga._auto_label_modifier
    else:
        label_modifier = None

    # get xs_mean or load/create if not stored yet
    if zero_center:
        xs_mean = orga.get_xs_mean()
    else:
        xs_mean = None

    generator = Hdf5BatchGenerator(
        files_dict=files_dict,
        batchsize=orga.cfg.batchsize,
        key_x_values=orga.cfg.key_x_values,
        key_y_values=orga.cfg.key_y_values,
        sample_modifier=orga.cfg.sample_modifier,
        label_modifier=label_modifier,
        phase=phase,
        xs_mean=xs_mean,
        f_size=f_size,
        keras_mode=keras_mode,
        shuffle=shuffle,
        class_weights=orga.cfg.class_weight,
        fixed_batchsize=orga.cfg.fixed_batchsize,
    )

    return generator


def _pad_to_size(x, size):
    """ Pad x to given size along axis 0 by repeating last element. """
    if len(x) > size:
        raise ValueError(f"Can't pad x with shape {x.shape} to length {size}")
    elif len(x) == size:
        return x
    else:
        return np.concatenate((x, np.broadcast_to(
                x[-1], (size - len(x),) + x.shape[1:])), axis=0)
