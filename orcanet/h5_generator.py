import h5py
import numpy as np


class Hdf5BatchGenerator:
    def __init__(self, files_dict,
                 batchsize=64,
                 key_x_values="x",
                 key_y_values="y",
                 sample_modifier=None,
                 label_modifier=None,
                 xs_mean=None,
                 max_queue_size=10,
                 f_size=None,
                 keras_mode=True,
                 shuffle=False):
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
            Batchsize that will be used for the training and validation of
            the network.
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
        xs_mean : ndarray or None
            Zero center image to be subtracted from data as preprocessing.
        max_queue_size : int
            max_queue_size option of the keras training and evaluation generator
            methods. How many batches get preloaded
            from the generator.
        f_size : int or None
            Specifies the number of samples to be read from the .h5 file.
            If none, the whole .h5 file will be used.
        keras_mode : bool
            If true, yield xs and ys (samples and labels) for the keras fit
            generator function.
            If false, yield the info_blob containing the full sample and label
            info, both before and after the modifiers have been applied.
        shuffle : bool
            Randomize the order in which batches are read from the file.
            Significantly reduces read out speed.

        """
        self.files_dict = files_dict
        self.batchsize = batchsize
        self.key_x_values = key_x_values
        self.key_y_values = key_y_values
        self.sample_modifier = sample_modifier
        self.label_modifier = label_modifier
        self.xs_mean = xs_mean
        self.max_queue_size = max_queue_size
        self.f_size = f_size
        self.keras_mode = keras_mode
        self.shuffle = shuffle

        # a dict with the names of list inputs as keys, and the opened
        # h5 files as values
        self._files = {}
        self._sample_pos = None
        self._i = 0
        self._total_f_size = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        Read another batch of data from the files.

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

        If keras_mode is False, it will return instead:

        info_blob : dict
            Blob containing, the x_values, y_values, xs and ys.

        """
        if self._i == 0:
            self.open_files()
            self._store_file_length()
            self._store_batch_indices()

        try:
            start_index = self._sample_pos[self._i]
        except IndexError:
            self.close_files()
            raise StopIteration

        info_blob = dict()
        info_blob["x_values"] = self.get_x_values(start_index)
        info_blob["y_values"] = self.get_y_values(start_index)

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

        self._i += 1
        if self.keras_mode:
            return xs, ys
        else:
            return info_blob

    def open_files(self):
        """ Open all files. """
        for input_key, file in self.files_dict.items():
            self._files[input_key] = h5py.File(file, 'r')

    def close_files(self):
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
                x_values[input_key] = np.subtract(x_values[input_key],
                                                  self.xs_mean[input_key])
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
            y_values = None
        return y_values

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
            return self.f_size
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
            self.close_files()
            raise ValueError("All data files must have the same length! "
                             "Given were:\n " + str(lengths))

        self._total_f_size = lengths[0]

    def _store_batch_indices(self):
        """
        Define the start indices of each batch in the h5 file and store this.
        """
        total_no_of_batches = int(np.ceil(self._size / self._batchsize))
        sample_pos = np.arange(total_no_of_batches) * self._batchsize

        if self.shuffle:
            np.random.shuffle(sample_pos)
        # append some samples due to preloading by the fit_generator method
        if self.max_queue_size is not None:
            sample_pos = np.append(sample_pos, sample_pos[:self.max_queue_size])

        self._sample_pos = sample_pos


def get_h5_generator(orga, files_dict, f_size=None, zero_center=False,
                     keras_mode=True, shuffle=False):
    """
    Initialize the hdf5_batch_generator_base with the paramters in orga.cfg.

    Parameters
    ----------
    orga : object Organizer
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
    else:
        assert orga._auto_label_modifier is not None, \
            "Auto label modifier has not been set up (can be done with " \
            "nn_utilities.get_auto_label_modifier)"
        label_modifier = orga._auto_label_modifier

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
        xs_mean=xs_mean,
        max_queue_size=orga.cfg.max_queue_size,
        f_size=f_size,
        keras_mode=keras_mode,
        shuffle=shuffle,
    )

    return generator
