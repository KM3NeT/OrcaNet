import warnings
import numpy as np
import orcanet.misc as misc

# for loading via toml
lmods, register = misc.get_register()


class ColumnLabels:
    """
    Label of each model output is column with the same name in the h5 file.
    This is the default label modifier.

    Example
    -------
    Model has output "energy" --> label is column "energy" from the label
    dataset in the h5 file.

    Parameters
    ----------
    model : ks.Model
            A keras model.

    """

    def __init__(self, model):
        self.output_names = model.output_names

    def __call__(self, info_blob):
        ys = {name: info_blob["y_values"][name] for name in self.output_names}
        return ys


@register
class RegressionLabels:
    """
    Generate labels for regression.

    Parameters
    ----------
    columns : str or list
            Name(s) of the columns in the label dataset that contain the labels.
    model_output : str, optional
            Name of the output of the network.
            Default: Same as columns (only valid if columns is a str).
    log10 : bool
            Take log10 of the labels. Invalid values in the label will produce 0
            and a warning.
    stacks : int, optional
            Stack copies of the label this many times along a new axis at position 1.
            E.g. if the label is shape (?, 3), it will become
            shape (?, stacks, 3). Used for lkl regression.

    Examples
    --------
    >>> RegressionLabels(columns=['dir_x', 'dir_y', 'dir_z'], model_output='dir')
    or in the config.toml:
    label_modifier = {name='RegressionLabels', columns=['dir_x','dir_y','dir_z'], model_output='dir'}
    Will produce array of shape (bs, 3) for model output 'dir'.
    >>> RegressionLabels(columns='dir_x')
    Will produce array of shape (bs, 1) for model output 'dir_x'.

    """

    def __init__(self, columns, model_output=None, log10=False, stacks=None):
        if isinstance(columns, str):
            columns = [
                columns,
            ]
        else:
            columns = list(columns)
        if model_output is None:
            if len(columns) != 1:
                raise ValueError(
                    f"If model_output is not given, columns must be length 1!"
                )
            model_output = columns[0]

        self.columns = columns
        self.model_output = model_output
        self.stacks = stacks
        self.log10 = log10
        self._warned = False

    def __call__(self, info_blob):
        y_values = info_blob["y_values"]
        if y_values is None:
            if not self._warned:
                warnings.warn(f"Can not generate labels: No y_values available!")
                self._warned = True
            return None
        try:
            y_value = y_values[self.columns]
        except KeyError:
            if not self._warned:
                warnings.warn(
                    f"Can not generate labels: {self.columns} " f"not found in y_values"
                )
                self._warned = True
            return None
        y_value = misc.to_ndarray(y_value, dtype="float32")
        return {self.model_output: self.process_label(y_value)}

    def process_label(self, y_value):
        ys = y_value
        if self.log10:
            gr_zero = ys > 0
            if not np.all(gr_zero):
                warnings.warn(
                    "invalid value encountered in log10, setting result to 1",
                    category=RuntimeWarning,
                )
            ys = np.log10(ys, where=gr_zero, out=np.ones_like(ys, dtype="float32"))
        if self.stacks:
            ys = np.repeat(ys[:, None], repeats=self.stacks, axis=1)

        return ys


@register
class RegressionLabelsSplit(RegressionLabels):
    """
    Generate labels for regression.

    Intended for networks that output recos and errs in seperate towers
    (for example when using OutputRegNormalSplit as output layer block).

    Example
    -------
    >>> RegressionLabelsSplit(columns=['dir_x', 'dir_y', 'dir_z'], model_output='dir')
    Will produce label 'dir' of shape (bs, 3),
    and label 'dir_err' of shape (bs, 2, 3).

    'dir_err' is just the label twice, along a new axis at -2.
    Necessary because pred and truth must be the same shape.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.err_output_format = "{}_err"
        if self.stacks is not None:
            warnings.warn(
                "Can not use stacks option with RegressionLabelsSplit, ignoring..."
            )
            self.stacks = None
        self._warned = False

    def __call__(self, info_blob):
        output_dict = super().__call__(info_blob)
        if output_dict is None:
            return None
        err_outputs = {}
        for name, label in output_dict.items():
            err_outputs[self.err_output_format.format(name)] = np.repeat(
                np.expand_dims(label, axis=-2), repeats=2, axis=-2
            )
        output_dict.update(err_outputs)
        return output_dict


@register
class ClassificationLabels:
    """
    One-hot encoding for general purpose classification labels based on one mc label column.

    Parameters
    ----------
    column : str
            Identifier of which mc info to create the labels from.
    classes : dict
            Specify for each class the conditions the column name has to fulfil.
            The keys have to be named "class1", "class2", etc
    model_output : str, optional
            The name of the output layer's outputs.

    Example
    -------
    2-class cf for signal and background; put this into the config.toml:
    label_modifier = {name="ClassificationLabels", column="particle_type", classes={class1 = [12, -12, 14, -14], class2 = [13, -13, 0]}, model_output="bg_output"}

    """

    def __init__(
        self,
        column,
        classes,
        model_output=None,
    ):
        self.column = column
        self.classes = classes
        self.model_output = model_output
        self._warned = False

        if "class1" not in self.classes:
            raise KeyError("Class names must be named 'class1', 'class2',...")
        if not len(self.classes["class1"]) > 0:
            raise ValueError("Not a valid list for a class")

        if model_output is None:
            self.model_output = column

    def __call__(self, info_blob):

        y_values = info_blob["y_values"]

        if y_values is None:
            if not self._warned:
                warnings.warn(f"Can not generate labels: No y_values available!")
                self._warned = True
            return None

        try:
            y_value = y_values[self.column]
        except ValueError:
            if not self._warned:
                warnings.warn(
                    f"Can not generate labels: {self.column} " f"not found in y_values"
                )
                self._warned = True
            #let this pass by for real data
            return None

        # create an array of the final shape, initialized with zeros
        n_classes = len(self.classes)
        batchsize = y_values.shape[0]
        categories = np.zeros((batchsize, n_classes), dtype="bool")

        # iterate over every class and set entries to 1 if condition is fulfilled
        for i in range(n_classes):
            categories[:, i] = np.in1d(
                y_values[self.column], self.classes["class" + str(i + 1)]
            )

        return {self.model_output: categories.astype(np.float32)}


@register
class TSClassifier:

    """
    One-hot encoding for track/shower classifier. Muon neutrino CC are tracks, the rest
    of neutrinos is shower. This means, this has to be extended for tau neutrinos. Atm.
    muon events, if any, are tracks.

    Parameters
    ----------
    is_cc_convention : int
            The convention used in the MC prod to indicate a charged current interaction.
            For post 2020 productions this is 2.
    model_output : str, optional
            Name of the output of the network.
            Default: Same as names (only valid if names is a str).

    Example
    -------
    label_modifier = {name='TSClassifier', is_cc_convention=2}

    """

    def __init__(
        self,
        is_cc_convention,
        model_output="ts_output",
    ):
        self.is_cc_convention = is_cc_convention
        self.model_output = model_output

    def __call__(self, info_blob):

        y_values = info_blob["y_values"]

        try:
            particle_type = y_values["particle_type"]
            is_cc = y_values["is_cc"] == self.is_cc_convention
        except ValueError:
            if not self._warned:
                warnings.warn(
                    f"Can not generate labels: particle_type or is_cc not found in y_values"
                )
                self._warned = True
            #let this pass by for real data
            return None

        ys = dict()

        # create conditions from particle_type and is cc
        is_muon_cc = np.logical_and(np.abs(particle_type) == 14, is_cc)

        # in case there are atm. muon events in the mix as well, declare them to be tracks
        is_track = np.logical_or(is_muon_cc, np.abs(particle_type) == 13)

        is_shower = np.invert(is_track)

        batchsize = y_values.shape[0]
        # categorical [shower, track] -> [1,0] = shower, [0,1] = track
        categorical_ts = np.zeros((batchsize, 2), dtype="bool")

        categorical_ts[:, 0] = is_track
        categorical_ts[:, 1] = is_shower

        ys[self.model_output] = categorical_ts.astype(np.float32)

        return ys
