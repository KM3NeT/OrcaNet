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
        Name(s) of the columns that contain the labels.
    model_output : str, optional
        Name of the output of the network.
        Default: Same as names (only valid if names is a str).
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
    Will produce array of shape (bs, 3) for model output 'dir'.
    >>> RegressionLabels(columns='dir_x')
    Will produce array of shape (bs, 1) for model output 'dir_x'.

    """
    def __init__(self, columns,
                 model_output=None,
                 log10=False,
                 stacks=None):
        if isinstance(columns, str):
            columns = [columns, ]
        else:
            columns = list(columns)
        if model_output is None:
            if len(columns) != 1:
                raise ValueError(f"If model_output is not given, names must be length 1!")
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
                warnings.warn(
                    f"Can not generate labels: No y_values available!")
                self._warned = True
            return None
        try:
            y_value = y_values[self.columns]
        except KeyError:
            if not self._warned:
                warnings.warn(
                    f"Can not generate labels: {self.columns} "
                    f"not found in y_values")
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
                    "invalid value encountered in log10, setting result to 0",
                    category=RuntimeWarning,
                )
            ys = np.log10(ys, where=gr_zero, out=np.zeros_like(ys, dtype="float32"))
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
                "Can not use stacks option with RegressionLabelsSplit, ignoring...")
            self.stacks = None

    def __call__(self, info_blob):
        output_dict = super().__call__(info_blob)
        if output_dict is None:
            return None
        err_outputs = {}
        for name, label in output_dict.items():
            err_outputs[self.err_output_format.format(name)] = np.repeat(
                np.expand_dims(label, axis=-2), repeats=2, axis=-2)
        output_dict.update(err_outputs)
        return output_dict
