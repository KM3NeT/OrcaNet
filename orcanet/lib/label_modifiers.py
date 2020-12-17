import warnings
import numpy as np
import orcanet.misc as misc

# for loading via toml TODO nothing in here yet...
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
        Take log10 of the labels.
    stacks : int, optional
        Stack copies of the label this many times along a new axis at position 1.
        E.g. if the label is shape (?, 3), it will become
        shape (?, stacks, 3). Used for lkl regression.

    Example
    -------
    >>> RegressionLabels(columns=['dir_x', 'dir_y', 'dir_z'], model_output='dir')
    Will produce label 'dir' of shape (bs, 3).

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
        try:
            y_value = info_blob["y_values"][self.columns]
        except KeyError as e:
            if not self._warned:
                warnings.warn(f"Can not generate labels: {e}")
                self._warned = True
            return None
        y_value = misc.to_ndarray(y_value, dtype="float32")
        return {self.model_output: self.process_label(y_value)}

    def process_label(self, y_value):
        ys = y_value
        if self.log10:
            ys = np.log10(ys)
        if self.stacks:
            ys = np.repeat(ys[:, None], repeats=self.stacks, axis=1)
        return ys
