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
    names : str or tuple
        Name(s) of the columns that contain the labels.
    model_outputs : str or tuple, optional
        Name(s) of the outputs of the network. Default: Same as names.
    log10 : bool
        Take log10 of the labels.
    stacks : int, optional
        Stack the label this many times along a new axis at -2.
        E.g. if the label is shape (?, 3), it will become
        shape (?, stacks, 3) by repeating it.

    """
    def __init__(self, names,
                 model_outputs=None,
                 log10=False,
                 stacks=None):
        if isinstance(names, str):
            names = (names, )
        if model_outputs is None:
            model_outputs = names
        elif isinstance(model_outputs, str):
            model_outputs = (model_outputs, )
        if len(names) != len(model_outputs):
            raise ValueError(f"names and model_outputs must be of same "
                             f"length ({names}, {model_outputs})")
        self.model_outputs = model_outputs
        self.names = names
        self.stacks = stacks
        self.log10 = log10
        self._warned = False

    def __call__(self, info_blob):
        ys = {}
        for i, name in enumerate(self.names):
            try:
                y_value = info_blob["y_values"][name]
            except KeyError as e:
                if not self._warned:
                    warnings.warn(f"can not generate labels: {e}")
                    self._warned = True
                return None
            ys[self.model_outputs[i]] = self.process_label(y_value)
        return ys

    def process_label(self, y_value):
        y = y_value
        if self.log10:
            y = np.log10(y)
        if self.stacks:
            y = np.repeat(y, repeats=self.stacks, axis=-1)
        return y
