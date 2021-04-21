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
                raise ValueError(f"If model_output is not given, columns must be length 1!")
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


@register
def ts_classifier(data):
	"""
	One-hot encoding for track/shower classifier. Muon neutrino CC are tracks, the rest
	shower. Set should not contain atm muons or tau neutrino events. Otherwise this needs 
	to be expanded. 
	"""
	y_values = data["y_values"]

	ys = dict()
	particle_type = y_values['particle_type']
	is_cc = y_values['is_cc'] == 2
	is_muon_cc = np.logical_and(np.abs(particle_type) == 14, is_cc)
	is_not_muon_cc = np.invert(is_muon_cc)

	batchsize = y_values.shape[0]
	# categorical [shower, track] -> [1,0] = shower, [0,1] = track
	categorical_ts = np.zeros((batchsize, 2), dtype='bool')

	categorical_ts[:, 0] = is_not_muon_cc
	categorical_ts[:, 1] = is_muon_cc

	ys['ts_output'] = categorical_ts.astype(np.float32)
	return ys

@register
def bg_classifier(data):
	"""
	One-hot encoding for background classification. Neutrino events are signal, everthing
	else is background.
	"""
	
	y_values = data["y_values"]
	
	# for every sample, [1,0,0] for neutrinos, [0,1,0] for mupage
	# particle types: mupage: np.abs(13), random_noise = 0
	ys = dict()
	particle_type = y_values['particle_type']
	is_mupage = np.abs(particle_type) == 13
	is_random_noise = np.abs(particle_type == 0)
	is_not_mupage_nor_rn = np.invert(np.logical_or(is_mupage,
												   is_random_noise))

	batchsize = y_values.shape[0]
	categorical_bg = np.zeros((batchsize, 2), dtype='bool')

	# neutrino
	categorical_bg[:, 0] = is_not_mupage_nor_rn
	# is not neutrino
	categorical_bg[:, 1] = np.invert(is_not_mupage_nor_rn)

	ys['bg_output'] = categorical_bg.astype(np.float32)
	return ys
  
