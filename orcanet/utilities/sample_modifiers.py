"""
Some basic sample modifiers to use with orcanet.
Use them by setting .cfg.sample_modifier of the orcanet.core.Organizer.

"""
from abc import abstractmethod
import warnings
import numpy as np
from orcanet.utilities.misc import get_register

# for loading via toml
smods, register = get_register()


class PerInputModifier:
    """
    For modifiers that do the same operation on each input.
    Apply modify on x_value of each input, and output as dict.

    """
    def __call__(self, info_blob):
        x_values = info_blob["x_values"]
        xs = dict()
        for key, x_value in x_values.items():
            xs[key] = self.modify(x_value)
        return xs

    @abstractmethod
    def modify(self, x_value):
        """ x_value is a batch of input data as a numpy array. """
        raise NotImplementedError


class JoinedModifier(PerInputModifier):
    """
    For applying multiple sample modifiers after each other.

    Example
    -------
    organizer.cfg.sample_modifier = JoinedModifier([
        Reshape((11, 13, 18)), Permute((2, 1, 3))
    ])
    --> Reshape each sample, then permute axes.

    """
    def __init__(self, sample_modifiers):
        self.sample_modifiers = sample_modifiers

    def modify(self, x_value):
        result = x_value
        for smod in self.sample_modifiers:
            result = smod.modify(result)
        return result


@register
class Permute(PerInputModifier):
    """
    Permute the axes of the samples to given order.
    Batchsize axis is excluded, i.e. start indexing with 1!

    Example
    -------
    organizer.cfg.sample_modifier = Permute((2, 1, 3))
    --> Swap first two axes of each sample.

    """
    def __init__(self, axes):
        self.axes = list(axes)

    def modify(self, x_value):
        return np.transpose(x_value, [0] + self.axes)


@register
class Reshape(PerInputModifier):
    """
    Reshape samples to given shape.
    Batchsize axis is excluded!

    Example
    -------
    organizer.cfg.sample_modifier = Reshape((11, 13, 18))
    --> Reshape each sample to that shape.
    
    """
    def __init__(self, newshape):
        self.newshape = list(newshape)

    def modify(self, x_value):
        return np.reshape(x_value, [x_value.shape[0]] + self.newshape)


@register
class GraphEdgeConv:
    """
    Read out points, coordinates and is_valid from the ndarray h5 set.
    Intended for the MEdgeConv layers.

    The array in the h5 file is expected to have shape
    (?, n_points_max, n_features), i.e. the hit features
    (like pos_x, time, is_valid, ...) are in the last dimension.

    Parameters
    ----------
    knn : int or None
        Number of nearest neighbors used in the edge conv.
        Pad events with too few hits by duping first hit, and give a warning.
    node_features : tuple
        Defines the node features.
    coord_features : tuple
        Defines the coordinates.
    is_valid_features : str
        Defines the is_valid.
    with_lightspeed : bool
        Multiply time for coordinates input with lightspeed.
    column_names : tuple, optional
        Name and order of the features in the last dimension of the array.
        If None is given, will attempt to auto-read the column names from
        the attributes of the dataset.

    """
    def __init__(self, knn=16,
                 node_features=("pos_x", "pos_y", "pos_z", "time", "dir_x", "dir_y", "dir_z"),
                 coord_features=("pos_x", "pos_y", "pos_z", "time"),
                 is_valid_features="is_valid",
                 with_lightspeed=True,
                 column_names=None):
        self.knn = knn
        self.with_lightspeed = with_lightspeed
        self.node_features = node_features
        self.coord_features = coord_features
        self.is_valid_features = is_valid_features
        self.column_names = column_names
        self.lightspeed = 0.225  # in water; m/ns

    def _str_to_idx(self, which):
        """ Given column name(s), get index of column(s). """
        if isinstance(which, str):
            return self.column_names.index(which)
        else:
            return [self.column_names.index(w) for w in which]

    def _cache_column_names(self, x_dataset):
        try:
            self.column_names = [x_dataset.attrs[f"hit_info_{i}"]
                                 for i in range(x_dataset.shape[-1])]
        except Exception:
            raise ValueError("Can not read column names from dataset attributes")

    def __call__(self, info_blob):
        # graph has only one file, take it no matter the name
        input_name = list(info_blob["x_values"].keys())[0]

        x_values = info_blob["x_values"][input_name]
        if self.column_names is None:
            self._cache_column_names(info_blob["meta"]["datasets"][input_name]["samples"])

        nodes = x_values[:, :, self._str_to_idx(self.node_features)]
        coords = x_values[:, :, self._str_to_idx(self.coord_features)]
        is_valid = x_values[:, :, self._str_to_idx(self.is_valid_features)]

        if self.with_lightspeed:
            coords[:, :, -1] *= self.lightspeed

        # pad events with too few hits by duping first hit
        if self.knn is not None:
            min_n_hits = self.knn + 1
            n_hits = is_valid.sum(axis=-1)
            too_small = n_hits < min_n_hits
            if any(too_small):
                warnings.warn(f"Event has too few hits! Needed {min_n_hits}, "
                              f"had {n_hits[too_small]}! Padding...")
                for event_no in np.where(too_small)[0]:
                    n_hits_event = int(n_hits[event_no])
                    nodes[event_no, n_hits_event:min_n_hits] = nodes[event_no, 0]
                    coords[event_no, n_hits_event:min_n_hits] = coords[event_no, 0]
                    is_valid[event_no, n_hits_event:min_n_hits] = 1.
        return {
            "nodes": nodes.astype("float32"),
            "is_valid": is_valid.astype("float32"),
            "coords": coords.astype("float32"),
        }
