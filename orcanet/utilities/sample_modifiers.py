"""
Some basic sample modifiers to use with orcanet.
Use them by setting .cfg.sample_modifier of the orcanet.core.Organizer.

"""
from abc import abstractmethod
import warnings
import numpy as np
from orcanet.utilities.misc import get_register

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

    Parameters
    ----------
    knn : int or None
        Number of nearest neighbors used in the edge conv.
        Pad events with too few hits by duping first hit, and give a warning.
    with_lightspeed : bool
        Multiply time for coordinates input with lightspeed.
    nodes : tuple
        Defines the node features.
    coords : tuple
        Defines the coordinates.
    is_valid : str
        Defines the is_valid.

    """
    def __init__(self, knn=16,
                 with_lightspeed=True,
                 nodes=("pos_x", "pos_y", "pos_z", "time", "dir_x", "dir_y", "dir_z"),
                 coords=("pos_x", "pos_y", "pos_z", "time"),
                 is_valid="is_valid"):
        self.knn = knn
        self.with_lightspeed = with_lightspeed
        self.for_nodes = nodes
        self.for_coords = coords
        self.for_isvalid = is_valid

        # which index in the array from the file contains which data
        # TODO hardcoded
        self.column_names = (
            "pos_x", "pos_y", "pos_z", "time", 'tot',
            'channel_id', "dir_x", "dir_y", "dir_z", "is_valid")
        self.lightspeed = 0.225  # in water; m/ns

    def _str_to_idx(self, which):
        """ Given column name(s), get index of column(s). """
        if isinstance(which, str):
            return self.column_names.index(which)
        else:
            return [self.column_names.index(w) for w in which]

    def __call__(self, info_blob):
        x_value = list(info_blob["x_values"].values())[0]
        nodes = x_value[:, :, self._str_to_idx(self.for_nodes)]
        coords = x_value[:, :, self._str_to_idx(self.for_coords)]
        is_valid = x_value[:, :, self._str_to_idx(self.for_valid)]

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
