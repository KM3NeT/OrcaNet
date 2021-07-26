"""
Some basic sample modifiers to use with orcanet.
Use them by setting .cfg.sample_modifier of the orcanet.core.Organizer.

"""
from abc import abstractmethod
import warnings
import numpy as np
from orcanet.misc import get_register
import tensorflow as tf

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
    Read out points and coordinates, intended for the MEdgeConv layers.

    For DL files produced with OrcaSong in graph mode.

    Parameters
    ----------
    knn : int or None
        Number of nearest neighbors used in the edge conv.
        Pad events with too few hits by duping first hit, and give a warning.
    node_features : tuple
        Defines the node features.
    coord_features : tuple
        Defines the coordinates.
    ragged : bool, optional
        If True, return ragged tensors (nodes, coordinates).
        If False, return regular tensors, padded to fixed length.
        n_hits_padded and is_valid_features need to be given in this case.
    with_lightspeed : bool
        Multiply time for coordinates input with lightspeed.
        Requires coord_features to have the entry 'time'.
    column_names : tuple, optional
        Name and order of the features in the last dimension of the array.
        If None is given, will attempt to auto-read the column names from
        the attributes of the dataset.
    is_valid_features : str
        Only for when ragged = False.
        Defines the is_valid.
    n_hits_padded : int, optional
        Only for when ragged = False.
        Pad or cut to exactly this many hits using 0s.
        Non-indexed datasets will automatically set this value.

    """
    def __init__(self, knn=16,
                 node_features=("pos_x", "pos_y", "pos_z", "time", "dir_x", "dir_y", "dir_z"),
                 coord_features=("pos_x", "pos_y", "pos_z", "time"),
                 ragged=True,
                 with_lightspeed=True,
                 column_names=None,
                 is_valid_features="is_valid",
                 n_hits_padded=None,
                 ):
        self.knn = knn
        self.node_features = node_features
        self.coord_features = coord_features
        self.ragged = ragged
        self.with_lightspeed = with_lightspeed
        self.column_names = column_names
        self.lightspeed = 0.225  # in water; m/ns
        self.is_valid_features = is_valid_features
        self.n_hits_padded = n_hits_padded

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

    def reset_cache(self):
        """ Clear cached column names. """
        self.column_names = None

    def __call__(self, info_blob):
        # graph has only one file, take it no matter the name
        input_name = list(info_blob["x_values"].keys())[0]
        datasets_meta = info_blob["meta"]["datasets"][input_name]
        is_indexed = datasets_meta.get("samples_is_indexed")
        if self.column_names is None:
            self._cache_column_names(datasets_meta["samples"])

        if is_indexed is True:
            # for indexed sets, x_values is 2d (nodes x features)
            x_values, n_items = info_blob["x_values"][input_name]
            n_hits_padded = None
        else:
            # otherwise it's 3d (batch x max_nodes x features)
            x_values = info_blob["x_values"][input_name]
            is_valid = x_values[:, :, self._str_to_idx(self.is_valid_features)]
            n_hits_padded = is_valid.shape[-1]
            x_values = x_values[is_valid == 1]
            n_items = is_valid.sum(-1)

        x_values = x_values.astype("float32")
        n_items = n_items.astype("int32")

        # pad events with too few hits by duping first hit
        if np.any(n_items < self.knn + 1):
            x_values, n_items = _pad_disjoint(
                x_values, n_items, min_items=self.knn + 1)

        nodes = x_values[:, self._str_to_idx(self.node_features)]
        coords = x_values[:, self._str_to_idx(self.coord_features)]

        if self.with_lightspeed:
            coords[:, self.coord_features.index("time")] *= self.lightspeed

        nodes_t = tf.RaggedTensor.from_row_lengths(nodes, n_items)
        coords_t = tf.RaggedTensor.from_row_lengths(coords, n_items)

        if self.ragged is True:
            return {
                "nodes": nodes_t,
                "coords": coords_t,
            }
        else:
            if self.n_hits_padded is not None:
                n_hits_padded = self.n_hits_padded
            if n_hits_padded is None:
                raise ValueError("Have to give n_hits_padded if ragged is False!")

            sh = [nodes_t.shape[0], n_hits_padded]
            return {
                "nodes": nodes_t.to_tensor(
                    default_value=0., shape=sh+[nodes_t.shape[-1]]),
                "is_valid": tf.ones_like(nodes_t[:, :, 0]).to_tensor(
                    default_value=0., shape=sh),
                "coords": coords_t.to_tensor(
                    default_value=0., shape=sh+[coords_t.shape[-1]]),
            }


def _pad_disjoint(x, n_items, min_items):
    """ Pad disjoint graphs to have a minimum number of hits per event. """
    n_items = np.array(n_items)
    missing = np.clip(min_items - n_items, 0, None)
    for batchno in np.where(missing > 0)[0]:
        warnings.warn(
            f"Event has too few hits! Needed {min_items}, "
            f"had {n_items[batchno]}! Padding...")
        cumu = np.concatenate([[0, ], n_items.cumsum()])
        first_hit = x[cumu[batchno]]
        x = np.insert(
            x,
            cumu[batchno + 1],
            np.repeat(first_hit[None, :], missing[batchno], axis=0),
            axis=0,
        )
        n_items[batchno] = min_items
    return x, n_items

