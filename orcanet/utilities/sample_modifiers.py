"""
Some basic sample modifiers to use with orcanet.
Use them by setting .cfg.sample_modifier of the orcanet.core.Organizer.

"""
import numpy as np


class Permute:
    """
    Permute the axes of the samples to given order.
    Batchsize axis is excluded, i.e. start indexing with 1!

    Example
    -------
    organizer.cfg.sample_modifier = Permute((2, 1, 3))
    --> Swap first two axes of each sample.

    """
    def __init__(self, axes):
        self.axes = axes

    @classmethod
    def from_str(cls, string):
        """ E.g. '1,0,2' --> [1, 0, 2] """
        return cls([int(i) for i in string.split(",")])

    def __call__(self, info_blob):
        x_values = info_blob["x_values"]
        xs = dict()
        for key, x in x_values.items():
            xs[key] = np.transpose(
                x, [0] + [i for i in self.axes]
            )
        return xs


class Reshape:
    """
    Reshape samples to given shape.
    Batchsize axis is excluded!

    Example
    -------
    organizer.cfg.sample_modifier = Reshape((11, 13, 18))
    --> Reshape each sample to that shape.
    
    """
    def __init__(self, newshape):
        self.newshape = newshape

    @classmethod
    def from_str(cls, string):
        """ E.g. '11,13,18' --> [11, 13, 18] """
        return cls([int(i) for i in string.split(",")])

    def __call__(self, info_blob):
        x_values = info_blob["x_values"]
        xs = {}
        for key, x in x_values.items():
            xs[key] = np.reshape(
                x, [x.shape[0]] + list(self.newshape)
            )
        return xs
