"""
Some basic sample modifiers to use with orcanet.
Use them by setting .cfg.sample_modifier of the orcanet.core.Organizer.

"""
from abc import abstractmethod
import numpy as np


class BaseModifier:
    """
    Parent class for modifiers that do the same operation on each input.
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


class Permute(BaseModifier):
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

    @classmethod
    def from_str(cls, string):
        """ E.g. '1,0,2' --> [1, 0, 2] """
        return cls([int(i) for i in string.split(",")])

    def modify(self, x_value):
        return np.transpose(x_value, [0] + self.axes)


class Reshape(BaseModifier):
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

    @classmethod
    def from_str(cls, string):
        """ E.g. '11,13,18' --> [11, 13, 18] """
        return cls([int(i) for i in string.split(",")])

    def modify(self, x_value):
        return np.reshape(x_value, [x_value.shape[0]] + self.newshape)


class JoinedModifier(BaseModifier):
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
