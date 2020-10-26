import numpy as np
import orcanet.misc as misc
from unittest import TestCase


class TestFunctions(TestCase):
    def test_dict_to_recarray(self):
        inp = {"aa": np.ones((5, 3)), "bb": np.ones((5, 1))}
        output = misc.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(output.dtype.names, ('aa_1', 'aa_2', 'aa_3', 'bb_1'))

    def test_dict_to_recarray_len_1(self):
        inp = {"aa": np.ones((5, 3)), "bb": np.ones((5, ))}
        output = misc.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(output.dtype.names, ('aa_1', 'aa_2', 'aa_3', 'bb_1'))

    def test_dict_to_recarray_wrong_dim(self):
        inp = {"aa": np.ones((4, 3)), "bb": np.ones((5, 1))}
        with self.assertRaises(ValueError):
            misc.dict_to_recarray(inp)

    def test_dict_to_recarray_weird_shape(self):
        inp = {"aa": np.ones((5, 3, 2)), "bb": np.ones((5, 1))}
        output = misc.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(
            output.dtype.names,
            ('aa_1', 'aa_2', 'aa_3', 'aa_4', 'aa_5', 'aa_6', 'bb_1'))