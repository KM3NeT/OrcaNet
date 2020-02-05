from unittest import TestCase
import numpy as np
import orcanet.utilities.sample_modifiers as smods


class TestAppli(TestCase):
    def setUp(self):
        self.inp_a_name = "inp_A"
        self.batchsize = 32
        self.info_blob = {
            "x_values": {
                self.inp_a_name: np.ones((self.batchsize, 5, 7, 11))
            }
        }

    def test_reshape(self):
        new_shape = (5*7, 11)
        result = smods.Reshape(new_shape)(self.info_blob)
        self.assertEqual(
            result[self.inp_a_name].shape,
            (self.batchsize, ) + new_shape
        )

    def test_permute(self):
        result = smods.Permute((3, 2, 1))(self.info_blob)
        self.assertEqual(
            result[self.inp_a_name].shape,
            (self.batchsize, ) + (11, 7, 5)
        )

    def test_permute_fromstr(self):
        result = smods.Permute.from_str("3,2,1")(self.info_blob)
        self.assertEqual(
            result[self.inp_a_name].shape,
            (self.batchsize, ) + (11, 7, 5)
        )

    def test_joined_modifier(self):
        new_shape = (5*7, 11)
        modifiers = [
            smods.Reshape(new_shape),
            smods.Permute((2, 1))
        ]

        result = smods.JoinedModifier(modifiers)(self.info_blob)
        self.assertEqual(
            result[self.inp_a_name].shape,
            (self.batchsize, ) + tuple(reversed(new_shape))
        )
