from unittest import TestCase
import numpy as np
import orcanet.lib.sample_modifiers as smods


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


class TestGraphSampleMod(TestCase):
    def setUp(self) -> None:
        self.smod = smods.GraphEdgeConv(
            column_names=("c1", "time"),
            node_features=("c1", ),
            coord_features=("c1", "time"),
            knn=10,
        )
        self.info_blob = {
            "x_values": {"file_1":
                (
                    # hits
                    np.arange(40, dtype="float32").reshape(20, 2),
                    # n_items
                    np.array([5, 15], dtype="float32"),
                )
            },
            "meta": {"datasets": {"file_1": {
                "samples_is_indexed": True,
            }}}
        }

    def test_nodes_shape(self):
        out = self.smod(self.info_blob)
        self.assertEqual(
            list(out["nodes"].shape),
            [2, None, 1],
        )

    def test_nodes_n_items(self):
        out = self.smod(self.info_blob)
        np.testing.assert_array_equal(
            out["nodes"].row_lengths().numpy(),
            np.array([11, 15], dtype="int32"),
        )

    def test_nodes_content(self):
        out = self.smod(self.info_blob)
        np.testing.assert_array_equal(
            out["nodes"].merge_dims(0, 1).numpy(),
            np.concatenate(
                [np.arange(0, 10, 2), np.zeros(6), np.arange(10, 40, 2)]
            ).astype("float32")[:, None],
        )
