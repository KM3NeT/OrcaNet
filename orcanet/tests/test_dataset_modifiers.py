from unittest import TestCase
import numpy as np
import orcanet.lib.dataset_modifiers as dmods


class TestFunctions(TestCase):
    def test_as_array(self):
        y_values = "y_values gets simply passed forward"
        y_true = {
            "out_A": 1,
            "out_B": 2,
        }
        y_pred = {
            "out_pred_A": 3,
            "out_pred_B": 4,
        }
        info_blob = {
            "y_values": y_values,
            "ys": y_true,
            "y_pred": y_pred,
        }

        target = {
            "y_values": y_values,
            "label_out_A": 1,
            "label_out_B": 2,
            "pred_out_pred_A": 3,
            "pred_out_pred_B": 4,
        }

        datasets = dmods.as_array(info_blob)
        self.assertDictEqual(datasets, target)


class TestAsRecarrayDist(TestCase):
    def setUp(self):
        inp = {
            "y_pred": {"aa": np.ones((5, 2)), "bb": np.ones((5, 2, 3))*2},
            "ys": {"aa": np.ones((5, 2))*3, "bb": np.ones((5, 2, 3))*4},
        }
        self.output = dmods.as_recarray_dist(inp)

    def test_pred_shape(self):
        self.assertTrue(self.output["pred"].shape == (5, ))

    def test_pred_names(self):
        self.assertTupleEqual(
            self.output["pred"].dtype.names,
            ('aa_1', 'aa_err_1', 'bb_1', 'bb_2', 'bb_3', 'bb_err_1', 'bb_err_2', 'bb_err_3'))

    def test_pred_array_1(self):
        np.testing.assert_array_equal(self.output["pred"]["aa_1"], np.ones(5))

    def test_pred_array_2(self):
        np.testing.assert_array_equal(self.output["pred"]["bb_1"], np.ones(5)*2)

    def test_true_shape(self):
        self.assertTrue(self.output["true"].shape == (5, ))

    def test_true_names(self):
        self.assertTupleEqual(
            self.output["true"].dtype.names, ('aa_1', 'bb_1', 'bb_2', 'bb_3'))

    def test_true_array_1(self):
        np.testing.assert_array_equal(self.output["true"]["aa_1"], np.ones(5)*3)

    def test_true_array_2(self):
        np.testing.assert_array_equal(self.output["true"]["bb_1"], np.ones(5)*4)


class TestAsRecarrayDistSplit(TestAsRecarrayDist):
    def setUp(self):
        inp = {
            "y_pred": {
                "aa": np.ones(5,)*5, "aa_err": np.ones((5, 2)),
                "bb": np.ones((5, 3))*6, "bb_err": np.ones((5, 2, 3))*2,
            },
            "ys": {
                "aa": np.ones(5,)*7, "aa_err": np.ones((5, 2))*3,
                "bb": np.ones((5, 3))*8, "bb_err": np.ones((5, 2, 3))*4,
            },
        }
        self.output = dmods.as_recarray_dist_split(inp)
