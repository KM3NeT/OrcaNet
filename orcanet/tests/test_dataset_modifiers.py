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

    def test_as_recarray_dist(self):
        inp = {
            "y_pred": {"aa": np.ones((5, 2)), "bb": np.ones((5, 2, 3))},
            "ys": {"aa": np.ones((5, 2)), "bb": np.ones((5, 2, 3))},
        }
        output = dmods.as_recarray_dist(inp)
        self.assertTrue(output["pred"].shape == (5, ))
        self.assertTupleEqual(
            output["pred"].dtype.names,
            ('aa_1', 'aa_err_1', 'bb_1', 'bb_2', 'bb_3', 'bb_err_1', 'bb_err_2', 'bb_err_3'))

        self.assertTrue(output["true"].shape == (5, ))
        self.assertTupleEqual(
            output["true"].dtype.names,
            ('aa_1', 'bb_1', 'bb_2', 'bb_3'))
