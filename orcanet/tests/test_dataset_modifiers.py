from unittest import TestCase
import numpy as np
import orcanet.utilities.dataset_modifiers as dmods


class TestFunctions(TestCase):
    def test_dict_to_recarray(self):
        inp = {"aa": np.ones((5, 3)), "bb": np.ones((5, 1))}
        output = dmods.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(output.dtype.names, ('aa_1', 'aa_2', 'aa_3', 'bb_1'))

    def test_dict_to_recarray_len_1(self):
        inp = {"aa": np.ones((5, 3)), "bb": np.ones((5, ))}
        output = dmods.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(output.dtype.names, ('aa_1', 'aa_2', 'aa_3', 'bb_1'))

    def test_dict_to_recarray_wrong_dim(self):
        inp = {"aa": np.ones((4, 3)), "bb": np.ones((5, 1))}
        with self.assertRaises(ValueError):
            dmods.dict_to_recarray(inp)

    def test_dict_to_recarray_weird_shape(self):
        inp = {"aa": np.ones((5, 3, 2)), "bb": np.ones((5, 1))}
        output = dmods.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))
        self.assertTupleEqual(
            output.dtype.names,
            ('aa_1', 'aa_2', 'aa_3', 'aa_4', 'aa_5', 'aa_6', 'bb_1'))

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

    def test_as_recarray_distr(self):
        inp = {
            "y_pred": {"aa": np.ones((5, 2)), "bb": np.ones((5, 2, 3))},
            "ys": {"aa": np.ones((5, 2)), "bb": np.ones((5, 2, 3))},
        }
        output = dmods.as_recarray_distr(inp)
        self.assertTrue(output["pred"].shape == (5, ))
        self.assertTupleEqual(
            output["pred"].dtype.names,
            ('aa_1', 'aa_err_1', 'bb_1', 'bb_2', 'bb_3', 'bb_err_1', 'bb_err_2', 'bb_err_3'))

        self.assertTrue(output["true"].shape == (5, ))
        self.assertTupleEqual(
            output["true"].dtype.names,
            ('aa_1', 'bb_1', 'bb_2', 'bb_3'))
