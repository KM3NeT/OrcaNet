from unittest import TestCase
import numpy as np
import orcanet.lib.label_modifiers as lmods


class TestRegressionLabels(TestCase):
    @staticmethod
    def _get_ys(columns="obs1", data_factor=None, **kwargs):
        data = np.ones((5,))
        if data_factor is not None:
            data *= data_factor
        info_blob = {"y_values": data.astype(
            np.dtype([("obs1", float), ("obs2", float), ("obs3", float)])
        )}
        lmod = lmods.RegressionLabels(
            columns=columns,
            model_output="log_obs",
            **kwargs,
        )
        return lmod(info_blob)

    def test_keys(self):
        ys = self._get_ys(log10=True)
        self.assertTrue("log_obs" in ys)

    def test_content_shape1(self):
        ys = self._get_ys(log10=True)
        np.testing.assert_array_equal(ys["log_obs"], np.zeros((5, 1)))

    def test_content_stacks_shape1(self):
        ys = self._get_ys(log10=True, stacks=2)
        np.testing.assert_array_equal(ys["log_obs"], np.zeros((5, 2, 1)))

    def test_content(self):
        ys = self._get_ys(columns=["obs1", "obs2", "obs3"])
        np.testing.assert_array_equal(ys["log_obs"], np.ones((5, 3)))

    def test_content_stacks(self):
        ys = self._get_ys(columns=["obs1", "obs2", "obs3"], stacks=2)
        np.testing.assert_array_equal(ys["log_obs"], np.ones((5, 2, 3)))

    def test_log10_invalid_elements_produce_output_zeroes(self):
        ys = self._get_ys(
            columns="obs1",
            data_factor=np.array([100, 10, 1, 0, -1]),
            log10=True,
        )
        np.testing.assert_array_equal(
            ys["log_obs"], np.array([2, 1, 0, 0, 0]).reshape(5, 1))
