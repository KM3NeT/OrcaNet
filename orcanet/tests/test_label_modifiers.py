from unittest import TestCase
import numpy as np
import orcanet.lib.label_modifiers as lmods


class TestRegressionLabels(TestCase):
    @staticmethod
    def _get_ys(columns="obs1", data_factor=None, model_output="log_obs",**kwargs):
        data = np.ones((5,))
        if data_factor is not None:
            data *= data_factor
        info_blob = {"y_values": data.astype(
            np.dtype([("obs1", float), ("obs2", float), ("obs3", float)])
        )}
        lmod = lmods.RegressionLabels(
            columns=columns,
            model_output=model_output,
            **kwargs,
        )
        return lmod(info_blob)

    def test_keys(self):
        ys = self._get_ys(log10=True)
        self.assertTrue("log_obs" in ys)

    def test_content_shape1(self):
        ys = self._get_ys(log10=True)
        np.testing.assert_array_equal(ys["log_obs"], np.zeros((5, 1)))

    def test_content_shape1_no_outputname(self):
        ys = self._get_ys(log10=True, model_output=None)
        np.testing.assert_array_equal(ys["obs1"], np.zeros((5, 1)))

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
            ys["log_obs"], np.array([2, 1, 0, 1, 1]).reshape(5, 1))


class TestRegressionLabelsSplit(TestCase):
    def setUp(self):
        data = np.ones((5,))
        self.info_blob = {"y_values": data.astype(
            np.dtype([("obs1", float), ("obs2", float), ("obs3", float)])
        )}
        self.lmod = lmods.RegressionLabelsSplit(
            columns="obs1",
            model_output="log_obs",
        )

    def test_keys(self):
        ys = self.lmod(self.info_blob)
        self.assertListEqual(list(ys.keys()), ["log_obs", "log_obs_err"])

    def test_content_shape1(self):
        target_shapes = {
            "log_obs": (5, 1),
            "log_obs_err": (5, 2, 1),
        }
        ys = self.lmod(self.info_blob)
        for key, shape in target_shapes.items():
            self.assertTupleEqual(shape, ys[key].shape)

    def test_no_stacks(self):
        lmod = lmods.RegressionLabelsSplit(
            columns="obs1",
            model_output="log_obs",
            stacks=2,
        )
        self.assertTrue(lmod.stacks is None)

    def test_yvalues_is_none(self):
        info_blob = {"y_values": None}
        self.assertIsNone(self.lmod(info_blob))

    def test_yvalues_does_not_have_right_column(self):
        info_blob = {"y_values": np.ones((5,)).astype(
            np.dtype([("asdasd", float), ])
        )}
        self.assertIsNone(self.lmod(info_blob))


class TestClassificationLabels(TestCase):
    @staticmethod
    def _get_ys(
        column="obs1", model_output="cat", classes=None,
    ):
        if classes is None:
            classes = {"class1": [1], "class2": [2]}
        data = np.array([1, 1, 2, 2])
        info_blob = {
            "y_values": data.astype(
                np.dtype([("obs1", float), ("obs2", float), ("obs3", float)])
            )
        }
        lmod = lmods.ClassificationLabels(
            column=column,
            model_output=model_output,
            classes=classes,
        )
        return lmod(info_blob)

    def test_keys(self):
        ys = self._get_ys()
        self.assertTrue("cat" in ys)

    def test_content_shape(self):
        ys = self._get_ys()
        np.testing.assert_array_equal(
            ys["cat"], np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        )

    def test_wrong_classes(self):
        with self.assertRaises(KeyError):
            self._get_ys(classes={"class42": [1], "class2": [2]})


class TestTSClassifier(TestCase):
    @staticmethod
    def _get_ys():
        # take 3 events, one muon CC and one muon NC, and one atm muon
        y_values = np.array(
            [(14, 2), (14, 3), (13, None)],
            dtype=[("particle_type", float), ("is_cc", float)],
        )

        info_blob = {}
        info_blob["y_values"] = y_values

        lmod = lmods.TSClassifier(is_cc_convention=2, model_output="TS_output")
        return lmod(info_blob)

    def test_keys(self):
        ys = self._get_ys()
        self.assertTrue("TS_output" in ys)

    def test_content_shape(self):
        ys = self._get_ys()
        np.testing.assert_array_equal(
            ys["TS_output"], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        )
