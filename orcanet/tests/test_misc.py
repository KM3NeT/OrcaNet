import numpy as np
import orcanet.misc as misc
from unittest import TestCase


class TestFunctions(TestCase):
    def test_dict_to_recarray_shape(self):
        inp = {"aa": np.ones((5, 3)), "bb": np.ones((5, 1))}
        output = misc.dict_to_recarray(inp)
        self.assertTrue(output.shape == (5, ))

    def test_dict_to_recarray_dtype(self):
        inp = {"aa": np.ones((5, 3), dtype="int16"), "bb": np.ones((5, 1), dtype="float32")*4}
        output = misc.dict_to_recarray(inp)
        dtypes = {
            "aa_1": np.dtype("int16"),
            "aa_2": np.dtype("int16"),
            "aa_3": np.dtype("int16"),
            "bb_1": np.dtype("float32"),
        }
        self.assertTupleEqual(output.dtype.names, ('aa_1', 'aa_2', 'aa_3', 'bb_1'))
        for dtype_name, dtype in dtypes.items():
            np.testing.assert_equal(output[dtype_name].dtype, dtype)
        np.testing.assert_equal(output["aa_1"], np.ones(5, dtype="int16"))
        np.testing.assert_equal(output["bb_1"], np.ones(5, dtype="float32")*4)

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


class TestRegister(TestCase):
    def setUp(self):
        self.register = {
            "_func": _func,
            "_Cls": _Cls,
        }

    def _parse_entry(self, toml_entry):
        return misc.from_register(toml_entry, self.register)

    def test_func_str(self):
        self.assertEqual(self._parse_entry("_func")(12), 12)

    def test_func_list(self):
        with self.assertRaises(TypeError):
            self._parse_entry(["_func", 5])

    def test_class_str(self):
        with self.assertRaises(TypeError):
            self._parse_entry("_Cls")

    def test_class_list(self):
        self.assertTupleEqual(self._parse_entry(["_Cls", 0])(12), (0, 12))

    def test_class_dict(self):
        self.assertTupleEqual(
            self._parse_entry({"name": "_Cls", "a": 0})(12), (0, 12))

    def test_class_dict_name_missing(self):
        with self.assertRaises(KeyError):
            self._parse_entry({"names": "_Cls", "a": 0})(12), (0, 12)

    def test_class_list_dict(self):
        self.assertTupleEqual(
            self._parse_entry(["_Cls", {"a": 0}])(12), (0, 12))

    def test_register(self):
        saved, register = misc.get_register()
        register(_func)
        register(_Cls)
        self.assertDictEqual(self.register, saved)


def _func(a):
    return a


class _Cls:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a, x
