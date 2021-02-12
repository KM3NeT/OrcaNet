""" Odds and ends. """
import os
import inspect
import numpy as np


def get_register():
    """ E.g. for storing orcanet layer blocks as custom objects. """
    saved = {}

    def register(obj):
        saved[obj.__name__] = obj
        return obj
    return saved, register


def from_register(toml_entry, register):
    """
    Get an initilized object via a toml entry.
    Used for loading orcanet built-in sample modifiers etc.

    Parameters
    ----------
    toml_entry : str or dict or list
        The 'sample_modifier' given in the config toml.
        E.g., to initialize "obj_name" from register, these are possible formats:
        "obj_name"
        ["obj_name", True]
        ["obj_name", {"setting_1": True}]
        {"name": "obj_name", "setting_1": True}
    register : dict
        Maps class names to class references.

    """
    args, kwargs = [], {}
    if isinstance(toml_entry, str):
        name = toml_entry
    elif isinstance(toml_entry, dict):
        if "name" not in toml_entry:
            raise KeyError(f"missing entry in dict: 'name', given: {toml_entry}")
        name = toml_entry["name"]
        kwargs = {k: v for k, v in toml_entry.items() if k != "name"}
    else:
        name = toml_entry[0]
        if len(toml_entry) == 2 and isinstance(toml_entry[1], dict):
            kwargs = toml_entry[1]
        else:
            args = toml_entry[1:]

    obj = register[name]
    try:
        if inspect.isfunction(obj):
            if args or kwargs:
                raise TypeError(f"Can not pass arguments to function ({args}, {kwargs})")
            return obj
        else:
            return obj(*args, **kwargs)
    except TypeError:
        raise TypeError(f"Error initializing {obj}")


def dict_to_recarray(array_dict):
    """
    Convert a dict with np arrays to a 2d recarray.
    Column names are derived from the dict keys.

    Parameters
    ----------
    array_dict : dict
        Keys: string
        Values: ND arrays, same length and number of dimensions.
            All dimensions expect first will get flattened.

    Returns
    -------
    The recarray.

    """
    column_names, arrays = [], []
    for key, array in array_dict.items():
        if len(array.shape) == 1:
            array = np.expand_dims(array, -1)
        elif len(array.shape) > 2:
            array = np.reshape(array, (len(array), -1))
        for i in range(array.shape[-1]):
            arrays.append(array[:, i])
            column_names.append(f"{key}_{i+1}")
    return np.core.records.fromarrays(arrays, names=column_names)


def to_ndarray(x, dtype="float32"):
    """ Turn recarray to ndarray. """
    new_dtype = [(name, dtype) for name in x.dtype.names]
    new_shape = (len(x), len(x.dtype.names))
    return np.ascontiguousarray(x).astype(new_dtype).view(dtype).reshape(new_shape)


def find_file(directory, filename):
    """ Look for file in given directoy. Error if there are multiple. """
    found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                found.append(os.path.join(root, file))
    if len(found) >= 2:
        raise ValueError(
            f"Can not find {filename}: More than one file found ({found})")
    elif len(found) == 0:
        return None
    else:
        fpath = found[0]
        print(f"Found {fpath}")
        return fpath
