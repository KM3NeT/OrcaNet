""" Odds and ends. """


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

    cls = register[name]
    try:
        return cls(*args, **kwargs)
    except TypeError:
        raise TypeError(f"Error initializing {cls}")
