""" Odds and ends. """


def get_register():
    """ E.g. for storing orcanet layer blocks as custom objects. """
    saved = {}

    def register(obj):
        saved[obj.__name__] = obj
        return obj
    return saved, register
