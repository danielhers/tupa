import os
import shelve
import time
from collections import defaultdict


class UnknownDict(defaultdict):
    """
    defaultdict that has a single default value for missing keys
    """
    UNKNOWN = "<UNKNOWN>"

    def __init__(self, d, unknown=None):
        """
        :param d: base dict to initialize by
        :param unknown: value to return for missing keys
        """
        super(UnknownDict, self).__init__(None, d)
        if self.UNKNOWN not in self:
            assert unknown is not None, "Default value must not be None"
            self[self.UNKNOWN] = unknown
        self.unknown = self[self.UNKNOWN]

    def __missing__(self, key):
        return self.unknown


class KeyDefaultDict(defaultdict):
    """
    defaultdict that allows the default_factory to use the key
    """
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


class AutoIncrementDict(UnknownDict):
    """
    defaultdict that returns an auto-incrementing index for new keys
    """
    def __init__(self, max_size=None, keys=()):
        """
        :param max_size: maximum index to allow, after which 0 will be returned
        :param keys: initial sequence of keys
        """
        super(AutoIncrementDict, self).__init__({}, unknown=0)
        self.max = max_size
        for key in keys:
            self.__missing__(key)

    def __missing__(self, key):
        ret = self[key] = len(self) if self.max is not None and len(self) < self.max else self.unknown
        return ret


def save_dict(filename, d):
    """
    Save dictionary to file
    :param filename: file to write to; the actual written file may have an additional suffix
    :param d: dictionary to save
    """
    print("Saving to '%s'... " % filename, end="", flush=True)
    started = time.time()
    with shelve.open(filename) as db:
        db.update(d)
    print("Done (%.3fs)." % (time.time() - started))


def load_dict(filename):
    """
    Load dictionary from file
    :param filename: file to read from; the actual read file may have an additional suffix
    """

    def try_open(*names):
        exception = None
        for f in names:
            # noinspection PyBroadException
            try:
                return shelve.open(f, flag="r")
            except Exception as e:
                exception = e
        if exception is not None:
            raise IOError("File not found: " + filename) from exception

    print("Loading from '%s'... " % filename, end="", flush=True)
    started = time.time()
    with try_open(filename, os.path.splitext(filename)[0]) as db:
        d = dict(db)
    print("Done (%.3fs)." % (time.time() - started))
    return d
