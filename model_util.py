import shelve
import time

import numpy as np
import os
from collections import defaultdict, Counter

from features.feature_params import UNKNOWN_VALUE


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


class AutoIncrementDict(UnknownDict):
    """
    defaultdict that returns an auto-incrementing index for new keys
    """
    def __init__(self, max_size=None, keys=(), d=None):
        """
        :param max_size: maximum index to allow, after which `unknown' will be returned
        :param keys: initial sequence of keys
        :param d: dictionary to initialize from
        """
        super(AutoIncrementDict, self).__init__({} if d is None else d, unknown=UNKNOWN_VALUE)
        self.max = max_size
        for key in keys:
            self.__missing__(key)

    def __missing__(self, key):
        ret = self[key] = len(self) + UNKNOWN_VALUE if self.max is not None and len(self) + UNKNOWN_VALUE < self.max else self.unknown
        return ret


class DropoutDict(AutoIncrementDict):
    """
    UnknownDict that sometimes returns the unknown value even for existing keys
    """
    def __init__(self, d=None, dropout=0, max_size=None, keys=()):
        """
        :param d: base dict to initialize by
        :param dropout: dropout parameter
        """
        super(DropoutDict, self).__init__(max_size, keys, d=d)
        assert dropout >= 0, "Dropout value must be >= 0, but given %f" % dropout
        if d is not None and isinstance(d, DropoutDict):
            self.dropout = d.dropout
            self.counts = d.counts if self.dropout > 0 else None
        else:
            self.dropout = dropout
            self.counts = Counter() if self.dropout > 0 else None

    def __getitem__(self, item):
        if item != self.UNKNOWN and self.dropout > 0:
            self.counts[item] += 1
            if self.dropout / (self.counts[item] + self.dropout) > np.random.random_sample():
                item = UnknownDict.UNKNOWN
        return super(DropoutDict, self).__getitem__(item)


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
            raise FileNotFoundError("File not found: '%s'" % filename) from exception

    print("Loading from '%s'... " % filename, end="", flush=True)
    started = time.time()
    with try_open(filename, os.path.splitext(filename)[0]) as db:
        d = dict(db)
    print("Done (%.3fs)." % (time.time() - started))
    return d
