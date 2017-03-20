import os
import pickle
import time
from collections import OrderedDict, Counter

import numpy as np

from features.feature_params import UNKNOWN_VALUE


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/223267
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable")
        self._all = []
        OrderedDict.__init__(self, *args, **kwargs)
        self._all = list(self.keys())
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        args = tuple() if self.default_factory is None else self.default_factory,
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory, copy.deepcopy(tuple(self.items())))

    def __repr__(self):
        return "%s(%s, %s)" % (type(self), self.default_factory, OrderedDict.__repr__(self))

    def __setitem__(self, key, value):
        super(DefaultOrderedDict, self).__setitem__(key, value)
        self._all.append(key)

    @property
    def all(self):
        return self._all

    @all.setter
    def all(self, keys):
        self._all = []
        self.clear()
        for i, key in enumerate(keys):
            self[key] = i


class UnknownDict(DefaultOrderedDict):
    """
    DefaultOrderedDict that has a single default value for missing keys
    """
    def __init__(self, d, unknown=None):
        """
        :param d: base dict to initialize by
        :param unknown: value to return for missing keys
        """
        # noinspection PyTypeChecker
        super(UnknownDict, self).__init__(None, d)
        self.unknown = self.setdefault(None, unknown)

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
        ret = self[key] = len(self) if self.max is not None and len(self) < self.max else self.unknown
        return ret


class DropoutDict(AutoIncrementDict):
    """
    UnknownDict that sometimes returns the unknown value even for existing keys
    """
    def __init__(self, d=None, dropout=0, max_size=None, keys=()):
        """
        :param d: base dict to initialize by
        :param dropout: dropout parameter
        :param reverse: whether to keep reverse map (simply list) from numeric id to value
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
        if item != None and self.dropout > 0:
            self.counts[item] += 1
            if self.dropout / (self.counts[item] + self.dropout) > np.random.random_sample():
                item = None
        return super(DropoutDict, self).__getitem__(item)


def save_dict(filename, d):
    """
    Save dictionary to file
    :param filename: file to write to
    :param d: dictionary to save
    """
    print("Saving to '%s'... " % filename, end="", flush=True)
    started = time.time()
    with open(filename, 'wb') as h:
        pickle.dump(d, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done (%.3fs)." % (time.time() - started))


def load_dict(filename):
    """
    Load dictionary from file
    :param filename: file to read from
    """

    def try_load(*names):
        exception = None
        for f in names:
            # noinspection PyBroadException
            try:
                with open(f, 'rb') as h:
                    return pickle.load(h)
            except FileNotFoundError as e:
                exception = e
        if exception is not None:
            raise FileNotFoundError("File not found: '%s'" % "', '".join(names)) from exception

    print("Loading from '%s'... " % filename, end="", flush=True)
    started = time.time()
    d = try_load(filename, os.path.splitext(filename)[0])
    print("Done (%.3fs)." % (time.time() - started))
    return d
