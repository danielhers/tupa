import sys
import time
from collections import OrderedDict, Counter, defaultdict

import csv
import json
import numpy as np
import os
import pickle
import pprint as pp
import shutil
from glob import glob
from operator import itemgetter
from tqdm import tqdm

from .labels import Labels

MISSING_VALUE = -1
UNKNOWN_VALUE = 0


class DefaultOrderedDict(OrderedDict, Labels):
    # Source: http://stackoverflow.com/a/6190500/223267
    def __init__(self, default_factory=None, *args, size=None, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable")
        Labels.__init__(self, size)
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

    def __setitem__(self, key, value, **kwargs):
        super().__setitem__(key, value, **kwargs)
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


class AutoIncrementDict(DefaultOrderedDict):
    """
    DefaultOrderedDict that returns an auto-incrementing index for new keys
    """
    def __init__(self, size=None, keys=(), d=None, unknown=UNKNOWN_VALUE):
        """
        :param size: maximum size to allow, after which `unknown' will be returned
        :param keys: initial sequence of keys
        :param d: dictionary to initialize from
        :param unknown: value to return for missing keys
        """
        super().__init__(None, sorted(d.items(), key=itemgetter(1)) if d else {}, size=size)
        self.finalized = (size is None)
        self.unknown = self.setdefault(None, unknown)
        for key in keys:
            self.__missing__(key)

    def __missing__(self, key):
        if not self.finalized and len(self) < self.size:
            ret = self[key] = len(self)
            return ret
        return self.unknown

    def first_items(self, n=3):
        return ", ".join(map(str, sorted(self, key=self.get)[:n])) + (", ..." if len(self) > n else "")

    def __str__(self):
        return "{%s}" % self.first_items()

    def __repr__(self):
        return "%s(size=%s, keys=[%s], unknown=%s)" % (type(self).__name__, self.size, self.first_items(), self.unknown)


class UnknownDict(AutoIncrementDict):
    """
    DefaultOrderedDict that has a single default value for missing keys
    """
    def __init__(self, d=None):
        """
        :param d: base dict to initialize by
        """
        super().__init__(size=None, d=d)


class DropoutDict(AutoIncrementDict):
    """
    UnknownDict that sometimes returns the unknown value even for existing keys
    """
    def __init__(self, d=None, dropout=0, size=None, keys=(), min_count=1):
        """
        :param d: base dict to initialize by
        :param dropout: dropout parameter
        :param min_count: minimum number of occurrences for a key before it is actually added to the dict
        """
        super().__init__(size, keys, d=d)
        assert dropout >= 0, "Dropout value must be >= 0, but given %f" % dropout
        self.dropout, self.counts, self.min_count = (d.dropout, d.counts, d.min_count) \
            if d is not None and isinstance(d, DropoutDict) else (dropout, Counter(), min_count)

    def __getitem__(self, item):
        if item is not None:
            self.counts[item] += 1
            count = self.counts[item]
            if count < self.min_count or self.dropout and self.dropout/(count+self.dropout) > np.random.random_sample():
                item = None
        return super().__getitem__(item)


class KeyBasedDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = self.default_factory(key)
        return self[key]


def remove_existing(*filenames):
    for filename in filenames:
        try:
            shutil.copy2(filename, filename + "~")
            os.remove(filename)
            print("Removed existing '%s'." % filename)
        except OSError:
            pass


def remove_backup(*filenames):
    for filename in filenames:
        for backup in glob(filename + "*~"):
            try:
                os.remove(backup)
            except OSError:
                pass


def save_dict(filename, d):
    """
    Save dictionary to Pickle file
    :param filename: file to write to
    :param d: dictionary to save
    """
    remove_existing(filename)
    sys.setrecursionlimit(2000)
    print("Saving to '%s'... " % filename, end="", flush=True)
    started = time.time()
    with open(filename, "wb") as h:
        try:
            pickle.dump(d, h, protocol=pickle.HIGHEST_PROTOCOL)
        except RecursionError as e:
            raise IOError("Failed dumping dictionary:\n" + pp.pformat(d, compact=True)) from e
    print("Done (%.3fs)." % (time.time() - started))


def load_dict(filename):
    """
    Load dictionary from Pickle file
    :param filename: file to read from
    """

    def try_load(*names):
        exception = None
        for f in names:
            # noinspection PyBroadException
            try:
                with open(f, "rb") as h:
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


def jsonify(o):
    try:
        return o.__dict__
    except AttributeError:
        return o.tolist()


def save_json(filename, d):
    """
    Save dictionary to JSON file
    :param filename: file to write to
    :param d: dictionary to save
    """
    remove_existing(filename)
    print("Saving to '%s'." % filename)
    with open(filename, "w") as h:
        json.dump(d, h, default=jsonify)


def load_json(filename):
    """
    Load dictionary from JSON file
    :param filename: file to read from
    """
    print("Loading from '%s'." % filename)
    with open(filename, "r") as h:
        d = json.load(h)
    return d


class Lexeme:
    def __init__(self, index, text):
        self.index = self.orth = index
        self.text = self.orth_ = text


class Strings:
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, item):
        lex = self.vocab[item]
        return lex.index if isinstance(item, str) else lex.text


class Vocab(dict):
    def __init__(self, tuples):
        super().__init__()
        for k, v in tuples:
            self[int(k)] = self[v] = Lexeme(int(k), v)
        self.strings = Strings(self)


class IdentityVocab(Vocab):
    def __init__(self):
        super().__init__(())

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return Lexeme(item, item)


def load_enum(filename):
    if filename == "-":
        return IdentityVocab()
    else:
        with open(filename, encoding="utf-8") as f:
            return Vocab(tqdm(csv.reader(f), desc="Loading '%s'" % filename, file=sys.stdout, unit=" rows"))
