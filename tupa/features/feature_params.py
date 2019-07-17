from collections import OrderedDict
from itertools import islice
from operator import attrgetter

import numpy as np

from ..config import Config
from ..labels import Labels
from ..model_util import DropoutDict


def load_spacy_model(model):
    import spacy
    try:
        return spacy.load(model)
    except OSError:
        spacy.cli.download(model)
        from spacy.cli import link
        from spacy.util import get_package_path
        link(model, model, force=True, model_path=get_package_path(model))
        try:
            return spacy.load(model)
        except OSError as e:
            raise OSError("Failed to get spaCy model. Download it manually using "
                          "`python -m spacy download %s`." % model) from e


def get_word_vectors(dim=None, size=None, filename=None, vocab=None):
    """
    Get word vectors from spaCy model or from text file
    :param dim: dimension to trim vectors to (default: keep original)
    :param size: maximum number of vectors to load (default: all)
    :param filename: text file to load vectors from (default: from spaCy model)
    :param vocab: instead of strings, look up keys of returned dict in vocab (use lang str, e.g. "en", for spaCy vocab)
    :return: tuple of (dict of word [string or integer] -> vector [NumPy array], dimension)
    """
    orig_keys = vocab is None
    if isinstance(vocab, str) or not filename:
        vocab = load_spacy_model(vocab).vocab

    def _lookup(word):
        try:
            return word.orth_ if orig_keys else word.orth
        except AttributeError:
            if orig_keys:
                return word
        lex = vocab[word]
        return getattr(lex, "orth", lex)

    if filename:
        it = read_word_vectors(dim, size, filename)
        nr_row, nr_dim = next(it)
        vectors = OrderedDict(islice(((_lookup(w), v) for w, v in it if orig_keys or w in vocab), nr_row))
    else:  # return spaCy vectors
        nr_row, nr_dim = vocab.vectors.shape
        if dim is not None and dim < nr_dim:
            nr_dim = int(dim)
            vocab.vectors.resize(shape=(int(size or nr_row), nr_dim))
        lexemes = sorted([l for l in vocab if l.has_vector], key=attrgetter("prob"), reverse=True)[:size]
        vectors = OrderedDict((_lookup(l), l.vector) for l in lexemes)
    return vectors, nr_dim


def read_word_vectors(dim, size, filename):
    """
    Read word vectors from text file, with an optional first row indicating size and dimension
    :param dim: dimension to trim vectors to
    :param size: maximum number of vectors to load
    :param filename: text file to load vectors from
    :return: generator: first element is (#vectors, #dims); and all the rest are (word [string], vector [NumPy array])
    """
    try:
        first_line = True
        nr_row = nr_dim = None
        with open(filename, encoding="utf-8") as f:
            for line in f:
                fields = line.split()
                if first_line:
                    first_line = False
                    try:
                        nr_row, nr_dim = map(int, fields)
                        is_header = True
                    except ValueError:
                        nr_dim = len(fields) - 1  # No header, just get vector length from first one
                        is_header = False
                    if dim and dim < nr_dim:
                        nr_dim = dim
                    yield size or nr_row, nr_dim
                    if is_header:
                        continue  # Read next line
                word, *vector = fields
                if len(vector) >= nr_dim:  # May not be equal if word is whitespace
                    yield word, np.asarray(vector[-nr_dim:], dtype="f")
    except OSError as e:
        raise IOError("Failed loading word vectors from '%s'" % filename) from e


class FeatureParameters(Labels):
    def __init__(self, suffix, dim, size, dropout=0, updated=True, num=1, init=None, data=None, indexed=False,
                 copy_from=None, filename=None, min_count=1, enabled=True, node_dropout=0, vocab=None):
        """
        :param suffix: one-character title for feature
        :param dim: vector dimension or, filename to load vectors from, or Word2Vec object
        :param size: maximum number of distinct values
        :param dropout: value of dropout parameter to use during training
        :param updated: whether the feature is learned (otherwise kept constant)
        :param num: how many such features exist per step
        :param init: array of values to use as initial value of embedding matrix
        :param data: DefaultOrderedDict of raw value to running numerical representation, or embedding matrix
        :param indexed: whether the feature is to be used as index into initialized values (otherwise used directly)
        :param copy_from: suffix of other parameter to copy values from instead of extracting them directly
        :param filename: name of file to load data from
        :param min_count: minimum number of occurrences for a feature value before it is actually added
        :param enabled: whether to actually use this parameter in feature extraction
        :param node_dropout: probability to drop whole node in feature extraction
        :param vocab: name of file to load mapping of integer ID to word form (to avoid loading spaCy)
        """
        super().__init__(size)
        self.suffix = suffix
        self.dim = dim
        self.size = size
        self.dropout = dropout
        self.updated = updated
        self.num = num
        self.init = init
        self.data = data
        self.indexed = indexed
        self.copy_from = copy_from
        self.filename = filename
        self.min_count = min_count
        self.enabled = enabled
        self.node_dropout = node_dropout
        self.vocab = vocab

    def __repr__(self):
        return type(self).__name__ + "(" + ", ".join(
            map(str, (self.suffix, self.dim, self.size, self.dropout, self.updated, self.num, self.init, self.data,
                      self.indexed, self.copy_from, self.filename, self.min_count, self.enabled, self.node_dropout,
                      self.vocab))) + ")"

    def __eq__(self, other):
        return self.suffix == other.suffix and self.dim == other.dim and self.size == other.size and \
               self.dropout == other.dropout and self.updated == other.updated and self.num == other.num and \
               self.indexed == other.indexed and self.min_count == other.min_count and \
               self.numeric == other.numeric and self.node_dropout == other.node_dropout

    def __hash__(self):
        return hash(self.suffix)

    def init_data(self):
        if self.data is None and not self.numeric:
            keys = ()
            if self.dim and self.external:
                vectors = self.word_vectors()
                if vectors:
                    keys = vectors.keys()
                    self.init = np.array(list(vectors.values()))
            self.data = DropoutDict(size=self.size, keys=keys, dropout=self.dropout, min_count=self.min_count)

    def word_vectors(self):
        vectors, self.dim = get_word_vectors(self.dim, self.size, self.filename, Config().vocab(self.vocab) or "en")
        if self.size is not None:
            assert len(vectors) <= self.size, "Wrong number of loaded vectors: %d > %d" % (len(vectors), self.size)
        self.size = len(vectors)
        return vectors

    @property
    def all(self):
        return self.data.all

    @all.setter
    def all(self, labels):
        self.data.all = labels

    @property
    def numeric(self):
        return False

    @property
    def prop(self):
        return self.copy_from or self.suffix

    @property
    def external(self):
        return self.copy_from is not None

    @staticmethod
    def copy(params, copy_dict=dict, copy_init=True, order=None):
        return OrderedDict((key, param.copy_with_data(copy_dict, copy_init)) for key, param in
                           (params.items() if order is None else sorted(params.items(), key=lambda x:
                            -1 if x[0] == NumericFeatureParameters.SUFFIX else order.index(x[0][0]))))

    def copy_with_data(self, copy_dict, copy_init):
        data = None if self.data is None else copy_dict(self.data)
        if hasattr(data, "size"):  # It may be an UnknownDict but we still want it to know its size
            data.size = self.size
        return FeatureParameters(suffix=self.suffix, dim=self.dim, size=self.size, dropout=self.dropout,
                                 updated=self.updated, num=self.num, init=self.init if copy_init else None, data=data,
                                 indexed=self.indexed, copy_from=self.copy_from, filename=self.filename,
                                 min_count=self.min_count, enabled=self.enabled,
                                 node_dropout=getattr(self, "node_dropout", 0),
                                 vocab=getattr(self, "vocab", None))

    def unfinalize(self):
        self.data = DropoutDict(self.data, size=self.size, dropout=self.dropout, min_count=self.min_count)


class NumericFeatureParameters(FeatureParameters):
    SUFFIX = "numeric"

    def __init__(self, num, node_dropout=0):
        super().__init__(NumericFeatureParameters.SUFFIX, 1, None, num=num, node_dropout=node_dropout)

    def __repr__(self):
        return "%s(%d)" % (type(self).__name__, self.num)

    @property
    def numeric(self):
        return True

    def copy_with_data(self, copy_dict, copy_init):
        return NumericFeatureParameters(self.num, node_dropout=getattr(self, "node_dropout", 0))
