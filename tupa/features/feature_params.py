from collections import OrderedDict

import numpy as np
from ucca.textutil import get_word_vectors

from ..config import Config
from ..labels import Labels
from ..model_util import DropoutDict


class FeatureParameters(Labels):
    def __init__(self, suffix, dim, size, dropout=0, updated=True, num=1, init=None, data=None, indexed=False,
                 copy_from=None, filename=None, min_count=1, enabled=True, node_dropout=0, vocab=None,
                 lang_specific=False):
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
        :param lang_specific: whether the feature params should be separate per language
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
        self.lang_specific = lang_specific

    def __repr__(self):
        return type(self).__name__ + "(" + ", ".join(
            map(str, (self.suffix, self.dim, self.size, self.dropout, self.updated, self.num, self.init, self.data,
                      self.indexed, self.copy_from, self.filename, self.min_count, self.enabled, self.node_dropout,
                      self.vocab, self.lang_specific))) + ")"

    def __eq__(self, other):
        return self.suffix == other.suffix and self.dim == other.dim and self.size == other.size and \
               self.dropout == other.dropout and self.updated == other.updated and self.num == other.num and \
               self.indexed == other.indexed and self.min_count == other.min_count and \
               self.numeric == other.numeric and self.node_dropout == other.node_dropout and \
               self.lang_specific == other.lang_specific

    def __hash__(self):
        return hash(self.suffix)

    def init_data(self):
        if self.data is None and not self.numeric:
            keys = ()
            if self.dim and self.external:
                vectors = self.word_vectors()
                keys = vectors.keys()
                self.init = np.array(list(vectors.values()))
            self.data = DropoutDict(size=self.size, keys=keys, dropout=self.dropout, min_count=self.min_count)

    def word_vectors(self):
        lang = Config().args.lang
        vectors, self.dim = get_word_vectors(self.dim, self.size, self.filename, Config().vocab(self.vocab) or lang)
        if self.size is not None:
            assert len(vectors) <= self.size, "Wrong number of loaded vectors: %d > %d" % (len(vectors), self.size)
        assert vectors, "Cannot load word vectors. Install using `python -m spacy download %s` or choose a file " \
                        "using the --word-vectors option." % lang
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
                                 vocab=getattr(self, "vocab", None),
                                 lang_specific=getattr(self, "lang_specific", False))

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
