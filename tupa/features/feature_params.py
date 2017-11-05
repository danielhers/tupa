from copy import copy

from ..labels import Labels

MISSING_VALUE = -1
UNKNOWN_VALUE = 0


class FeatureParameters(Labels):
    def __init__(self, suffix, dim, size, dropout=0, updated=True, num=1, init=None, data=None, indexed=False,
                 copy_from=None, filename=None, min_count=1):
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
        """
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

    def __repr__(self):
        return type(self).__name__ + "(" + ", ".join(
            map(str, (self.suffix, self.dim, self.size, self.dropout, self.updated, self.num, self.init, self.data,
                      self.indexed, self.copy_from, self.filename, self.min_count))) + ")"

    def __eq__(self, other):
        return self.suffix == other.suffix and self.dim == other.dim and self.size == other.size and \
               self.dropout == other.dropout and self.updated == other.updated and self.num == other.num and \
               self.indexed == other.indexed and self.min_count == other.min_count and self.numeric == other.numeric

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
    def effective_suffix(self):
        return self.suffix if self.copy_from is None else self.copy_from

    @property
    def external(self):
        return self.copy_from is not None


class NumericFeatureParameters(FeatureParameters):
    SUFFIX = "numeric"

    def __init__(self, num):
        super().__init__(NumericFeatureParameters.SUFFIX, 1, None, num=num)

    def __repr__(self):
        return "%s(%d)" % (
            type(self).__name__, self.num)

    @property
    def numeric(self):
        return True


def copy_params(params, copy_dict=dict):
    params_copy = {}
    for suffix, param in params.items():
        param_copy = copy(param)
        if param.data is not None:
            param_copy.data = copy_dict(param.data)
            if hasattr(param_copy.data, "size"):  # It may be an UnknownDict but we still want it to know its size
                param_copy.data.size = param.size
        params_copy[suffix] = param_copy
    return params_copy
