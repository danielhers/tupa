from collections import Counter
from copy import copy

MISSING_VALUE = 0
UNKNOWN_VALUE = 1


class FeatureParameters(object):
    def __init__(self, suffix, dim, size, dropout=0, updated=True, num=1, init=None, data=None, indexed=False,
                 copy_from=None, filename=None):
        """
        :param suffix: one-character title for feature
        :param dim: vector dimension or, filename to load vectors from, or Word2Vec object
        :param size: maximum number of distinct values
        :param dropout: value of dropout parameter to use during training
        :param updated: whether the feature is learned (otherwise kept constant)
        :param num: how many such features exist per step
        :param init: array of values to use as initial value of embedding matrix
        :param data: dictionary of raw value to running numerical representation, or embedding matrix
        :param indexed: whether the feature is to be used as index into initialized values (otherwise used directly)
        :param copy_from: suffix of other parameter to copy values from instead of extracting them directly
        :param filename: name of file to load data from
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

    def __repr__(self):
        return "%s(%s, %d, %d, %f, %s, %s, %s, %s, %s, %s, %s)" % (
            self.__class__.__name__, self.suffix, self.dim, self.size, self.dropout, self.updated, self.num, self.init,
            self.data, self.indexed, self.copy_from, self.filename)

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
        super(NumericFeatureParameters, self).__init__(NumericFeatureParameters.SUFFIX, 1, None, num=num)

    def __repr__(self):
        return "%s(%d)" % (
            self.__class__.__name__, self.num)

    @property
    def numeric(self):
        return True


def copy_params(params, copy_dict=dict):
    params_copy = {}
    for suffix, param in params.items():
        param_copy = copy(param)
        if param.data is not None:
            param_copy.data = copy_dict(param.data)
        params_copy[suffix] = param_copy
    return params_copy
