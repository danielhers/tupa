from collections import Counter
from copy import copy


class FeatureParameters(object):
    def __init__(self, suffix, dim, size, dropout=0, num=1, init=None, data=None):
        self.suffix = suffix
        self.dim = dim
        self.size = size
        self.dropout = dropout
        self.num = num
        self.init = init
        self.data = data
        self.counts = Counter() if self.dropout else None

    def __repr__(self):
        return "%s(%s, %d, %d, %f, %s, %s, %s)" % (
            self.__class__.__name__, self.suffix, self.dim, self.size, self.dropout, self.num, self.init, self.data)

    @property
    def numeric(self):
        return False


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
