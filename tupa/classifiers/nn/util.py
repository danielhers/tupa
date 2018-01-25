from itertools import groupby

from scipy.stats import ortho_group

from ...config import Config


def dim(param):
    return param.shape()[0]


def is_square(param):
    shape = param.shape()
    return len(shape) == 2 and shape[0] == shape[1] > 1


def randomize_orthonormal(*parameters):  # Saxe et al., 2014 (https://arxiv.org/abs/1312.6120)
    for d, params in groupby(sorted(filter(is_square, parameters), key=dim), dim):
        params = list(params)
        for param, init in zip(params, ortho_group.rvs(d, size=len(params), random_state=Config().random)):
            param.set_value(init)
