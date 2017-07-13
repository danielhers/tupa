import numpy as np

from tupa.config import NOOP
from .classifier import Classifier


class NoOp(Classifier):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__(NOOP, *args, **kwargs)

    def score(self, features, axis):
        super(NoOp, self).score(features, axis)
        return np.zeros(self.num_labels[axis])

    def update(self, *args, **kwargs):
        super(NoOp, self).update(*args, **kwargs)

    def resize(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        super(NoOp, self).finalize(*args, **kwargs)
        return self
