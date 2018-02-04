import numpy as np

from .classifier import Classifier


class NoOp(Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(self, features, axis):
        super().score(features, axis)
        return np.zeros(self.num_labels[axis])

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def resize(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        super().finalize(*args, **kwargs)
        return self
