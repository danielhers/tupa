import time
from collections import defaultdict
from itertools import repeat

import numpy as np

from tupa.model_util import KeyBasedDefaultDict, save_dict, load_dict
from ..classifier import Classifier


class FeatureWeights:
    """
    The weights for one feature, for all labels
    """
    def __init__(self, num_labels=None, weights=None):
        if num_labels is not None and weights is None:
            self.weights = np.zeros(num_labels, dtype=float)  # 0.01 * np.random.randn(num_labels)
            self._totals = np.zeros(num_labels, dtype=float)
            self._last_update = np.zeros(num_labels, dtype=int)
            self.update_count = 0
        else:
            self.weights = weights

    def update(self, label, value, update_index):
        """
        Add a value to the entry of the given label
        :param label: label to index by
        :param value: value to add
        :param update_index: which update this is (for averaging)
        """
        self.update_count += 1
        self._update_totals(label, update_index)
        self.weights[label] += value

    def finalize(self, update_index, average):
        """
        Average weights over all updates, and keep only true label columns
        :param update_index: number of updates to average over
        :param average: whether to really average the weights or just return them as they are now
        :return new Weights object with the weights averaged and only selected indices remaining
        """
        self._update_totals(None, update_index)
        weights = self._totals / update_index if average else self.weights
        return FeatureWeights(weights=weights)

    def _update_totals(self, label, update_index):
        self._totals[label] += self.weights[label] * (update_index - self._last_update[label])
        self._last_update[label] = update_index

    def resize(self, num_labels):
        self.weights.resize(num_labels, refcheck=False)
        self._totals.resize(num_labels, refcheck=False)
        self._last_update.resize(num_labels, refcheck=False)


class FeatureWeightsCreator:
    def __init__(self, perceptron, axis):
        self.perceptron = perceptron
        self.axis = axis

    def create(self):
        return FeatureWeights(self.perceptron.num_labels[self.axis])


class SparsePerceptron(Classifier):
    """
    Multi-class averaged perceptron with min-update for sparse features.
    Keeps weights in a dictionary by feature name, allowing adding new features on-the-fly.
    Also allows adding new labels on-the-fly.
    Expects features from SparseFeatureExtractor.
    """

    def __init__(self, *args, epoch=0, **kwargs):
        """
        Create a new untrained SparsePerceptron or copy the weights from an existing one
        :param labels: tuple of lists of labels that can be updated later to add new labels
        :param min_update: minimum number of updates to a feature required for consideration
        :param model: if given, copy the weights (from a trained model)
        """
        super().__init__(*args, **kwargs)
        model = KeyBasedDefaultDict(self.create_axis_weights)
        if self.is_frozen:
            for axis, feature_weights in self.model.items():
                model[axis].update(feature_weights)
        self.model = model
        self.min_update = self.config.args.min_update  # Minimum number of updates for a feature to be used in scoring
        self.dropped = set()  # Features that did not get min_updates after a full epoch
        self.initial_learning_rate = self.learning_rate if self.learning_rate else 1.0
        self.epoch = epoch
        self.update_learning_rate()

    @property
    def input_dim(self):
        return {a: len(m) for a, m in self.model.items()}

    def update_learning_rate(self):
        self.learning_rate = self.initial_learning_rate / (1.0 + self.epoch * self.learning_rate_decay)

    def create_axis_weights(self, axis):
        return defaultdict(FeatureWeightsCreator(self, axis).create)

    def copy_model(self):
        return {a: dict(m) for a, m in self.model.items()}

    def update_model(self, model):
        for axis, feature_weights in model.items():
            self.model[axis].update(feature_weights)

    def score(self, features, axis):
        """
        Calculate score for each label
        :param features: extracted feature values, in the form of a dict (name -> value)
        :param axis: axis of the label we are predicting
        :return: array with score for each label
        """
        super().score(features, axis)
        scores = np.zeros(self.num_labels[axis])
        model = self.model[axis]
        for feature, value in features.items():
            if not value:
                continue
            weights = model.get(feature)
            if weights is not None:
                if self.is_frozen or weights.update_count >= self.min_update:
                    scores += value * weights.weights
        return scores

    def update(self, features, axis, pred, true, importance=None):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true labels (non-negative integers bounded by num_labels[axis])
        :param importance: how much to scale the update for the weight update for each true label
        """
        super().update(features, axis, pred, true, importance)
        self.updates += 1
        model = self.model[axis]
        for feature, value in features.items():
            if not value or feature in self.dropped:
                continue
            weights = model[feature]
            for t, i in zip(true, importance or repeat(1)):
                weights.update(t, i * self.learning_rate * value, self.updates)
            weights.update(pred, -self.learning_rate * value, self.updates)

    def resize(self):
        for axis, model in self.model.items():
            num_labels = self.num_labels[axis]
            for weights in model.values():
                weights.resize(num_labels)

    def finalize(self, finished_epoch=False, average=True, **kwargs):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :param finished_epoch: whether to decay the learning rate and drop rare features
        :return new SparsePerceptron object with the weights averaged
        """
        super().finalize(finished_epoch=finished_epoch, **kwargs)
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        finalized = self._finalize_model(finished_epoch, average)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print(self)
        return finalized

    def _finalize_model(self, finished_epoch, average):
        # If finished an epoch, remove rare features from our model directly. Otherwise, copy it.
        model, dropped = (self.model, self.dropped) if finished_epoch else (self.copy_model(), set())
        num_features = 0
        for axis, axis_model in self.model.items():
            for feature, weights in list(axis_model.items()):
                if weights.update_count < self.min_update:
                    del axis_model[feature]
                    dropped.add(feature)
            num_features += len(axis_model)
        print("%d features occurred at least %d times, dropped %d rare features" % (
            num_features, self.min_update, len(dropped)))
        finalized = {a: {f: w.finalize(self.updates, average=average) for f, w in m.items()} for a, m in model.items()}
        ret = SparsePerceptron(self.config, self.labels, epoch=self.epoch)
        ret.update_model(finalized)
        ret.is_frozen = True
        ret.min_update = self.min_update
        ret.updates = self.updates
        ret.initial_learning_rate = self.initial_learning_rate
        ret.learning_rate = self.learning_rate
        ret.learning_rate_decay = self.learning_rate_decay
        ret.epoch = self.epoch
        return ret

    def save_model(self, filename, d):
        super().save_model(filename, d)
        d.update((
            ("initial_learning_rate", self.initial_learning_rate),
            ("min_update", self.min_update),
        ))
        save_dict(filename + ".data", self.copy_model())

    def load_model(self, filename, d):
        self.model.clear()
        self.update_model(load_dict(filename + ".data"))
        self.initial_learning_rate = d["initial_learning_rate"]
        self.config.args.min_update = self.min_update = d["min_update"]
        super().load_model(filename, d)

    def all_params(self):
        d = super().all_params()
        d.update(("_".join((axis, k)), v.weights) for axis, model in self.model.items() for k, v in model.items())
        return d

    def print_params(self, max_rows=10):
        for axis, model in self.model.items():
            for key, value in model.params.items():
                print(axis, key)
                # noinspection PyBroadException
                try:
                    print(list(value.items())[:max_rows])
                except Exception:
                    pass
