import os
import shelve
import time
from collections import defaultdict

import numpy as np


class Weights(object):
    """
    The weights for one feature, for all labels
    """
    def __init__(self, num_labels, weights=None):
        self.num_labels = num_labels
        if weights is None:
            self.weights = np.zeros(num_labels, dtype=float)  # 0.01 * np.random.randn(num_labels)
            self.update_count = 0
            self._last_update = np.zeros(num_labels, dtype=int)
            self._totals = np.zeros(num_labels, dtype=float)
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
        self._last_update[label] = update_index
        n = update_index - self._last_update[label]
        self._totals[label] += n * self.weights[label]
        self.weights[label] += value

    def average(self, update_index, label_map):
        """
        Average weights over all updates, and keep only true label columns
        :param update_index: number of updates to average over
        :param label_map: list of label indices to keep
        :return new Weights object with the weights averaged
        """
        n = update_index - self._last_update[label_map]
        totals = self._totals[label_map] + n * self.weights[label_map]
        averaged_weights = totals / update_index
        return Weights(len(label_map), averaged_weights)


class AveragedPerceptron(object):
    def __init__(self, num_labels, min_update=1, weights=None, label_map=None):
        self.num_labels = num_labels
        self.weights = defaultdict(lambda: Weights(num_labels))
        self.is_frozen = weights is not None
        if self.is_frozen:
            self.weights.update(weights)
            self._label_map = label_map  # List of original indices for all current labels
        else:
            self._min_update = min_update  # Minimum number of updates for a feature to be used in scoring
            self._update_index = 0  # Counter for calls to update()
            self._true_labels = [False] * num_labels  # For is each, has it ever been a true label in update()?

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, in the form of a dict (name -> value)
        :return: score for each label: dict (label -> score)
        """
        scores = np.zeros(self.num_labels)
        for feature, value in features.items():
            if not value:
                continue
            weights = self.weights.get(feature)
            if weights is None or not self.is_frozen and weights.update_count < self._min_update:
                continue
            scores += value * weights.weights
        if self.is_frozen:
            return {self._label_map[i]: score for i, score in enumerate(scores)}
        else:
            return dict(enumerate(scores))

    def update(self, features, pred, true, learning_rate=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param learning_rate: how much to scale the feature vector for the weight update
        """
        assert not self.is_frozen, "Cannot update a frozen model"
        self._update_index += 1
        self._true_labels[true] = True
        for feature, value in features.items():
            if not value:
                continue
            weights = self.weights[feature]
            weights.update(true, learning_rate * value, self._update_index)
            weights.update(pred, -learning_rate * value, self._update_index)

    def average(self):
        """
        Average all weights over all updates, as a form of regularization
        :return new AveragedPerceptron object with the weights averaged
        """
        started = time.time()
        # Freeze set of features and set of labels; also allow pickle
        label_map = [i for i, is_true in enumerate(self._true_labels) if is_true]
        print("Averaging weights (keeping %d true labels out of the original %d)... " % (
            len(label_map), self.num_labels), end="", flush=True)
        averaged_weights = {feature: weights.average(self._update_index, label_map)
                            for feature, weights in self.weights.items()
                            if weights.update_count >= self._min_update}
        averaged = AveragedPerceptron(len(label_map), weights=averaged_weights, label_map=label_map)
        print("Done (%.3fs)." % (time.time() - started))
        return averaged

    def save(self, filename):
        """
        Save all parameters to file
        :param filename: file to write to; the actual written file may have an additional suffix
        """
        print("Saving model to '%s'... " % filename, end="", flush=True)
        started = time.time()
        with shelve.open(filename) as db:
            db["num_labels"] = self.num_labels
            db["weights"] = dict(self.weights)
            db["is_frozen"] = self.is_frozen
            if self.is_frozen:
                db["_label_map"] = self._label_map
            else:
                db["_min_update"] = self._min_update
                db["_update_index"] = self._update_index
                db["_true_labels"] = self._true_labels
        print("Done (%.3fs)." % (time.time() - started))

    def load(self, filename):
        """
        Load all parameters from file
        :param filename: file to read from; the actual read file may have an additional suffix
        """
        def try_open(*names):
            for f in names:
                # noinspection PyBroadException
                try:
                    return shelve.open(f, flag="r")
                except Exception as e:
                    exception = e
            raise IOError("Model file not found: " + filename) from exception

        print("Loading model from '%s'... " % filename, end="", flush=True)
        started = time.time()
        with try_open(filename, os.path.splitext(filename)[0]) as db:
            self.num_labels = db["num_labels"]
            self.weights.clear()
            self.weights.update(db["weights"])
            self.is_frozen = db["is_frozen"]
            if self.is_frozen:
                self._label_map = db["_label_map"]
            else:
                self._min_update = db["_min_update"]
                self._update_index = db["_update_index"]
                self._true_labels = db["_true_labels"]
        print("Done (%.3fs)." % (time.time() - started))
