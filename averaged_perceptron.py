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

    def resize(self, num_labels):
        if num_labels > self.num_labels:
            self.num_labels = num_labels
            self.weights.resize(num_labels)
            self._last_update.resize(num_labels)
            self._totals.resize(num_labels)


class AveragedPerceptron(object):
    def __init__(self, num_labels, min_update=1, weights=None, label_map=None):
        self._init_num_labels = num_labels
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
        num_labels = max(true, pred) + 1
        if num_labels > self.num_labels:
            self._true_labels += [False] * (num_labels - self.num_labels)
            self.num_labels = num_labels
            for weights in self.weights.values():
                weights.resize(num_labels)
            self.weights.default_factory = lambda: Weights(num_labels)
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
        print("Averaging weights (labels: %d original, %d new, %d removed)... " % (
            self._init_num_labels,
            len(label_map) - self._init_num_labels,
            self.num_labels - len(label_map)),
              end="", flush=True)
        averaged_weights = {feature: weights.average(self._update_index, label_map)
                            for feature, weights in self.weights.items()
                            if weights.update_count >= self._min_update}
        averaged = AveragedPerceptron(len(label_map), weights=averaged_weights, label_map=label_map)
        print("Done (%.3fs)." % (time.time() - started))
        return averaged

    def save(self):
        """
        Return dictionary of all parameters for saving
        """
        d = {
            "num_labels": self.num_labels,
            "weights": dict(self.weights),
            "is_frozen": self.is_frozen,
        }
        if self.is_frozen:
            d["_label_map"] = self._label_map
        else:
            d.update({
                "_min_update": self._min_update,
                "_update_index": self._update_index,
                "_true_labels": self._true_labels,
            })
        return d

    def load(self, d):
        """
        Load all parameters from dictionary
        :param d: dictionary to load from
        """
        self.num_labels = d["num_labels"]
        self.weights.clear()
        self.weights.update(d["weights"])
        self.is_frozen = d["is_frozen"]
        if self.is_frozen:
            self._label_map = d["_label_map"]
        else:
            self._min_update = d["_min_update"]
            self._update_index = d["_update_index"]
            self._true_labels = d["_true_labels"]

    def __str__(self):
        return ("%d labels total, " % self.num_labels) + (
                "frozen" if self.is_frozen else
                "%d labels occurred" % self._true_labels.count(True))
