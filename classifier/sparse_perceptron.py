import time
from collections import defaultdict

import numpy as np

from parsing.classifier.classifier import Classifier


class FeatureWeights(object):
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

    def finalize(self, update_index, average=True):
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


class SparsePerceptron(Classifier):
    """
    Multi-class averaged perceptron with min-update for sparse features.
    Keeps weights in a dictionary by feature name, allowing adding new features on-the-fly.
    Also allows adding new labels on-the-fly.
    """

    def __init__(self, labels=None, min_update=1, weights=None):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param min_update: minimum number of updates to a feature required for consideration
        :param weights: if given, copy the weights (from a trained model)
        """
        super(SparsePerceptron, self).__init__(labels=labels, weights=weights)
        assert labels is not None or weights is not None
        self.weights = defaultdict(lambda: FeatureWeights(self.num_labels))
        if self.is_frozen:
            self.weights.update(weights)
        else:
            self._min_update = min_update  # Minimum number of updates for a feature to be used in scoring
            self._update_index = 0  # Counter for calls to update()

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, in the form of a dict (name -> value)
        :return: score for each label: dict (label -> score)
        """
        if not self.is_frozen:
            self._update_num_labels()
        scores = np.zeros(self.num_labels)
        for feature, value in features.items():
            if not value:
                continue
            weights = self.weights.get(feature)
            if weights is None or not self.is_frozen and weights.update_count < self._min_update:
                continue
            scores += value * weights.weights
        return dict(enumerate(scores))

    def update(self, features, pred, true, learning_rate=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param learning_rate: how much to scale the feature vector for the weight update
        """
        super(SparsePerceptron, self).update(features, pred, true, learning_rate)
        self._update_index += 1
        for feature, value in features.items():
            if not value:
                continue
            weights = self.weights[feature]
            weights.update(true, learning_rate * value, self._update_index)
            weights.update(pred, -learning_rate * value, self._update_index)

    def resize(self):
        for weights in self.weights.values():
            weights.resize(self.num_labels)
        self.weights.default_factory = lambda: FeatureWeights(self.num_labels)

    def finalize(self, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :return new SparsePerceptron object with the weights averaged
        """
        super(SparsePerceptron, self).finalize(average=average)
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        weights = {f: w.finalize(self._update_index, average=average)
                   for f, w in self.weights.items() if w.update_count >= self._min_update}
        finalized = SparsePerceptron(list(self.labels), weights=weights)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d original, %d new" % (
            self._init_num_labels, self.num_labels - self._init_num_labels))
        print("Features: %d overall, %d occurred at least %d times" % (
            self.num_features, len(weights), self._min_update))
        return finalized

    def save(self, filename, io):
        """
        Save all parameters to file
        :param filename: file to save to
        :param io: module with 'save' function to write a dictionary to file
        """
        d = {
            "labels": self.labels,
            "weights": dict(self.weights),
            "is_frozen": self.is_frozen,
        }
        if not self.is_frozen:
            d.update({
                "_min_update": self._min_update,
                "_update_index": self._update_index,
            })
        io.save(filename, d)

    def load(self, filename, io):
        """
        Load all parameters from file
        :param filename: file to load from
        :param io: module with 'load' function to read a dictionary from file
        """
        d = io.load(filename)
        self.labels = list(d["labels"])
        self.weights.clear()
        self.weights.update(d["weights"])
        self.is_frozen = d["is_frozen"]
        if not self.is_frozen:
            self._min_update = d["_min_update"]
            self._update_index = d["_update_index"]

    @property
    def num_features(self):
        return len(self.weights)

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.num_features)

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            print(sep.join(["feature"] + list(map(str, self.labels))), file=f)
            for feature, weights in self.weights.items():
                print(sep.join([feature] +
                               ["%.8f" % w for w in weights.weights]), file=f)
