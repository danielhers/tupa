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
        self.num_labels = num_labels
        self.weights.resize(num_labels, refcheck=False)
        self._last_update.resize(num_labels)
        self._totals.resize(num_labels)


class AveragedPerceptron(object):
    def __init__(self, labels=None, min_update=1, weights=None, label_map=None):
        self.labels = labels or []
        self._init_num_labels = len(self.labels)
        self.weights = defaultdict(lambda: Weights(self.num_labels))
        self.is_frozen = weights is not None
        self._label_map = label_map  # List of original indices for all current labels
        if self.is_frozen:
            self.weights.update(weights)
        else:
            self._min_update = min_update  # Minimum number of updates for a feature to be used in scoring
            self._update_index = 0  # Counter for calls to update()
            self._true_labels = [False] * self.num_labels  # Has it ever been a true label in update()?

    @property
    def num_labels(self):
        return len(self.labels)

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
        return dict(enumerate(scores)) if self._label_map is None else \
            {self._label_map[i]: score for i, score in enumerate(scores)}

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
        self._update_num_labels()
        self._true_labels[true] = True
        for feature, value in features.items():
            if not value:
                continue
            weights = self.weights[feature]
            weights.update(true, learning_rate * value, self._update_index)
            weights.update(pred, -learning_rate * value, self._update_index)

    def _update_num_labels(self):
        if self.num_labels > len(self._true_labels):
            self._true_labels += [False] * (self.num_labels - len(self._true_labels))
            for weights in self.weights.values():
                weights.resize(self.num_labels)
            self.weights.default_factory = lambda: Weights(self.num_labels)

    def average(self):
        """
        Average all weights over all updates, as a form of regularization
        :return new AveragedPerceptron object with the weights averaged
        """
        assert not self.is_frozen, "Cannot freeze a frozen model"
        started = time.time()
        # Freeze set of features and set of labels; also allow pickle
        self._update_num_labels()
        label_map, labels = zip(*[(i, l) for i, (t, l) in
                                  enumerate(zip(self._true_labels, self.labels)) if t])
        print("Averaging weights... ", end="", flush=True)
        averaged_weights = {feature: weights.average(self._update_index, list(label_map))
                            for feature, weights in self.weights.items()
                            if weights.update_count >= self._min_update}
        averaged = AveragedPerceptron(labels, weights=averaged_weights, label_map=label_map)
        print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d original, %d new, %d removed (%s)" % (
            self._init_num_labels,
            len(label_map) - self._init_num_labels,
            self.num_labels - len(label_map),
            ", ".join(str(l) for i, l in enumerate(self.labels) if not self._true_labels[i])))
        print("Features: %d overall, %d occurred at least %d times" % (
            len(self.weights), len(averaged_weights), self._min_update))
        return averaged

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
                "_true_labels": self._true_labels,
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
            self._true_labels = d["_true_labels"]

    def __str__(self):
        return ("%d labels total, " % self.num_labels) + (
                "frozen" if self.is_frozen else
                "%d labels occurred" % self._true_labels.count(True))

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            print(sep.join(["feature"] + list(map(str, self.labels))), file=f)
            for feature, weights in self.weights.items():
                print(sep.join([feature] +
                               ["%.8f" % w for w in weights.weights]), file=f)
