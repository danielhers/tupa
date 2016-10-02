import time
from collections import defaultdict

import numpy as np

from classifiers.classifier import Classifier
from parsing import config
from parsing.model_util import load_dict, save_dict


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
    Expects features from SparseFeatureExtractor.
    """

    def __init__(self, filename, labels=None, min_update=1, model=None):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param min_update: minimum number of updates to a feature required for consideration
        :param model: if given, copy the weights (from a trained model)
        """
        super(SparsePerceptron, self).__init__(model_type=config.SPARSE_PERCEPTRON, filename=filename,
                                               labels=labels, model=model)
        assert labels is not None or model is not None
        self.model = defaultdict(lambda: FeatureWeights(self.num_labels))
        if self.is_frozen:
            self.model.update(model)
        else:
            self._min_update = min_update  # Minimum number of updates for a feature to be used in scoring
            self._update_index = 0  # Counter for calls to update()

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, in the form of a dict (name -> value)
        :return: array with score for each label
        """
        super(SparsePerceptron, self).score(features)
        scores = np.zeros(self.num_labels)
        for feature, value in features.items():
            if not value:
                continue
            weights = self.model.get(feature)
            if weights is None or not self.is_frozen and weights.update_count < self._min_update:
                continue
            scores += value * weights.weights
        return scores

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: how much to scale the feature vector for the weight update
        """
        super(SparsePerceptron, self).update(features, pred, true, importance)
        self._update_index += 1
        for feature, value in features.items():
            if not value:
                continue
            weights = self.model[feature]
            weights.update(true, importance * value, self._update_index)
            weights.update(pred, -importance * value, self._update_index)

    def resize(self):
        for weights in self.model.values():
            weights.resize(self.num_labels)
        self.model.default_factory = lambda: FeatureWeights(self.num_labels)

    def finalize(self, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :return new SparsePerceptron object with the weights averaged
        """
        super(SparsePerceptron, self).finalize()
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        model = {f: w.finalize(self._update_index, average=average)
                 for f, w in self.model.items() if w.update_count >= self._min_update}
        finalized = SparsePerceptron(self.filename, list(self.labels), model=model)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d" % self.num_labels)
        print("Features: %d overall, %d occurred at least %d times" % (
            self.num_features, len(model), self._min_update))
        return finalized

    def save(self):
        """
        Save all parameters to file
        :param filename: file to save to
        """
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "model": dict(self.model),
            "is_frozen": self.is_frozen,
        }
        if not self.is_frozen:
            d.update({
                "_min_update": self._min_update,
                "_update_index": self._update_index,
            })
        save_dict(self.filename, d)

    def load(self):
        """
        Load all parameters from file
        :param filename: file to load from
        """
        d = load_dict(self.filename)
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, \
            "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.model.clear()
        self.model.update(d["model"])
        self.is_frozen = d["is_frozen"]
        if not self.is_frozen:
            self._min_update = d["_min_update"]
            self._update_index = d["_update_index"]

    @property
    def num_features(self):
        return len(self.model)

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.num_features)

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            print(sep.join(["feature"] + list(map(str, self.labels))), file=f)
            for feature, weights in self.model.items():
                print(sep.join([feature] +
                               ["%.8f" % w for w in weights.weights]), file=f)
