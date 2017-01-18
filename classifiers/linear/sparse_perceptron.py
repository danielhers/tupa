from collections import defaultdict

import numpy as np

from linear.perceptron import Perceptron
from parsing.config import Config, SPARSE_PERCEPTRON


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


class SparsePerceptron(Perceptron):
    """
    Multi-class averaged perceptron with min-update for sparse features.
    Keeps weights in a dictionary by feature name, allowing adding new features on-the-fly.
    Also allows adding new labels on-the-fly.
    Expects features from SparseFeatureExtractor.
    """

    def __init__(self, *args, model=None, epoch=0):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param min_update: minimum number of updates to a feature required for consideration
        :param model: if given, copy the weights (from a trained model)
        """
        super(SparsePerceptron, self).__init__(SPARSE_PERCEPTRON, *args, model=model, epoch=epoch)
        model = defaultdict(lambda: FeatureWeights(self.num_labels))
        if self.is_frozen:
            model.update(self.model)
        self.model = model
        self._min_update = Config().args.min_update  # Minimum number of updates for a feature to be used in scoring

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
        for feature, value in features.items():
            if not value:
                continue
            weights = self.model[feature]
            weights.update(true, importance * self.learning_rate * value, self._update_index)
            weights.update(pred, -importance * self.learning_rate * value, self._update_index)

    def resize(self):
        for weights in self.model.values():
            weights.resize(self.num_labels)
        self.model.default_factory = lambda: FeatureWeights(self.num_labels)

    def _finalize_model(self, average):
        model = {f: w.finalize(self._update_index, average=average)
                 for f, w in self.model.items() if w.update_count >= self._min_update}
        print("%d features occurred at least %d times" % (len(model), self._min_update))
        return SparsePerceptron(self.filename, list(self.labels), model=model, epoch=self.epoch)

    def save_extra(self):
        return {
            "model": dict(self.model),
            "_min_update": self._min_update,
        }

    def load_extra(self, d):
        self.model.clear()
        self.model.update(d["model"])
        self._min_update = d["_min_update"]

    @property
    def input_dim(self):
        return len(self.model)

    def write_model(self, f, sep):
        print(sep.join(["feature"] + list(map(str, self.labels))), file=f)
        for feature, weights in self.model.items():
            print(sep.join([feature] +
                           ["%.8f" % w for w in weights.weights]), file=f)
