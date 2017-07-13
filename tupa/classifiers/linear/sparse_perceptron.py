import numpy as np
from collections import defaultdict

from tupa.config import Config, SPARSE
from .perceptron import Perceptron


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
        :param labels: tuple of lists of labels that can be updated later to add new labels
        :param min_update: minimum number of updates to a feature required for consideration
        :param model: if given, copy the weights (from a trained model)
        """
        super(SparsePerceptron, self).__init__(SPARSE, *args, model=model, epoch=epoch)
        model = defaultdict(self.create_weights)
        if self.is_frozen:
            model.update(self.model)
        self.model = model
        self.input_dim = len(self.model)
        self.min_update = Config().args.min_update  # Minimum number of updates for a feature to be used in scoring
        self.dropped = set()  # Features that did not get min_updates after a full epoch

    def create_weights(self):
        return tuple(map(FeatureWeights, self.num_labels))

    def score(self, features, axis):
        """
        Calculate score for each label
        :param features: extracted feature values, in the form of a dict (name -> value)
        :param axis: axis of the label we are predicting
        :return: array with score for each label
        """
        super(SparsePerceptron, self).score(features, axis)
        scores = np.zeros(self.num_labels[axis])
        for feature, value in features.items():
            if not value:
                continue
            weights = self.model.get(feature)
            if weights is not None:
                w = weights[axis]
                if self.is_frozen or w.update_count >= self.min_update:
                    scores += value * w.weights
        return scores

    def update(self, features, axis, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true label (non-negative integer bounded by num_labels[axis])
        :param importance: how much to scale the feature vector for the weight update
        """
        super(SparsePerceptron, self).update(features, axis, pred, true, importance)
        for feature, value in features.items():
            if not value or feature in self.dropped:
                continue
            w = self.model[feature][axis]
            w.update(true, importance * self.learning_rate * value, self.updates)
            w.update(pred, -importance * self.learning_rate * value, self.updates)
        self.input_dim = len(self.model)

    def resize(self, axis=None):
        for weights in self.model.values():
            for i, (l, w) in enumerate(zip(self.num_labels, weights)):
                if axis is None or i == axis:
                    w.resize(l)

    def _finalize_model(self, finished_epoch, average):
        # If finished an epoch, remove rare features from our model directly. Otherwise, copy it.
        model, dropped = (self.model, self.dropped) if finished_epoch else (dict(self.model), set())
        for f, weights in list(model.items()):
            if weights[0].update_count < self.min_update:
                del model[f]
                dropped.add(f)
        print("%d features occurred at least %d times, dropped %d rare features" % (
            len(model), self.min_update, len(dropped)))
        finalized = {f: tuple(w.finalize(self.updates, average=average) for w in weights)
                     for f, weights in model.items()}
        return SparsePerceptron(self.filename, tuple(map(list, self.labels)), model=finalized, epoch=self.epoch)

    def save_extra(self):
        return {
            "model": dict(self.model),
            "min_update": self.min_update,
        }

    def load_extra(self, d):
        self.model.clear()
        self.model.update(d["model"])
        Config().args.min_update = self.min_update = d.get("min_update", d.get("_min_update"))
