import time

import numpy as np

from classifiers.classifier import Classifier
from parsing import config
from parsing.model_util import load_dict, save_dict


class DensePerceptron(Classifier):
    """
    Multi-class averaged perceptron for dense features.
    Keeps weights in a constant-size matrix. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly.
    Expects features from FeatureEmbedding.
    """

    def __init__(self, labels=None, num_features=None, model=None):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param num_features: number of features that will be used for the matrix size
        :param model: if given, copy the weights (from a trained model)
        """
        super(DensePerceptron, self).__init__(model_type=config.DENSE_PERCEPTRON, labels=labels, model=model)
        assert labels is not None and num_features is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self._num_labels = self.num_labels
            self.num_features = num_features
            self.model = np.zeros((self.num_features, self.num_labels), dtype=float)
            self._totals = np.zeros((self.num_features, self.num_labels), dtype=float)
            self._last_update = np.zeros(self.num_labels, dtype=int)
            self._update_index = 0  # Counter for calls to update()

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size num_features
        :return: array with score for each label
        """
        super(DensePerceptron, self).score(features)
        return self.model.T.dot(features).reshape((-1,))

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size num_features
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: how much to scale the feature vector for the weight update
        """
        super(DensePerceptron, self).update(features, pred, true, importance)
        self._update_index += 1
        self._update(pred, -importance * features)
        self._update(true, importance * features)

    def _update(self, label, values):
        self._update_totals(label)
        self.model[:, label] += values.reshape((-1,))

    def _update_totals(self, label=None):
        self._totals[:, label] += self.model[:, label] * (self._update_index - self._last_update[label])
        self._last_update[label] = self._update_index

    def resize(self):
        self.model.resize((self.num_features, self.num_labels), refcheck=False)
        self._totals.resize((self.num_features, self.num_labels), refcheck=False)
        self._last_update.resize(self.num_labels, refcheck=False)

    def finalize(self, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :return new DensePerceptron object with the weights averaged
        """
        super(DensePerceptron, self).finalize()
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        self._update_totals()
        model = self._totals / self._update_index if average else self.model
        finalized = DensePerceptron(list(self.labels), model=model)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d" % self.num_labels)
        print("Features: %d" % self.num_features)
        return finalized

    def save(self, filename):
        """
        Save all parameters to file
        :param filename: file to save to
        """
        d = {
            "type": "dense",
            "labels": self.labels,
            "model": self.model,
            "is_frozen": self.is_frozen,
        }
        if not self.is_frozen:
            d.update({
                "_update_index": self._update_index,
            })
        save_dict(filename, d)

    def load(self, filename):
        """
        Load all parameters from file
        :param filename: file to load from
        """
        d = load_dict(filename)
        model_type = d.get("type")
        assert model_type == "dense", "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.model = d["model"]
        self.is_frozen = d["is_frozen"]
        if not self.is_frozen:
            self._update_index = d["_update_index"]

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.num_features)

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            print(list(map(str, self.labels)), file=f)
            for row in self.model:
                print(sep.join(["%.8f" % w for w in row]), file=f)
