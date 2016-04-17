import time

import numpy as np

from classifiers.classifier import Classifier


class DensePerceptron(Classifier):
    """
    Multi-class averaged perceptron for dense features.
    Keeps weights in a constant-size matrix. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly.
    """

    def __init__(self, labels=None, num_features=None, weights=None):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param num_features: number of features that will be used for the matrix size
        :param weights: if given, copy the weights (from a trained model)
        """
        super(DensePerceptron, self).__init__(labels=labels, weights=weights)
        assert labels is not None and num_features is not None or weights is not None
        if self.is_frozen:
            self.weights = weights
        else:
            self._num_labels = self.num_labels
            self.num_features = num_features
            self.weights = np.zeros((self.num_features, self.num_labels), dtype=float)
            self._totals = np.zeros((self.num_features, self.num_labels), dtype=float)
            self._last_update = np.zeros(self.num_labels, dtype=int)
            self._update_index = 0  # Counter for calls to update()

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size num_features
        :return: array with score for each label
        """
        if not self.is_frozen:
            self._update_num_labels()
        return self.weights.T.dot(features)

    def update(self, features, pred, true, learning_rate=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size num_features
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param learning_rate: how much to scale the feature vector for the weight update
        """
        super(DensePerceptron, self).update(features, pred, true, learning_rate)
        self._update_index += 1
        self._update(pred, -learning_rate * features)
        self._update(true, learning_rate * features)

    def _update(self, label, values):
        self._update_totals(label)
        self.weights[:, label] += values

    def _update_totals(self, label=None):
        self._totals[:, label] += self.weights[:, label] * (self._update_index - self._last_update[label])
        self._last_update[label] = self._update_index

    def resize(self):
        self.weights.resize((self.num_features, self.num_labels), refcheck=False)
        self._totals.resize((self.num_features, self.num_labels), refcheck=False)
        self._last_update.resize(self.num_labels, refcheck=False)

    def finalize(self, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :return new DensePerceptron object with the weights averaged
        """
        super(DensePerceptron, self).finalize(average=average)
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        self._update_totals()
        weights = self._totals / self._update_index if average else self.weights
        finalized = DensePerceptron(list(self.labels), weights=weights)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d original, %d new" % (
            self._init_num_labels, self.num_labels - self._init_num_labels))
        print("Features: %d" % self.num_features)
        return finalized

    def save(self, filename, io):
        """
        Save all parameters to file
        :param filename: file to save to
        :param io: module with 'save' function to write a dictionary to file
        """
        d = {
            "type": "dense",
            "labels": self.labels,
            "weights": self.weights,
            "is_frozen": self.is_frozen,
        }
        if not self.is_frozen:
            d.update({
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
        model_type = d.get("type")
        assert model_type == "dense", "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.weights = d["weights"]
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
            for row in self.weights:
                print(sep.join(["%.8f" % w for w in row]), file=f)
