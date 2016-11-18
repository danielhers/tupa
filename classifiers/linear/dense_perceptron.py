import numpy as np

from linear.perceptron import Perceptron
from parsing.config import DENSE_PERCEPTRON


class DensePerceptron(Perceptron):
    """
    Multi-class averaged perceptron for dense features.
    Keeps weights in a constant-size matrix. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly.
    Expects features from FeatureEmbedding.
    """

    def __init__(self, *args, model=None, num_features=None, epoch=0):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param num_features: number of features that will be used for the matrix size
        :param model: if given, copy the weights (from a trained model)
        """
        super(DensePerceptron, self).__init__(DENSE_PERCEPTRON, *args, model=model, epoch=epoch)
        if not self.is_frozen:
            self._num_labels = self.num_labels
            self.input_dim = num_features
            self.model = np.zeros((self.input_dim, self.num_labels), dtype=float)
            self._totals = np.zeros((self.input_dim, self.num_labels), dtype=float)
            self._last_update = np.zeros(self.num_labels, dtype=int)

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
        self._update(pred, -importance * self.learning_rate * features)
        self._update(true, importance * self.learning_rate * features)

    def _update(self, label, values):
        self._update_totals(label)
        self.model[:, label] += values.reshape((-1,))

    def _update_totals(self, label=None):
        self._totals[:, label] += self.model[:, label] * (self._update_index - self._last_update[label])
        self._last_update[label] = self._update_index

    def resize(self):
        self.model.resize((self.input_dim, self.num_labels), refcheck=False)
        self._totals.resize((self.input_dim, self.num_labels), refcheck=False)
        self._last_update.resize(self.num_labels, refcheck=False)

    def _finalize_model(self, average):
        model = self._totals / self._update_index if average else self.model
        return DensePerceptron(self.filename, list(self.labels), model=model, epoch=self.epoch)

    def write_model(self, f, sep):
        print(list(map(str, self.labels)), file=f)
        for row in self.model:
            print(sep.join(["%.8f" % w for w in row]), file=f)
