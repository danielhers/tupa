import time

import os

from classifiers.classifier import Classifier
from parsing.model_util import load_dict, save_dict


class Perceptron(Classifier):
    """
    Abstract multi-class averaged perceptron.
    """

    def __init__(self, *args, model=None):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param model: if given, copy the weights (from a trained model)
        """
        super(Perceptron, self).__init__(*args, model=model)
        if self.is_frozen:
            self.model = model
        self._update_index = 0  # Counter for calls to update()

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size num_features
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: how much to scale the feature vector for the weight update
        """
        super(Perceptron, self).update(features, pred, true, importance)
        self._update_index += 1

    def finalize(self, average=True, finished_epoch=False):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :param finished_epoch: whether to decay the learning rate
        :return new Perceptron object with the weights averaged
        """
        super(Perceptron, self).finalize()
        started = time.time()
        if average:
            print("Averaging weights... ", end="", flush=True)
        finalized = self._finalize_model(average)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d" % self.num_labels)
        print("Features: %d" % self.num_features)
        return finalized

    def _finalize_model(self, average):
        raise NotImplementedError()

    def resize(self):
        raise NotImplementedError()

    def save(self):
        """
        Save all parameters to file
        """
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "is_frozen": self.is_frozen,
            "_update_index": self._update_index
        }
        d.update(self.save_model())
        save_dict(self.filename, d)

    def save_model(self):
        return {"model": self.model}

    def load(self):
        """
        Load all parameters from file
        """
        d = load_dict(self.filename)
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, \
            "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.is_frozen = d["is_frozen"]
        self._update_index = d["_update_index"]
        self.load_model(d)

    def load_model(self, d):
        self.model = d["model"]

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.num_features)

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            self.write_model(f, sep)
