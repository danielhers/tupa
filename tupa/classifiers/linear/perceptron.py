import time

from tupa.classifiers.classifier import Classifier


class Perceptron(Classifier):
    """
    Abstract multi-class averaged perceptron.
    """

    def __init__(self, *args, epoch, **kwargs):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        """
        super().__init__(*args, **kwargs)
        self.initial_learning_rate = self.learning_rate if self.learning_rate else 1.0
        self.epoch = epoch
        self.update_learning_rate()

    def update(self, features, axis, pred, true, importance=None):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size num_features
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true labels (non-negative integers bounded by num_labels[axis])
        :param importance: how much to scale the update for the weight update for each true label
        """
        super().update(features, axis, pred, true, importance)
        self.updates += 1

    def resize(self):
        raise NotImplementedError()

    def finalize(self, finished_epoch=False, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :param finished_epoch: whether to decay the learning rate and drop rare features
        :return new Perceptron object with the weights averaged
        """
        super().finalize()
        started = time.time()
        if finished_epoch:
            self.epoch += 1
        if average:
            print("Averaging weights... ", end="", flush=True)
        finalized = self._finalize_model(finished_epoch, average)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print(self)
        return finalized

    def _finalize_model(self, finished_epoch, average):
        raise NotImplementedError()

    def update_learning_rate(self):
        self.learning_rate = self.initial_learning_rate / (1.0 + self.epoch * self.learning_rate_decay)

    def save_model(self, filename, d):
        super().save_model(filename, d)
        d.update((
            ("initial_learning_rate", self.initial_learning_rate),
        ))

    def load_model(self, filename, d):
        self.initial_learning_rate = d["initial_learning_rate"]
        super().load_model(filename, d)
