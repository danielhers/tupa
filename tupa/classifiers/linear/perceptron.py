import time

from tupa.classifiers.classifier import Classifier, ClassifierProperty


class Perceptron(Classifier):
    """
    Abstract multi-class averaged perceptron.
    """

    def __init__(self, *args, model=None, epoch=0):
        """
        Create a new untrained Perceptron or copy the weights from an existing one
        :param model: if given, copy the weights (from a trained model)
        """
        super(Perceptron, self).__init__(*args, model=model)
        if self.is_frozen:
            self.model = model
        self.initial_learning_rate = self.learning_rate if self.learning_rate else 1.0
        self.epoch = epoch
        self.update_learning_rate()

    def update(self, features, axis, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size num_features
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true label (non-negative integer bounded by num_labels[axis])
        :param importance: how much to scale the feature vector for the weight update
        """
        super(Perceptron, self).update(features, axis, pred, true, importance)
        self.updates += 1

    def resize(self, axis=None):
        raise NotImplementedError()

    def finalize(self, finished_epoch=False, average=True):
        """
        Average all weights over all updates, as a form of regularization
        :param average: whether to really average the weights or just return them as they are now
        :param finished_epoch: whether to decay the learning rate and drop rare features
        :return new Perceptron object with the weights averaged
        """
        super(Perceptron, self).finalize()
        started = time.time()
        if finished_epoch:
            self.epoch += 1
        if average:
            print("Averaging weights... ", end="", flush=True)
        finalized = self._finalize_model(finished_epoch, average)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: " + self.num_labels_str())
        print("Features: %d" % self.input_dim)
        return finalized

    def _finalize_model(self, finished_epoch, average):
        raise NotImplementedError()

    def update_learning_rate(self):
        self.learning_rate = self.initial_learning_rate / (1.0 + self.epoch * self.learning_rate_decay)

    def save_model(self):
        """
        Save all parameters to file
        """
        d = {
            "initial_learning_rate": self.initial_learning_rate,
        }
        d.update(self.save_extra())
        return d

    def save_extra(self):
        return {"model": self.model}

    def load_model(self, d):
        self.initial_learning_rate = d["initial_learning_rate"]
        self.load_extra(d)

    def load_extra(self, d):
        self.model = d["model"]

    def get_classifier_properties(self):
        return super(Perceptron, self).get_classifier_properties() + \
               (ClassifierProperty.update_only_on_error,)
