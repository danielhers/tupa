import time

from classifiers.classifier import Classifier, ClassifierProperty


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
        self._update_index = 0  # Counter for calls to update()
        self.initial_learning_rate = self.learning_rate
        self.epoch = epoch
        self.update_learning_rate()

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
        if finished_epoch:
            self.epoch += 1
        if average:
            print("Averaging weights... ", end="", flush=True)
        finalized = self._finalize_model(average)
        if average:
            print("Done (%.3fs)." % (time.time() - started))
        print("Labels: %d" % self.num_labels)
        print("Features: %d" % self.input_dim)
        return finalized

    def _finalize_model(self, average):
        raise NotImplementedError()

    def update_learning_rate(self):
        self.learning_rate = self.initial_learning_rate / (1.0 + self.epoch * self.learning_rate_decay)

    def resize(self):
        raise NotImplementedError()

    def save_model(self):
        """
        Save all parameters to file
        """
        d = {
            "_update_index": self._update_index,
            "initial_learning_rate": self.initial_learning_rate,
            "epoch": self.epoch,
        }
        d.update(self.save_extra())
        return d

    def save_extra(self):
        return {"model": self.model}

    def load_model(self, d):
        self._update_index = d["_update_index"]
        self.initial_learning_rate = d["initial_learning_rate"]
        self.epoch = d["epoch"]
        self.load_extra(d)

    def load_extra(self, d):
        self.model = d["model"]

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            self.write_model(f, sep)

    def write_model(self, f, sep):
        raise NotImplementedError()

    def get_classifier_properties(self):
        return super(Perceptron, self).get_classifier_properties() + \
               (ClassifierProperty.update_only_on_error,)
