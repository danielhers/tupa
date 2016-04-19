from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in a constant-size matrix. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly.
    """

    def __init__(self, labels=None, input_dim=None, model=None, max_num_labels=100):
        """
        Create a new untrained NN or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param input_dim: number of features that will be used for the input matrix
        :param model: if given, copy the weights (from a trained model)
        """
        super(NeuralNetwork, self).__init__(labels=labels, model=model)
        assert labels is not None and input_dim is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self.max_num_labels = max_num_labels
            self._num_labels = self.num_labels
            self._input_dim = input_dim
            self.model = Sequential()
            self.model.add(Dense(self.max_num_labels, input_dim=input_dim, init="uniform"))
            self.model.add(Activation("softmax"))
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss="categorical_crossentropy", optimizer=sgd)

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :return: array with score for each label
        """
        super(NeuralNetwork, self).score(features)
        scores = self.model.predict(features.T, batch_size=1).reshape((-1,))
        return scores[:self.num_labels]

    def update(self, features, pred, true, learning_rate=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param learning_rate: how much to scale the feature vector for the weight update
        """
        super(NeuralNetwork, self).update(features, pred, true, learning_rate)
        y = np_utils.to_categorical((true,), nb_classes=self.max_num_labels)
        self.model.fit(features.T, y, batch_size=1, nb_epoch=10, verbose=0)

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def finalize(self):
        """
        Return a frozen model
        :return new NeuralNetwork object with the same weights
        """
        super(NeuralNetwork, self).finalize()
        finalized = NeuralNetwork(list(self.labels), model=self.model)
        print("Labels: %d" % self.num_labels)
        print("Features: %d" % self._input_dim)
        return finalized

    def save(self, filename, io):
        """
        Save all parameters to file
        :param filename: file to save to
        :param io: module with 'save' function to write a dictionary to file
        """
        d = {
            "type": "nn",
            "labels": self.labels,
            "model": self.model,
            "is_frozen": self.is_frozen,
        }
        io.save(filename, d)

    def load(self, filename, io):
        """
        Load all parameters from file
        :param filename: file to load from
        :param io: module with 'load' function to read a dictionary from file
        """
        d = io.load(filename)
        model_type = d.get("type")
        assert model_type == "nn", "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.model = d["model"]
        self.is_frozen = d["is_frozen"]

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self._input_dim)

    def write(self, filename, sep="\t"):
        print("Writing model to '%s'..." % filename)
        with open(filename, "w") as f:
            print(list(map(str, self.labels)), file=f)
            for row in self.model:
                print(sep.join(["%.8f" % w for w in row]), file=f)