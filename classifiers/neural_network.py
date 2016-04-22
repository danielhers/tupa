import time
from collections import defaultdict

import numpy as np
from keras.layers import Input, Dense, merge
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureIndexer.
    """

    def __init__(self, labels=None, inputs=None, model=None,
                 max_num_labels=100, batch_size=None,
                 minibatch_size=200, nb_epochs=5):
        """
        Create a new untrained NN or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param inputs: dict of feature type name -> FeatureInformation
        :param model: if given, copy the weights (from a trained model)
        :param max_num_labels: since model size is fixed, set maximum output size
        :param batch_size: if given, fit model every this many samples
        :param minibatch_size: batch size for SGD
        :param nb_epochs: number of epochs for SGD
        """
        super(NeuralNetwork, self).__init__(labels=labels, model=model)
        assert inputs is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self.max_num_labels = max_num_labels
            self._num_labels = self.num_labels
            self._batch_size = batch_size
            self._minibatch_size = minibatch_size
            self._nb_epochs = nb_epochs
            self.feature_types = inputs
            self.model = self.build_model(inputs, max_num_labels)
            self.init_samples()
            self._iteration = 0

    def init_samples(self):
        self._samples = defaultdict(list)
        self._update_index = 0

    @staticmethod
    def build_model(feature_types, num_labels):
        inputs = []
        encoded = []
        for name, feature_type in feature_types.items():
            if feature_type.indices is None:  # numeric feature
                i = Input(shape=(feature_type.num,), name=name)
                x = BatchNormalization()(i)
            else:  # index feature
                i = Input(shape=(feature_type.num,), dtype="int32", name=name)
                x = Embedding(output_dim=feature_type.dim, input_dim=feature_type.size,
                              weights=feature_type.init, input_length=feature_type.num)(i)
                x = Flatten()(x)
            inputs.append(i)
            encoded.append(x)
        x = merge(encoded, mode="concat")
        out = Dense(num_labels, activation="softmax", name="out")(x)
        model = Model(input=inputs, output=[out])
        model.compile(optimizer="adam", loss={"out": "categorical_crossentropy"})
        return model

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :return: array with score for each label
        """
        super(NeuralNetwork, self).score(features)
        if not self.is_frozen and self._iteration == 0:  # not fit yet
            return np.zeros(self.num_labels)
        features = {k: np.array(v).reshape((1, -1)) for k, v in features.items()}
        scores = self.model.predict(features, batch_size=1).reshape((-1,))
        return scores[:self.num_labels]

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: how much to scale the feature vector for the weight update
        """
        super(NeuralNetwork, self).update(features, pred, true, importance)
        for name, value in features.items():
            self._samples[name].append(value)
        self._samples["out"].append(true)
        self._update_index += 1
        if self._batch_size is not None and self._update_index >= self._batch_size:
            self.finalize(freeze=False)

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def finalize(self, freeze=True):
        """
        Fit this model on collected samples, and return a frozen model
        :return new NeuralNetwork object with the same weights, after fitting
        """
        super(NeuralNetwork, self).finalize()
        started = time.time()
        print("\nFitting model...", flush=True)
        x = {}
        for name, values in self._samples.items():
            if name == "out":
                y = np_utils.to_categorical(values, nb_classes=self.max_num_labels)
            else:
                x[name] = np.array(values)
        self.model.fit(x, y, batch_size=self._minibatch_size,
                       nb_epoch=self._nb_epochs, verbose=2)
        self.init_samples()
        self._iteration += 1
        finalized = NeuralNetwork(list(self.labels), model=self.model) if freeze else None
        print("Done (%.3fs)." % (time.time() - started))
        if freeze:
            print("Labels: %d" % self.num_labels)
            print("Features: %d" % sum(f.num * (f.dim or 1) for f in self.feature_types.values()))
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
