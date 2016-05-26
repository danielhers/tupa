import time
from collections import defaultdict

import numpy as np
from keras.layers import Input, Dense, merge
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.utils import np_utils

from classifiers.classifier import Classifier
from parsing import config
from parsing.model_util import load_dict, save_dict


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureIndexer.
    """

    def __init__(self, labels=None, feature_params=None, model=None,
                 layers=1, layer_dim=100, activation="tanh",
                 init="glorot_normal", max_num_labels=100, batch_size=None,
                 minibatch_size=200, nb_epochs=5,
                 optimizer="adam", loss="categorical_crossentropy"):
        """
        Create a new untrained NN or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param feature_params: dict of feature type name -> FeatureInformation
        :param model: if given, copy the weights (from a trained model)
        :param layers: number of hidden layers
        :param layer_dim: size of hidden layer
        :param activation: activation function at hidden layers
        :param init: initialization type for hidden layers
        :param max_num_labels: since model size is fixed, set maximum output size
        :param batch_size: if given, fit model every this many samples
        :param minibatch_size: batch size for SGD
        :param nb_epochs: number of epochs for SGD
        :param optimizer: algorithm to use for optimization
        :param loss: objective function to use for optimization
        """
        super(NeuralNetwork, self).__init__(model_type=config.NEURAL_NETWORK, labels=labels, model=model)
        assert feature_params is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self.max_num_labels = max_num_labels
            self._layers = layers
            self._layer_dim = layer_dim
            self._activation = activation
            self._init = init
            self._num_labels = self.num_labels
            self._batch_size = batch_size
            self._minibatch_size = minibatch_size
            self._nb_epochs = nb_epochs
            self._optimizer = optimizer
            self._loss = loss
            self.feature_params = feature_params
            self.model = None
            self._samples = defaultdict(list)
            self._update_index = 0
            self._iteration = 0

    def init_model(self):
        if self.model is not None:
            return
        if config.Config().args.verbose:
            print("Input: " + self.feature_params)
        inputs = []
        encoded = []
        for suffix, param in self.feature_params.items():
            if param.data is None:  # numeric feature
                i = Input(shape=(param.num,), name=suffix)
                x = BatchNormalization()(i)
            else:  # index feature
                i = Input(shape=(param.num,), dtype="int32", name=suffix)
                x = Embedding(output_dim=param.dim, input_dim=param.size,
                              weights=param.init, input_length=param.num)(i)
                x = Flatten()(x)
            inputs.append(i)
            encoded.append(x)
        x = merge(encoded, mode="concat")
        for _ in range(self._layers):
            x = Dense(self._layer_dim, activation=self._activation, init=self._init)(x)
        out = Dense(self.max_num_labels, activation="softmax", init=self._init, name="out")(x)
        self.model = Model(input=inputs, output=[out])
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss={"out": self._loss})

    @property
    def input_dim(self):
        return sum(f.num * f.dim for f in self.feature_params.values())

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
        self.init_model()
        scores = self.model.predict(features, batch_size=1).reshape((-1,))
        return scores[:self.num_labels]

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: add this many samples with the same features
        """
        super(NeuralNetwork, self).update(features, pred, true, importance)
        for _ in range(int(importance)):
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
        y = None
        for name, values in self._samples.items():
            if name == "out":
                y = np_utils.to_categorical(values, nb_classes=self.max_num_labels)
            else:
                x[name] = np.array(values)
        self.init_model()
        self.model.fit(x, y, batch_size=self._minibatch_size,
                       nb_epoch=self._nb_epochs, verbose=2)
        self._samples = defaultdict(list)
        self._update_index = 0
        self._iteration += 1
        finalized = NeuralNetwork(list(self.labels), model=self.model) if freeze else None
        print("Done (%.3fs)." % (time.time() - started))
        if freeze:
            print("Labels: %d" % self.num_labels)
            print("Features: %d" % sum(f.num * (f.dim or 1) for f in self.feature_params.values()))
        return finalized

    def save(self, filename):
        """
        Save all parameters to file
        :param filename: file to save to
        """
        d = {
            "type": "nn",
            "labels": self.labels,
            "is_frozen": self.is_frozen,
        }
        save_dict(filename, d)
        self.init_model()
        with open(filename + ".json", "w") as f:
            f.write(self.model.to_json())
        try:
            self.model.save_weights(filename + ".h5", overwrite=True)
        except ValueError as e:
            print("Failed saving model weights: %s" % e)

    def load(self, filename):
        """
        Load all parameters from file
        :param filename: file to load from
        """
        d = load_dict(filename)
        model_type = d.get("type")
        assert model_type == "nn", "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.is_frozen = d["is_frozen"]
        with open(filename + ".json") as f:
            self.model = model_from_json(f.read())
        try:
            self.model.load_weights(filename + ".h5")
        except KeyError as e:
            print("Failed loading model weights: %s" % e)
        self.compile()

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)
