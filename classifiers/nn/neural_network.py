import os
import time

import sys
from collections import OrderedDict

import dynet as dy
from classifiers.classifier import Classifier
from parsing.model_util import load_dict, save_dict

TRAINERS = {
    "sgd": dy.SimpleSGDTrainer,
    "momentum": dy.MomentumSGDTrainer,
    "adagrad": dy.AdagradTrainer,
    "adadelta": dy.AdadeltaTrainer,
    "adam": dy.AdamTrainer,
}

INITIALIZERS = {
    "glorot_uniform": dy.GlorotInitializer(),
    "normal": dy.NormalInitializer(),
    "uniform": dy.UniformInitializer(1),
    "const": dy.ConstInitializer(0),
}

ACTIVATIONS = {
    "square": dy.square,
    "cube": dy.cube,
    "tanh": dy.tanh,
    "sigmoid": dy.logistic,
    "relu": dy.rectify,
}


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureEnumerator.
    """

    def __init__(self, filename, labels, model_type, input_params=None,
                 layers=1, layer_dim=100, activation="tanh",
                 init="glorot_uniform", max_num_labels=100,
                 minibatch_size=200, dropout=0, optimizer="adam"):
        """
        Create a new untrained NN or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param input_params: dict of feature type name -> FeatureInformation
        :param layers: number of hidden layers
        :param layer_dim: size of hidden layer
        :param activation: activation function at hidden layers
        :param init: initialization type for hidden layers
        :param max_num_labels: since model size is fixed, set maximum output size
        :param minibatch_size: batch size for training
        :param dropout: dropout to apply to each layer
        :param optimizer: algorithm to use for optimization
        """
        super(NeuralNetwork, self).__init__(model_type=model_type, filename=filename, labels=labels)
        assert input_params is not None
        self.max_num_labels = max_num_labels
        self._layers = layers
        self._layer_dim = layer_dim
        self._activation_str = activation
        self._activation = ACTIVATIONS[self._activation_str]
        self._init_str = init
        self._init = INITIALIZERS[self._init_str]
        self._num_labels = self.num_labels
        self._minibatch_size = minibatch_size
        self._dropout = dropout
        self._optimizer_str = optimizer
        self._optimizer = TRAINERS[self._optimizer_str]
        self._params = OrderedDict()
        self._input_params = input_params
        self._losses = []
        self._iteration = 0
        self.model = None
        self._trainer = None

    @property
    def input_dim(self):
        return sum(f.num * f.dim for f in self._input_params.values())

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def init_model(self):
        raise NotImplementedError()

    def save(self):
        """
        Save all parameters to file
        """
        self.finalize()
        param_keys, param_values = zip(*self._params.items())
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "input_params": self._input_params,
            "param_keys": param_keys,
            "layers": self._layers,
            "layer_dim": self._layer_dim,
            "activation": self._activation_str,
            "init": self._init_str,
            "optimizer": self._optimizer_str,
        }
        save_dict(self.filename, d)
        model_filename = self.filename + ".model"
        started = time.time()
        try:
            os.remove(model_filename)
            print("Removed existing '%s'." % model_filename)
        except OSError:
            pass
        print("Saving model to '%s'... " % model_filename, end="", flush=True)
        try:
            self.model.save(model_filename, param_values)
            print("Done (%.3fs)." % (time.time() - started))
        except ValueError as e:
            print("Failed saving model: %s" % e)

    def load(self):
        """
        Load all parameters from file
        :param suffix: extra suffix to append to filename
        """
        self.init_model()
        d = load_dict(self.filename)
        model_type = d.get("type")
        assert model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self._input_params = d["input_params"]
        param_keys = d["param_keys"]
        self._layers = d["layers"]
        self._layer_dim = d["layer_dim"]
        self._activation_str = d["activation"]
        self._activation = ACTIVATIONS[self._activation_str]
        self._init_str = d["init"]
        self._init = INITIALIZERS[self._init_str]
        self._optimizer_str = d["optimizer"]
        self._optimizer = TRAINERS[self._optimizer_str]
        model_filename = self.filename + ".model"
        print("Loading model from '%s'... " % model_filename, end="", flush=True)
        started = time.time()
        try:
            param_values = self.model.load(model_filename)
            print("Done (%.3fs)." % (time.time() - started))
        except KeyError as e:
            print("Failed loading model: %s" % e)
        self._params = OrderedDict(zip(param_keys, param_values))

    def finalize(self):
        """
        Fit this model on collected samples
        :return self
        """
        super(NeuralNetwork, self).finalize()
        if self._losses:
            loss = -dy.esum(self._losses)
            loss.forward()
            loss.backward()
            self._trainer.update()
            self._losses = []
            self._iteration += 1
        return self

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)
