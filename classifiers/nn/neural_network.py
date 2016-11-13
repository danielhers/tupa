import sys
import time

import numpy as np
import os
from collections import OrderedDict

import dynet as dy
from classifiers.classifier import Classifier
from parsing import config
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

    def evaluate(self, features):
        raise NotImplementedError

    def generate_inputs(self, features):
        raise NotImplementedError

    def init_model(self):
        self.model = dy.Model()
        self._trainer = self._optimizer(self.model)
        input_dim = self.init_inputs()
        self.init_mlp(input_dim)

    def init_inputs(self):
        input_dim = 0
        for suffix, param in sorted(self._input_params.items()):
            if not param.numeric and param.dim > 0:  # index feature
                p = self.model.add_lookup_parameters((param.size, param.dim))
                if param.init is not None:
                    p.init_from_array(param.init)
                self._params[suffix] = p
            input_dim += self.init_extra_inputs(suffix, param)
        return input_dim

    def init_extra_inputs(self, suffix, param):
        return param.num * param.dim

    def init_mlp(self, input_dim):
        for i in range(1, self._layers + 1):
            in_dim = input_dim if i == 1 else self._layer_dim
            out_dim = self._layer_dim if i < self._layers else self.max_num_labels
            self._params["W%d" % i] = self.model.add_parameters((out_dim, in_dim), init=self._init)
            self._params["b%d" % i] = self.model.add_parameters(out_dim, init=self._init)

    def init_cg(self):
        if self.model is None:
            self.init_model()
        if not self._losses:
            dy.renew_cg()

    def generate_inputs(self, features):
        for suffix, values in sorted(features.items()):
            param = self._input_params[suffix]
            if param.numeric:
                yield dy.inputVector(values)
            elif param.dim > 0:
                v = self.index_input(suffix, param, values)
                yield dy.reshape(self._params[suffix].batch(values),
                                 (param.num * param.dim,)) if v is None else v

    def index_input(self, suffix, param, values):
        pass

    def evaluate_mlp(self, features, train=False):
        x = dy.concatenate(list(self.generate_inputs(features)))
        for i in range(1, self._layers + 1):
            W = dy.parameter(self._params["W%d" % i])
            b = dy.parameter(self._params["b%d" % i])
            if train and self._dropout:
                x = dy.dropout(x, self._dropout)
            x = self._activation(W * x + b)
        return dy.log_softmax(x, restrict=list(range(self.num_labels)))

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :return: array with score for each label
        """
        super(NeuralNetwork, self).score(features)
        return self.evaluate(features).npvalue()[:self.num_labels] if self._iteration > 0 else np.zeros(self.num_labels)

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
            self._losses.append(dy.pick(self.evaluate(features, train=True), true))
            if len(self._losses) >= self._minibatch_size:
                self.finalize()
            if config.Config().args.dynet_viz:
                dy.print_graphviz()
                sys.exit(0)

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
        d.update(self.save_extra())
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

    def save_extra(self):
        return {}

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
        self.load_extra(d)
        model_filename = self.filename + ".model"
        print("Loading model from '%s'... " % model_filename, end="", flush=True)
        started = time.time()
        try:
            param_values = self.model.load(model_filename)
            print("Done (%.3fs)." % (time.time() - started))
        except KeyError as e:
            print("Failed loading model: %s" % e)
        self._params = OrderedDict(zip(param_keys, param_values))

    def load_extra(self, d):
        pass

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)
