import sys
import time

import numpy as np
import os
from collections import OrderedDict

import dynet as dy
from classifiers.classifier import Classifier
from features.feature_params import MISSING_VALUE
from parsing.config import Config

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

    def __init__(self, *args, input_params):
        """
        Create a new untrained NN
        :param labels: a list of labels that can be updated later to add a new label
        :param input_params: dict of feature type name -> FeatureInformation
        """
        super(NeuralNetwork, self).__init__(*args)
        self.max_num_labels = Config().args.maxlabels
        self._layers = Config().args.layers
        self._layer_dim = Config().args.layerdim
        self._activation_str = Config().args.activation
        self._init_str = Config().args.init
        self._minibatch_size = Config().args.minibatchsize
        self._dropout = Config().args.dropout
        self._optimizer_str = Config().args.optimizer
        self._activation = ACTIVATIONS[self._activation_str]
        self._init = INITIALIZERS[self._init_str]
        self._optimizer = TRAINERS[self._optimizer_str]
        self._num_labels = self.num_labels
        self._params = OrderedDict()
        self._empty_values = OrderedDict()
        self._input_params = input_params
        self._indexed_num = None
        self._indexed_dim = None
        self._losses = []
        self._iteration = 0
        self._trainer = None

    @property
    def input_dim(self):
        return sum(f.num * f.dim for f in self._input_params.values())

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def evaluate(self, *args, **kwargs):
        self.init_cg()
        return self.evaluate_mlp(*args, **kwargs)

    def init_model(self):
        self.model = dy.Model()
        self._trainer = self._optimizer(self.model, )
        input_dim = self.init_input_params()
        self.init_mlp_params(input_dim)

    def init_input_params(self):
        """
        Initialize lookup parameters and any other parameters that process the input (e.g. LSTMs)
        :return: total output dimension of inputs
        """
        input_dim = 0
        self._indexed_dim = 0
        self._indexed_num = 0
        for suffix, param in sorted(self._input_params.items()):
            if not param.numeric and param.dim > 0:  # lookup feature
                p = self.model.add_lookup_parameters((param.size, param.dim))
                if param.init is not None:
                    p.init_from_array(param.init)
                self._params[suffix] = p
            if param.indexed:
                self._indexed_dim += param.dim
                self._indexed_num = max(self._indexed_num, param.num)  # indices to be looked up are collected
            else:
                input_dim += param.num * param.dim
        return input_dim + self.init_indexed_input_params()

    def init_indexed_input_params(self):
        """
        :return: total output dimension of indexed features
        """
        return self._indexed_dim * self._indexed_num

    def init_mlp_params(self, input_dim):
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
        for suffix, param in sorted(self._input_params.items()):
            if not param.numeric and param.dim > 0:  # lookup feature
                self._empty_values[suffix] = self.empty_rep(param.dim)

    @staticmethod
    def empty_rep(dim):
        """
        Representation for missing elements
        :param dim: dimension of vector to return
        :return: zero vector (an alternative could be to learn this value, as in e.g. Kiperwasser and Goldberg 2016)
        """
        return dy.inputVector(np.zeros(dim, dtype=float))

    def generate_inputs(self, features):
        indices = []  # list, not set, in order to maintain consistent order
        for suffix, values in sorted(features.items()):
            param = self._input_params[suffix]
            if param.numeric:
                yield dy.inputVector(values)
            elif param.dim > 0:
                if param.indexed:  # collect indices to be looked up
                    indices += values  # FeatureIndexer collapsed the features so there are no repetitions between them
                else:
                    yield dy.concatenate([self._empty_values[suffix] if x == MISSING_VALUE else self._params[suffix][x]
                                          for x in values])
        if indices:
            assert len(indices) == self._indexed_num, "Wrong number of index features: %d != %d" % (
                len(indices), self._indexed_num)
            yield self.index_input(indices)

    def index_input(self, indices):
        """
        :param indices: indices of inputs
        :return: feature values at given indices
        """
        raise Exception("Input representations not initialized, cannot evaluate indexed features")

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
            if Config().args.dynet_viz:
                dy.print_graphviz()
                sys.exit(0)

    def finalize(self, finished_epoch=False):
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
        if finished_epoch:
            self._trainer.update_epoch()
        if Config().args.verbose:
            self._trainer.status()
        return self

    def save_model(self):
        self.finalize()
        param_keys, param_values = zip(*self._params.items())
        d = {
            "input_params": self._input_params,
            "param_keys": param_keys,
            "layers": self._layers,
            "layer_dim": self._layer_dim,
            "activation": self._activation_str,
            "init": self._init_str,
            "optimizer": self._optimizer_str,
        }
        d.update(self.save_extra())
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
        return d

    def load_model(self, d):
        self.init_model()
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
            self._params = OrderedDict(zip(param_keys, param_values))
        except KeyError as e:
            print("Failed loading model: %s" % e)

    def get_classifier_properties(self):
        return ()
