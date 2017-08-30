import sys
import time

import dynet as dy
import numpy as np
import os
from collections import OrderedDict
from functools import partial

from tupa.classifiers.classifier import Classifier
from tupa.config import Config
from tupa.features.feature_params import MISSING_VALUE

TRAINERS = {
    "sgd": (dy.SimpleSGDTrainer, "e0"),
    "cyclic": (dy.CyclicalSGDTrainer, "e0_min"),
    "momentum": (dy.MomentumSGDTrainer, "e0"),
    "adagrad": (dy.AdagradTrainer, "e0"),
    "adadelta": (dy.AdadeltaTrainer, None),
    "rmsprop": (dy.RMSPropTrainer, "e0"),
    "adam": (partial(dy.AdamTrainer, beta_2=0.9), "alpha"),
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

    def __init__(self, *args):
        """
        Create a new untrained NN
        """
        super(NeuralNetwork, self).__init__(*args)
        self.layers = self.args.layers
        self.layer_dim = self.args.layer_dim
        self.output_dim = self.args.output_dim
        self.activation_str = self.args.activation
        self.init_str = self.args.init
        self.minibatch_size = self.args.minibatch_size
        self.dropout = self.args.dropout
        self.trainer_str = self.args.optimizer
        self.activation = ACTIVATIONS[self.activation_str]
        self.init = INITIALIZERS[self.init_str]
        self.trainer_type, self.learning_rate_param_name = TRAINERS[self.trainer_str]
        self.params = OrderedDict()
        self.empty_values = OrderedDict()
        self.losses = []
        self.indexed_num = self.indexed_dim = self.trainer = self.value = None
        self.axes = set()

    def resize(self, axis=None):
        for axis_, labels in self.labels.items():
            if axis in (axis_, None):  # None means all
                num_labels = self.num_labels[axis_]
                assert num_labels <= labels.size, "Exceeded maximum number of labels at axis '%s': %d > %d:\n%s" % (
                    axis_, num_labels, labels.size, "\n".join(map(str, labels.all)))

    def init_model(self, axis=None):
        init = self.model is None
        if init:
            self.model = dy.ParameterCollection()
            trainer_kwargs = {"edecay": self.learning_rate_decay}
            if self.learning_rate_param_name and self.learning_rate:
                trainer_kwargs[self.learning_rate_param_name] = self.learning_rate
            self.trainer = self.trainer_type(self.model, **trainer_kwargs)
            self.init_input_params()
        if axis and axis not in self.axes:
            self.axes.add(axis)
            self.init_mlp_params(axis)
        if init:
            self.init_cg()
            self.finished_step()

    def init_input_params(self):
        """
        Initialize lookup parameters and any other parameters that process the input (e.g. LSTMs)
        :return: total output dimension of inputs
        """
        self.input_dim = 0
        self.indexed_dim = 0
        self.indexed_num = 0
        for suffix, param in sorted(self.input_params.items()):
            if param.dim:
                if not param.numeric:  # lookup feature
                    p = self.model.add_lookup_parameters((param.size, param.dim))
                    p.set_updated(param.updated)
                    if param.init is not None and param.init.size:
                        p.init_from_array(param.init)
                    self.params[suffix] = p
                if param.indexed:
                    self.indexed_dim += param.dim  # add to the input dimensionality at each indexed time point
                    self.indexed_num = max(self.indexed_num, param.num)  # indices to be looked up are collected
                else:
                    self.input_dim += param.num * param.dim
        self.input_dim += self.init_indexed_input_params()

    def init_indexed_input_params(self):
        """
        :return: total output dimension of indexed features
        """
        return self.indexed_dim * self.indexed_num

    def init_mlp_params(self, axis):
        in_dim = [self.input_dim] + (self.layers - 1) * [self.layer_dim] + [self.output_dim]
        out_dim = (self.layers - 1) * [self.layer_dim] + [self.output_dim, self.labels[axis].size]
        for i in range(self.layers + 1):
            self.params[("W", i, axis)] = self.model.add_parameters((out_dim[i], in_dim[i]), init=self.init)
            self.params[("b", i, axis)] = self.model.add_parameters(out_dim[i], init=self.init)

    def init_cg(self):
        dy.renew_cg()
        for suffix, param in sorted(self.input_params.items()):
            if not param.numeric and param.dim:  # lookup feature
                self.empty_values[suffix] = self.zero_input(param.dim)

    @staticmethod
    def zero_input(dim):
        """
        Representation for missing elements
        :param dim: dimension of vector to return
        :return: zero vector (as in e.g. Kiperwasser and Goldberg 2016; an alternative could be to learn this value)
        """
        return dy.inputVector(np.zeros(dim, dtype=float))

    def generate_inputs(self, features):
        indices = []  # list, not set, in order to maintain consistent order
        for suffix, values in sorted(features.items()):
            param = self.input_params.get(suffix)
            if param is None:
                pass  # feature missing from model, so just ignore it
            elif param.numeric:
                yield dy.inputVector(values)
            elif param.dim:
                if param.indexed:  # collect indices to be looked up
                    indices += values  # FeatureIndexer collapsed the features so there are no repetitions between them
                else:
                    yield dy.concatenate([self.empty_values[suffix] if x == MISSING_VALUE else self.params[suffix][x]
                                          for x in values])
        if indices:
            assert len(indices) == self.indexed_num, "Wrong number of index features: got %d, expected %d" % (
                len(indices), self.indexed_num)
            yield self.index_input(indices)

    def index_input(self, indices):
        """
        :param indices: indices of inputs
        :return: feature values at given indices
        """
        raise ValueError("Input representations not initialized, cannot evaluate indexed features")

    def evaluate_mlp(self, features, axis, train=False):
        """
        Apply MLP and log softmax to input features
        :param features: dictionary of suffix, values for each feature type
        :param axis: axis of the label we are predicting
        :param train: whether to apply dropout
        :return: expression corresponding to log softmax applied to MLP output
        """
        x = dy.concatenate(list(self.generate_inputs(features)))
        for i in range(self.layers + 1):
            W = dy.parameter(self.params[("W", i, axis)])
            b = dy.parameter(self.params[("b", i, axis)])
            if train and self.dropout:
                x = dy.dropout(x, self.dropout)
            x = self.activation(W * x + b)
        return dy.log_softmax(x, restrict=None if "--dynet-gpu" in sys.argv else list(range(self.num_labels[axis])))

    def evaluate(self, features, axis, train=False):
        self.init_model(axis=axis)
        value = self.value.get(axis)
        if value is None:
            value = self.evaluate_mlp(features=features, axis=axis, train=train)
            self.value[axis] = value
        return value

    def score(self, features, axis):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :param axis: axis of the label we are predicting
        :return: array with score for each label
        """
        super(NeuralNetwork, self).score(features, axis)
        if self.updates > 0:
            return self.evaluate(features, axis).npvalue()[:self.num_labels[axis]]
        else:
            if self.args.verbose > 2:
                print("  no updates done yet, returning zero vector.")
            return np.zeros(self.num_labels[axis])

    def update(self, features, axis, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true label (non-negative integer bounded by num_labels[axis])
        :param importance: add this many samples with the same features
        """
        super(NeuralNetwork, self).update(features, axis, pred, true, importance)
        for _ in range(int(importance)):
            self.losses.append(dy.pick(self.evaluate(features, axis, train=True), true))
            if self.args.dynet_viz:
                dy.print_graphviz()
                sys.exit(0)

    def finished_step(self, train=False):
        self.value = {}  # For caching the result of _evaluate

    def finished_item(self, train=False):
        if len(self.losses) >= self.minibatch_size:
            self.finalize()
        elif not train:
            self.init_cg()
        self.finished_step(train)

    def finalize(self, finished_epoch=False):
        """
        Fit this model on collected samples
        :return self
        """
        super(NeuralNetwork, self).finalize()
        assert self.model, "Cannot finalize a model without initializing it first"
        if self.losses:
            loss = -dy.esum(self.losses)
            loss.forward()
            if self.args.verbose > 2:
                print("Total loss from %d time steps: %g" % (len(self.losses), loss.value()))
            loss.backward()
            # if np.linalg.norm(loss.gradient()) not in (np.inf, np.nan):
            try:
                self.trainer.update()
            except RuntimeError as e:
                Config().log("Error in update(): %s\n" % e)
            self.init_cg()
            self.losses = []
            self.updates += 1
        if finished_epoch:
            self.trainer.update_epoch()
            self.epoch += 1
        if self.args.verbose > 1:
            self.trainer.status()
        return self

    def save_labels(self):
        node_labels = self.input_params.get("n")  # Do not save node labels as they are saved as features already
        omit_node_labels = node_labels is not None and node_labels.size
        return {a: ([], 0) if a == "n" and omit_node_labels else l.save() for a, l in self.labels.items()}

    def save_model(self):
        self.finalize()
        d = {
            "param_keys": list(self.params.keys()),
            "axes": list(self.axes),
            "layers": self.layers,
            "layer_dim": self.layer_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_str,
            "init": self.init_str,
        }
        d.update(self.save_extra())
        started = time.time()
        try:
            os.remove(self.filename)
            print("Removed existing '%s'." % self.filename)
        except OSError:
            pass
        print("Saving model to '%s'... " % self.filename, end="", flush=True)
        try:
            dy.save(self.filename, self.params.values())
            print("Done (%.3fs)." % (time.time() - started))
        except ValueError as e:
            print("Failed saving model: %s" % e)
        return d

    def load_model(self, d):
        param_keys = [tuple(k) if isinstance(k, list) else k for k in d["param_keys"]]
        self.axes = set(d["axes"])
        self.args.layers = self.layers = d["layers"]
        self.args.layer_dim = self.layer_dim = d["layer_dim"]
        self.args.output_dim = self.output_dim = d["output_dim"]
        self.args.activation = self.activation_str = d["activation"]
        self.activation = ACTIVATIONS[self.activation_str]
        self.args.init = self.init_str = d["init"]
        self.init = INITIALIZERS[self.init_str]
        self.load_extra(d)
        for axis in self.axes:
            self.init_model(axis)
        print("Loading model from '%s'... " % self.filename, end="", flush=True)
        started = time.time()
        try:
            param_values = dy.load(self.filename, self.model)
            print("Done (%.3fs)." % (time.time() - started))
            self.params = OrderedDict(zip(param_keys, param_values))
        except KeyError as e:
            print("Failed loading model: %s" % e)
