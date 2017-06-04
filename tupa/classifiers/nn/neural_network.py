import os
import sys
import time
from collections import OrderedDict

import dynet as dy
import numpy as np

from classifiers.classifier import Classifier
from classifiers.classifier import ClassifierProperty
from features.feature_params import MISSING_VALUE
from tupa.config import Config

TRAINERS = {
    "sgd": dy.SimpleSGDTrainer,
    "cyclic": dy.CyclicalSGDTrainer,
    "momentum": dy.MomentumSGDTrainer,
    "adagrad": dy.AdagradTrainer,
    "adadelta": dy.AdadeltaTrainer,
    "rmsprop": dy.RMSPropTrainer,
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

    def __init__(self, *args, max_num_labels):
        """
        Create a new untrained NN
        :param labels: tuple of lists of labels that can be updated later to add new labels
        """
        super(NeuralNetwork, self).__init__(*args)
        self.max_num_labels = tuple(max_num_labels)
        self.layers = Config().args.layers
        self.layer_dim = Config().args.layer_dim
        self.output_dim = Config().args.output_dim
        self.activation_str = Config().args.activation
        self.init_str = Config().args.init
        self.minibatch_size = Config().args.minibatch_size
        self.dropout = Config().args.dropout
        self.optimizer_str = Config().args.optimizer
        self.activation = ACTIVATIONS[self.activation_str]
        self.init = INITIALIZERS[self.init_str]
        self.optimizer = TRAINERS[self.optimizer_str]
        self.params = OrderedDict()
        self.empty_values = OrderedDict()
        self.indexed_num = None
        self.indexed_dim = None
        self.losses = []
        self.trainer = None
        self.value = [None] * len(self.num_labels)  # For caching the result of _evaluate

    def resize(self, axis=None):
        for i, (l, m) in enumerate(zip(self.num_labels, self.max_num_labels)):
            if axis is None or i == axis:
                assert l <= m, "Exceeded maximum number of labels at dimension %d: %d > %d" % (i, l, m)

    def init_model(self):
        self.model = dy.Model()
        self.trainer = self.optimizer(self.model, )
        self.init_input_params()
        self.init_mlp_params()
        self.init_cg()

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
                    if param.init is not None:
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

    def init_mlp_params(self):
        for axis in range(len(self.num_labels)):
            in_dim = [self.input_dim] + (self.layers - 1) * [self.layer_dim] + [self.output_dim]
            out_dim = (self.layers - 1) * [self.layer_dim] + [self.output_dim, self.max_num_labels[axis]]
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
            param = self.input_params[suffix]
            if param.numeric:
                yield dy.inputVector(values)
            elif param.dim:
                if param.indexed:  # collect indices to be looked up
                    indices += values  # FeatureIndexer collapsed the features so there are no repetitions between them
                else:
                    yield dy.concatenate([self.empty_values[suffix] if x == MISSING_VALUE else self.params[suffix][x]
                                          for x in values])
        if indices:
            assert len(indices) == self.indexed_num, "Wrong number of index features: %d != %d" % (
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
        return dy.log_softmax(x, restrict=list(range(self.num_labels[axis])))

    def evaluate(self, features, axis, train=False):
        if self.model is None:
            self.init_model()
        if self.value[axis] is None:
            self.value[axis] = self.evaluate_mlp(features=features, axis=axis, train=train)
        return self.value[axis]

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
            if Config().args.verbose > 2:
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
            if Config().args.dynet_viz:
                dy.print_graphviz()
                sys.exit(0)

    def finished_step(self, train=False):
        self.value = [None] * len(self.num_labels)

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
        if self.model is None:
            self.init_model()
        if self.losses:
            loss = -dy.esum(self.losses)
            loss.forward()
            if Config().args.verbose > 2:
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
        if Config().args.verbose > 1:
            self.trainer.status()
        return self

    def save_labels(self):
        node_labels = self.input_params.get("n")  # Do not save node labels as they are saved as features already
        return (self.labels[0], []) if node_labels is not None and node_labels.size else self.labels

    def save_model(self):
        self.finalize()
        d = {
            "param_keys": list(self.params.keys()),
            "max_num_labels": self.max_num_labels,
            "layers": self.layers,
            "layer_dim": self.layer_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_str,
            "init": self.init_str,
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
            self.model.save(model_filename, self.params.values())
            print("Done (%.3fs)." % (time.time() - started))
        except ValueError as e:
            print("Failed saving model: %s" % e)
        return d

    def load_model(self, d):
        self.init_model()
        param_keys = d["param_keys"]
        self.max_num_labels = d["max_num_labels"]
        Config().args.layers = self.layers = d["layers"]
        Config().args.layer_dim = self.layer_dim = d["layer_dim"]
        Config().args.output_dim = self.output_dim = d.get("output_dim", Config().args.output_dim)
        Config().args.activation = self.activation_str = d["activation"]
        self.activation = ACTIVATIONS[self.activation_str]
        Config().args.init = self.init_str = d["init"]
        self.init = INITIALIZERS[self.init_str]
        self.load_extra(d)
        model_filename = self.filename + ".model"
        print("Loading model from '%s'... " % model_filename, end="", flush=True)
        started = time.time()
        try:
            param_values = self.model.load(model_filename)
            print("Done (%.3fs)." % (time.time() - started))
            self.params = OrderedDict(zip(param_keys, param_values))
        except KeyError as e:
            print("Failed loading model: %s" % e)

    def get_classifier_properties(self):
        return super(NeuralNetwork, self).get_classifier_properties() + \
               (ClassifierProperty.trainable_after_saving,)
