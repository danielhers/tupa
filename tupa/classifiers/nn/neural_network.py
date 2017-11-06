import os
import sys
import time
from collections import OrderedDict

import dynet as dy
import numpy as np

from .birnn import BiRNN
from .constants import ACTIVATIONS, INITIALIZERS, TRAINERS, TRAINER_LEARNING_RATE_PARAM_NAMES, TRAINER_KWARGS, \
    CategoricalParameter
from .mlp import MultilayerPerceptron
from ..classifier import Classifier
from ...config import Config, MLP_NN
from ...features.feature_params import MISSING_VALUE


class AxisModel(object):
    """
    Format-specific parameters that are part of the network
    """
    def __init__(self, axis, num_labels, *args, **kwargs):
        config_args = Config().hyperparams.specific[axis]
        self.birnn = BiRNN(config_args, *args, **kwargs)
        self.mlp = MultilayerPerceptron(config_args, *args, num_labels=num_labels, suffix2=(axis,), **kwargs)

    def init_params(self, input_dim, indexed_dim, indexed_num):
        self.mlp.init_params(input_dim + self.birnn.init_params(indexed_dim, indexed_num))


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureEnumerator.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new untrained NN
        """
        super().__init__(*args, **kwargs)
        self.layers = self.args.layers
        self.layer_dim = self.args.layer_dim
        self.output_dim = self.args.output_dim
        self.minibatch_size = self.args.minibatch_size
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.trainer_type = CategoricalParameter(TRAINERS, self.args.optimizer)
        self.dropout = self.args.dropout
        self.params = OrderedDict()  # string (param identifier) -> parameter
        self.empty_values = OrderedDict()  # string (feature suffix) -> expression
        self.axes = OrderedDict()  # string (axis) -> AxisModel
        self.losses = []
        self.indexed_num = self.indexed_dim = self.trainer = self.value = self.birnn = None

    def resize(self):
        for axis, labels in self.labels.items():
            if labels.size is not None:
                num_labels = self.num_labels[axis]
                assert num_labels <= labels.size, "Exceeded maximum number of labels at axis '%s': %d > %d:\n%s" % (
                    axis, num_labels, labels.size, "\n".join(map(str, labels.all)))

    def init_model(self, axis=None, init_params=True):
        init = self.model is None
        if init:
            self.model = dy.ParameterCollection()
            trainer_kwargs = dict(TRAINER_KWARGS.get(str(self.trainer_type), {}))
            learning_rate_param_name = TRAINER_LEARNING_RATE_PARAM_NAMES.get(str(self.trainer_type))
            if learning_rate_param_name and self.learning_rate:
                trainer_kwargs[learning_rate_param_name] = self.learning_rate
            self.trainer = self.trainer_type()(self.model, **trainer_kwargs)
            self.birnn = BiRNN(Config().hyperparams.shared, self.model, self.params, shared=True)
            if init_params:
                self.init_input_params()
        if axis and init_params:
            axis_model = self.axes.get(axis)
            if axis_model is None:
                axis_model = self.axes[axis] = AxisModel(axis, self.labels[axis].size, self.model, self.params)
                axis_model.init_params(self.input_dim, self.indexed_dim, self.indexed_num)
        if init:
            self.init_cg()
            self.finished_step()

    def init_input_params(self):
        """
        Initialize lookup parameters, and calculate the number and dimension of indexed and non-indexed inputs
        """
        self.input_dim = self.indexed_dim = self.indexed_num = 0
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
        self.input_dim += self.birnn.init_params(self.indexed_dim, self.indexed_num)

    def init_cg(self):
        dy.renew_cg()
        for suffix, param in sorted(self.input_params.items()):
            if not param.numeric and param.dim:  # lookup feature
                self.empty_values[suffix] = dy.inputVector(np.zeros(param.dim, dtype=float))

    def init_features(self, features, axes, train=False):
        for axis in axes:
            self.init_model(axis)
        self.birnn.init_features(features, train)
        for axis in axes:
            self.axes[axis].birnn.init_features(features, train)

    def generate_inputs(self, features, axis):
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
        assert self.indexed_num is None or len(indices) == self.indexed_num, \
            "Wrong number of index features: got %d, expected %d" % (len(indices), self.indexed_num)
        if indices:
            yield dy.concatenate(self.birnn.index_input(indices) + self.axes[axis].birnn.index_input(indices))

    def evaluate(self, features, axis, train=False):
        """
        Apply MLP and log softmax to input features
        :param features: dictionary of suffix, values for each feature type
        :param axis: axis of the label we are predicting
        :param train: whether to apply dropout
        :return: expression corresponding to log softmax applied to MLP output
        """
        self.init_model(axis=axis)
        value = self.value.get(axis)
        if value is None:
            self.value[axis] = value = dy.log_softmax(
                self.axes[axis].mlp.evaluate(self.generate_inputs(features, axis), train=train))
        return value

    def score(self, features, axis):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :param axis: axis of the label we are predicting
        :return: array with score for each label
        """
        super().score(features, axis)
        num_labels = self.num_labels[axis]
        if self.updates > 0 and num_labels > 1:
            return self.evaluate(features, axis).npvalue()[:num_labels]
        if self.args.verbose > 3:
            print("  no updates done yet, returning zero vector.")
        return np.zeros(num_labels)

    def update(self, features, axis, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true label (non-negative integer bounded by num_labels[axis])
        :param importance: add this many samples with the same features
        """
        super().update(features, axis, pred, true, importance)
        self.losses.append(importance * dy.pick(self.evaluate(features, axis, train=True), true))
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
        super().finalize()
        assert self.model, "Cannot finalize a model without initializing it first"
        if self.losses:
            loss = -dy.esum(self.losses)
            loss.forward()
            if self.args.verbose > 3:
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
            self.trainer.learning_rate /= (1 - self.learning_rate_decay)
            self.epoch += 1
        if self.args.verbose > 2:
            self.trainer.status()
        return self

    def save_model(self):
        self.finalize()
        d = OrderedDict(
            layers=self.layers,
            layer_dim=self.layer_dim,
            output_dim=self.output_dim,
            activation=str(self.activation),
            init=str(self.init),
            param_keys=list(self.params.keys()),
            axes_param_keys=[list(m.birnn.params.keys()) for m in self.axes.values()],
            axes=list(self.axes),
        )
        if self.model_type != MLP_NN:  # Save BiRNN hyperparams
            d.update(self.birnn.save())
            d.update((a, m.birnn.save()) for a, m in self.axes.items())
        started = time.time()
        try:
            os.remove(self.filename)
            print("Removed existing '%s'." % self.filename)
        except OSError:
            pass
        print("Saving model to '%s'... " % self.filename, end="", flush=True)
        try:
            dy.save(self.filename, [x for m in [self] + list(self.axes.values()) for x in m.birnn.params.values()])
            print("Done (%.3fs)." % (time.time() - started))
        except ValueError as e:
            print("Failed saving model: %s" % e)
        return d

    def load_model(self, d):
        self.model = None
        self.params.clear()
        if self.birnn:
            self.birnn.params.clear()
        for axis_model in self.axes.values():
            axis_model.birnn.params.clear()
        self.args.layers = self.layers = d["layers"]
        self.args.layer_dim = self.layer_dim = d["layer_dim"]
        self.args.output_dim = self.output_dim = d["output_dim"]
        self.args.activation = self.activation.string = d["activation"]
        self.args.init = self.init.string = d["init"]
        axes = d["axes"]
        param_keys, *axes_param_keys = [[tuple(k) if isinstance(k, list) else k for k in keys]  # param key can be tuple
                                        for keys in [d["param_keys"]] + d.get("axes_param_keys", [[]] * len(axes))]
        self.init_model(init_params=False)
        if self.model_type != MLP_NN:  # Load BiRNN hyperparams
            self.birnn.load(d)
        print("Loading model from '%s'... " % self.filename, end="", flush=True)
        started = time.time()
        param_values = dy.load(self.filename, self.model)  # All shared + specific parameter values concatenated
        print("Done (%.3fs)." % (time.time() - started))
        self.params.update(zip(param_keys, param_values))
        del param_values[:len(param_keys)]
        self.axes = OrderedDict()
        for axis, axis_param_keys in zip(axes, axes_param_keys):
            size = self.labels_t[axis][1]
            assert size, "Maximum size of %s labels list is %s" % (axis, size)
            self.axes[axis] = axis_model = AxisModel(axis, size, self.model,
                                                     global_params=self.params,
                                                     params=OrderedDict(zip(axis_param_keys, param_values)))
            del param_values[:len(axis_param_keys)]
            if self.model_type != MLP_NN:
                axis_model.birnn.load(d.get(axis, d))
