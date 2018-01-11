import os
import sys
import time
from collections import OrderedDict
from itertools import repeat

import dynet as dy
import numpy as np
from tqdm import tqdm

from .birnn import BiRNN
from .constants import TRAINERS, TRAINER_LEARNING_RATE_PARAM_NAMES, TRAINER_KWARGS, CategoricalParameter
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from ..classifier import Classifier
from ...config import Config, BIRNN
from ...model_util import MISSING_VALUE


class AxisModel:
    """
    Format-specific parameters that are part of the network
    """
    def __init__(self, axis, num_labels, model, with_birnn=True):
        args = Config().hyperparams.specific[axis]
        self.birnn = BiRNN(args, model, save_path=("axes", axis, "birnn"), with_birnn=with_birnn)
        self.mlp = MultilayerPerceptron(args, model, num_labels=num_labels, save_path=("axes", axis, "mlp"))


class NeuralNetwork(Classifier, SubModel):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from DenseFeatureExtractor.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new untrained NN
        """
        Classifier.__init__(self, *args, **kwargs)
        SubModel.__init__(self)
        self.minibatch_size = self.args.minibatch_size
        self.loss = self.args.loss
        self.empty_values = OrderedDict()  # string (feature suffix) -> expression
        self.axes = OrderedDict()  # string (axis) -> AxisModel
        self.losses = []
        self.steps = 0
        self.trainer_type = self.trainer = self.value = self.birnn = None

    @property
    def input_dim(self):
        return {a: m.mlp.input_dim for a, m in self.axes.items()}

    def resize(self):
        for axis, labels in self.labels.items():
            if labels.size is not None:
                num_labels = self.num_labels[axis]
                assert num_labels <= labels.size, "Exceeded maximum number of labels at axis '%s': %d > %d:\n%s" % (
                    axis, num_labels, labels.size, "\n".join(map(str, labels.all)))

    def init_model(self, axis=None):
        init = self.model is None
        if init:
            self.model = dy.ParameterCollection()
            self.birnn = BiRNN(Config().hyperparams.shared, self.model,
                               save_path=("shared", "birnn"), with_birnn=self.model_type == BIRNN)
        self.init_trainer()
        if axis:
            self.init_axis_model(axis)
        if init:
            self.init_cg()
            self.finished_step()
        self.init_empty_values()

    def init_trainer(self):
        if self.trainer_type is None or str(self.trainer_type) != self.args.optimizer:
            self.trainer_type = CategoricalParameter(TRAINERS, self.args.optimizer)
            trainer_kwargs = dict(TRAINER_KWARGS.get(str(self.trainer_type), {}))
            learning_rate_param_name = TRAINER_LEARNING_RATE_PARAM_NAMES.get(str(self.trainer_type))
            if learning_rate_param_name and self.learning_rate:
                trainer_kwargs[learning_rate_param_name] = self.learning_rate
            if self.args.verbose > 3:
                print("Initializing trainer=%s(%s)" % (
                    self.trainer_type, ", ".join("%s=%s" % (k, v) for k, v in trainer_kwargs.items())))
            self.trainer = self.trainer_type()(self.model, **trainer_kwargs)

    def init_axis_model(self, axis):
        model = self.axes.get(axis)
        if model:
            return
        model = self.axes[axis] = AxisModel(axis, self.labels[axis].size, self.model,
                                            with_birnn=self.model_type == BIRNN)
        if self.args.verbose > 3:
            print("Initializing %s model with %d labels" % (axis, self.labels[axis].size))
        input_dim = indexed_dim = indexed_num = 0
        for suffix, param in sorted(self.input_params.items()):
            if not param.enabled:
                continue
            if self.args.verbose > 3:
                print("Initializing input parameter: %s" % param)
            if not param.numeric and suffix not in self.params:  # lookup feature
                lookup = self.model.add_lookup_parameters((param.size, param.dim))
                lookup.set_updated(param.updated)
                param.init_data()
                if param.init is not None and param.init.size:
                    lookup.init_from_array(param.init)
                self.params[suffix] = lookup
            if param.indexed:
                indexed_dim += param.dim  # add to the input dimensionality at each indexed time point
                indexed_num = max(indexed_num, param.num)  # indices to be looked up are collected
            else:
                input_dim += param.num * param.dim
        for birnn in self.get_birnns(axis):
            input_dim += birnn.init_params(indexed_dim, indexed_num)
        model.mlp.init_params(input_dim)

    def init_cg(self):
        dy.renew_cg()
        self.init_empty_values(clear=True)

    def init_empty_values(self, clear=False):
        if clear:
            self.empty_values.clear()
        for suffix, param in self.input_params.items():
            if param.enabled and not param.numeric and suffix not in self.empty_values:  # lookup feature
                self.empty_values[suffix] = dy.inputVector(np.zeros(param.dim, dtype=float))

    def init_features(self, features, axes, train=False):
        for axis in axes:
            self.init_model(axis)
        embeddings = [[self.params[s][k] for k in ks] for s, ks in sorted(features.items())]  # lists of vectors
        if self.args.verbose > 3:
            print("Initializing %s BiRNN features for %d elements" % (", ".join(axes), len(embeddings)))
        self.birnn.init_features(embeddings, train)
        for axis in axes:
            self.axes[axis].birnn.init_features(embeddings, train)

    def generate_inputs(self, features, axis):
        indices = []  # list, not set, in order to maintain consistent order
        for suffix, values in sorted(features.items()):
            param = self.input_params[suffix]
            if param.numeric:
                yield dy.inputVector(values)
            elif param.indexed:  # collect indices to be looked up
                indices += values  # DenseFeatureExtractor collapsed features so there are no repetitions between them
            else:  # lookup feature
                yield dy.concatenate([self.empty_values[suffix] if x == MISSING_VALUE else self.params[suffix][x]
                                      for x in values])
        if indices:
            values = []
            for birnn in self.get_birnns(axis):
                values += birnn.evaluate(indices)
            yield dy.concatenate(values)

    def get_birnns(self, axis):
        """ Return shared + axis-specific BiRNNs """
        return [m.birnn for m in (self, self.axes[axis])]

    def evaluate(self, features, axis, train=False):
        """
        Apply MLP and log softmax to input features
        :param features: dictionary of suffix, values for each feature type
        :param axis: axis of the label we are predicting
        :param train: whether to apply dropout
        :return: expression corresponding to log softmax applied to MLP output
        """
        self.init_model(axis)
        value = self.value.get(axis)
        if value is None:
            self.value[axis] = value = self.axes[axis].mlp.evaluate(self.generate_inputs(features, axis), train=train)
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
            value = dy.log_softmax(self.evaluate(features, axis), restrict=list(range(num_labels))).npvalue()
            return value[:num_labels]
        if self.args.verbose > 3:
            print("  no updates done yet, returning zero vector.")
        return np.zeros(num_labels)

    def update(self, features, axis, pred, true, importance=None):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, in the form of a dict (name: value)
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true labels (non-negative integers bounded by num_labels[axis])
        :param importance: how much to scale the update for the weight update for each true label
        """
        super().update(features, axis, pred, true, importance)
        losses = self.calc_loss(self.evaluate(features, axis, train=True), axis, true, importance or repeat(1))
        if self.args.verbose > 3:
            print("  loss=" + ", ".join("%g" % l.value() for l in losses))
        self.losses += losses
        self.steps += 1

    def calc_loss(self, scores, axis, true, importance):
        if self.loss == "softmax":
            return [i * dy.pickneglogsoftmax(scores, t) for t, i in zip(true, importance)]
        elif self.loss == "max_margin":
            max_true = dy.emax([i * dy.pick(scores, t) for t, i in zip(true, importance)])
            max_false = dy.emax([dy.pick(scores, t) for t in range(self.num_labels[axis]) if t not in true])
            return [dy.rectify(1 - max_true + max_false)]
        raise NotImplementedError("%s loss is not supported" % self.loss)

    def finished_step(self, train=False):
        self.value = {}  # For caching the result of _evaluate

    def finished_item(self, train=False):
        if self.steps >= self.minibatch_size:
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
            loss = dy.esum(self.losses)
            loss.forward()
            if self.args.verbose > 3:
                print("Total loss from %d time steps: %g" % (self.steps, loss.value()))
            loss.backward()
            try:
                self.trainer.update()
            except RuntimeError as e:
                Config().log("Error in update(): %s\n" % e)
            self.init_cg()
            self.losses = []
            self.steps = 0
            self.updates += 1
        if finished_epoch:
            self.trainer.learning_rate /= (1 - self.learning_rate_decay)
            self.epoch += 1
        if self.args.verbose > 2:
            self.trainer.status()
        return self
            
    def sub_models(self):
        """ :return: ordered list of SubModels """
        axes = [self.axes[a] for a in self.labels or self.labels_t]
        return [self] + [m.mlp for m in axes] + [m.birnn for m in axes + [self]]
    
    def save_sub_model(self, d, *args):
        return SubModel.save_sub_model(
            self, d,
            ("loss", self.loss),
        )

    def load_sub_model(self, d, *args):
        d = SubModel.load_sub_model(self, d, *args)
        self.args.loss = self.loss = d["loss"]

    def save_model(self, filename, d):
        Classifier.save_model(self, filename, d)
        self.finalize()
        values = []
        for model in self.sub_models():
            values += model.save_sub_model(d)
            if self.args.verbose > 1:
                print(model.params_str())
        if self.args.verbose:
            print(self)
        try:
            os.remove(filename)
            print("Removed existing '%s'." % filename)
        except OSError:
            pass
        try:
            dy.save(filename, tqdm(values, desc="Saving model to '%s'" % filename, unit="param", file=sys.stdout))
        except ValueError as e:
            print("Failed saving model: %s" % e)

    def load_model(self, filename, d):
        self.model = None
        self.init_model()
        print("Loading model from '%s'... " % filename, end="", flush=True)
        started = time.time()
        values = dy.load(filename, self.model)  # All sub-model parameter values, concatenated
        print("Done (%.3fs)." % (time.time() - started))
        self.axes = OrderedDict()
        for axis, labels in self.labels_t.items():
            _, size = labels
            assert size, "Size limit for '%s' axis labels is %s" % (axis, size)
            self.axes[axis] = AxisModel(axis, size, self.model, with_birnn=self.model_type == BIRNN)
        for model in self.sub_models():
            model.load_sub_model(d, *values)
            del values[:len(model.params)]  # Take next len(model.params) values
            if self.args.verbose > 1:
                print(model.params_str())
        if self.args.verbose:
            print(self)
        assert not values, "Loaded values: %d more than expected" % len(values)

    def get_all_params(self):
        d = super().get_all_params()
        for model in self.sub_models():
            for key, value in model.params.items():
                for name, param in [("%s%s%d%d%d" % (key, p, i, j, k), v) for i, (f, b) in
                                    enumerate(value.builder_layers)
                                    for p, r in (("f", f), ("b", b)) for j, l in enumerate(r.get_parameters())
                                    for k, v in enumerate(l)] \
                        if isinstance(value, dy.BiRNNBuilder) else ((key, value),):
                    d["_".join(model.save_path + (name,))] = param.as_array()
        return d
