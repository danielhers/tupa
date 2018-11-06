import sys
from collections import OrderedDict
from itertools import repeat

import dynet as dy
import numpy as np
from tqdm import tqdm

from .birnn import EmptyRNN, BiRNN, HighwayRNN, HierarchicalBiRNN
from .constants import TRAINERS, TRAINER_LEARNING_RATE_PARAM_NAMES, TRAINER_KWARGS, CategoricalParameter
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from ..classifier import Classifier
from ...config import Config, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN
from ...model_util import MISSING_VALUE, remove_existing

BIRNN_TYPES = {BIRNN: BiRNN, HIGHWAY_RNN: HighwayRNN, HIERARCHICAL_RNN: HierarchicalBiRNN}

tqdm.monitor_interval = 0

try:
    print("[dynet] %s" % dy.__gitversion__, file=sys.stderr, flush=True)
except AttributeError:
    pass


class AxisModel:
    """
    Format-specific parameters that are part of the network
    """
    def __init__(self, axis, num_labels, config, model, birnn_type):
        args = config.hyperparams.specific[axis]
        self.birnn = birnn_type(config, args, model, save_path=("axes", axis, "birnn"),
                                copy_shared=args.copy_shared == [] or axis in (args.copy_shared or ()))
        self.mlp = MultilayerPerceptron(config, args, model, num_labels=num_labels, save_path=("axes", axis, "mlp"))


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
        self.minibatch_size = self.config.args.minibatch_size
        self.loss = self.config.args.loss
        self.weight_decay = self.config.args.dynet_weight_decay
        self.empty_values = OrderedDict()  # string (feature param key) -> expression
        self.axes = OrderedDict()  # string (axis) -> AxisModel
        self.losses = []
        self.steps = 0
        self.trainer_type = self.trainer = self.value = self.birnn = None

    @property
    def input_dim(self):
        return OrderedDict((a, m.mlp.input_dim) for a, m in self.axes.items())
    
    @property
    def birnn_type(self):
        return BIRNN_TYPES.get(self.model_type, EmptyRNN)

    def resize(self):
        for axis, labels in self.labels.items():
            if labels.size is not None:
                num_labels = self.num_labels[axis]
                assert num_labels <= labels.size, "Exceeded maximum number of labels at axis '%s': %d > %d:\n%s" % (
                    axis, num_labels, labels.size, "\n".join(map(str, labels.all)))

    def init_model(self, axis=None, train=False):
        init = self.model is None
        if init:
            self.model = dy.ParameterCollection()
            self.birnn = self.birnn_type(self.config, Config().hyperparams.shared, self.model,
                                         save_path=("shared", "birnn"), shared=True)
            self.set_weight_decay_lambda()
        if train:
            self.init_trainer()
        if axis:
            self.init_axis_model(axis)
        if init:
            self.init_cg()
            self.finished_step()

    def set_weight_decay_lambda(self, weight_decay=None):
        self.model.set_weight_decay_lambda(self.weight_decay if weight_decay is None else weight_decay)

    def init_trainer(self):
        if self.trainer_type is None or str(self.trainer_type) != self.config.args.optimizer:
            self.trainer_type = CategoricalParameter(TRAINERS, self.config.args.optimizer)
            trainer_kwargs = dict(TRAINER_KWARGS.get(str(self.trainer_type), {}))
            learning_rate_param_name = TRAINER_LEARNING_RATE_PARAM_NAMES.get(str(self.trainer_type))
            if learning_rate_param_name and self.learning_rate:
                trainer_kwargs[learning_rate_param_name] = self.learning_rate
            self.config.print("Initializing trainer=%s(%s)" % (
                self.trainer_type, ", ".join("%s=%s" % (k, v) for k, v in trainer_kwargs.items())), level=4)
            self.trainer = self.trainer_type()(self.model, **trainer_kwargs)
            self.trainer.set_sparse_updates(False)

    def init_axis_model(self, axis, init=True):
        if axis in self.axes:
            if init:
                return
        else:
            self.axes[axis] = AxisModel(axis, self.labels[axis].size, self.config, self.model, self.birnn_type)
            self.config.print("Initializing %s model with %d labels" % (axis, self.labels[axis].size), level=4)
        indexed_dim = np.array([0, 0], dtype=int)  # specific, shared
        indexed_num = np.array([0, 0], dtype=int)
        for key, param in sorted(self.input_params.items()):
            if not param.enabled:
                continue
            self.config.print("Initializing input parameter: %s" % param, level=4)
            if not param.numeric and key not in self.params:  # lookup feature
                if init:
                    lookup = self.model.add_lookup_parameters((param.size, param.dim))
                    lookup.set_updated(param.updated)
                    param.init_data()
                    if param.init is not None and param.init.size:
                        lookup.init_from_array(param.init)
                    self.params[key] = lookup
            if param.indexed:
                i = self.birnn_indices(param)
                indexed_dim[i] += param.dim  # add to the input dimensionality at each indexed time point
                indexed_num[i] = np.fmax(indexed_num[i], param.num)  # indices to be looked up are collected
        for birnn in self.get_birnns(axis):
            birnn.init_params(indexed_dim[int(birnn.shared)], indexed_num[int(birnn.shared)])

    def birnn_indices(self, param):  # both specific and shared or just specific
        return [0, 1] if not self.config.args.multilingual or not param.lang_specific else [0]

    def init_cg(self, renew=True):
        if renew:
            dy.renew_cg()
        self.empty_values.clear()

    def get_empty_values(self, key):
        value = self.empty_values.get(key)
        if value is None:
            self.empty_values[key] = value = dy.inputVector(np.zeros(self.input_params[key].dim, dtype=float))
        return value

    def init_features(self, features, axes, train=False):
        for axis in axes:
            self.init_model(axis, train)
        embeddings = [[], []]  # specific, shared
        self.config.print("Initializing %s %s features for %d elements" %
                          (", ".join(axes), self.birnn_type.__name__, len(features)), level=4)
        for key, indices in sorted(features.items()):
            param = self.input_params[key]
            lookup = self.params.get(key)
            if not param.indexed or lookup is None:
                continue
            vectors = [lookup[k] for k in indices]
            for index in self.birnn_indices(param):
                embeddings[index].append((key, vectors))
            self.config.print(lambda: "%s: %s" % (key, ", ".join("%d->%s" % (i, e.npvalue().tolist())
                                                                 for i, e in zip(indices, vectors))), level=4)
        for birnn in self.get_birnns(*axes):
            birnn.init_features(embeddings[int(birnn.shared)], train)

    def generate_inputs(self, features, axis):
        indices = []  # list, not set, in order to maintain consistent order
        for key, values in sorted(features.items()):
            param = self.input_params[key]
            lookup = self.params.get(key)
            if param.numeric:
                yield key, dy.inputVector(values)
            elif param.indexed:  # collect indices to be looked up
                indices += values  # DenseFeatureExtractor collapsed features so there are no repetitions between them
            elif lookup is None:  # ignored
                continue
            else:  # lookup feature
                yield from ((key, self.get_empty_values(key) if x == MISSING_VALUE else lookup[x]) for x in values)
            self.config.print(lambda: "%s: %s" % (key, values), level=4)
        if indices:
            for birnn in self.get_birnns(axis):
                yield from birnn.evaluate(indices)

    def get_birnns(self, *axes):
        """ Return shared + axis-specific BiRNNs """
        return [m.birnn for m in [self] + [self.axes[axis] for axis in axes]]

    def evaluate(self, features, axis, train=False):
        """
        Apply MLP and log softmax to input features
        :param features: dictionary of key, values for each feature type
        :param axis: axis of the label we are predicting
        :param train: whether to apply dropout
        :return: expression corresponding to log softmax applied to MLP output
        """
        self.init_model(axis, train)
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
        self.config.print("  no updates done yet, returning zero vector.", level=4)
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
        self.config.print(lambda: "  loss=" + ", ".join("%g" % l.value() for l in losses), level=4)
        self.losses += losses
        self.steps += 1

    def calc_loss(self, scores, axis, true, importance):
        ret = [i * dy.pickneglogsoftmax(scores, t) for t, i in zip(true, importance)]
        if self.loss == "max_margin":
            ret.append(dy.max_dim(dy.log_softmax(scores, restrict=list(set(range(self.num_labels[axis])) - set(true)))))
        return ret

    def finished_step(self, train=False):
        super().invalidate_caches()

    def invalidate_caches(self):
        self.value = {}  # For caching the result of _evaluate

    def finished_item(self, train=False, renew=True):
        if self.steps >= self.minibatch_size:
            self.finalize()
        elif not train:
            self.init_cg(renew)
        self.finished_step(train)

    def transition(self, action, axis):
        for birnn in self.get_birnns(axis):
            birnn.transition(action)

    def finalize(self, finished_epoch=False, **kwargs):
        """
        Fit this model on collected samples
        :return self
        """
        super().finalize(finished_epoch=finished_epoch, **kwargs)
        assert self.model, "Cannot finalize a model without initializing it first"
        if self.losses:
            loss = dy.esum(self.losses)
            loss.forward()
            self.config.print(lambda: "Total loss from %d time steps: %g" % (self.steps, loss.value()), level=4)
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
        if self.config.args.verbose > 2:
            self.trainer.status()
        return self
            
    def sub_models(self):
        """ :return: ordered list of SubModels """
        axes = list(filter(None, map(self.axes.get, self.labels or self.labels_t)))
        return [self] + [m.mlp for m in axes] + [m.birnn for m in axes + [self]]
    
    def save_sub_model(self, d, *args):
        return SubModel.save_sub_model(
            self, d,
            ("loss", self.loss),
            ("weight_decay", self.weight_decay),
        )

    def load_sub_model(self, d, *args, **kwargs):
        d = SubModel.load_sub_model(self, d, *args, **kwargs)
        self.config.args.loss = self.loss = d["loss"]
        self.config.args.dynet_weight_decay = self.weight_decay = d.get("weight_decay",
                                                                        self.config.args.dynet_weight_decay)

    def save_model(self, filename, d):
        Classifier.save_model(self, filename, d)
        self.finalize()
        values = []
        for model in self.sub_models():
            values += model.save_sub_model(d)
            if self.config.args.verbose <= 3:
                self.config.print(model.params_str, level=1)
        self.config.print(self, level=1)
        self.save_param_values(filename, values)

    def save_param_values(self, filename, values):
        remove_existing(filename + ".data", filename + ".meta")
        try:
            self.set_weight_decay_lambda(0.0)  # Avoid applying weight decay due to clab/dynet#1206, we apply it on load
            dy.save(filename, tqdm(values, desc="Saving model to '%s'" % filename, unit="param", file=sys.stdout))
            self.set_weight_decay_lambda()
        except ValueError as e:
            print("Failed saving model: %s" % e)

    def load_model(self, filename, d):
        self.model = None
        self.init_model()
        values = self.load_param_values(filename, d)
        self.axes = OrderedDict()
        for axis, labels in self.labels_t.items():
            _, size = labels
            assert size, "Size limit for '%s' axis labels is %s" % (axis, size)
            self.axes[axis] = AxisModel(axis, size, self.config, self.model, self.birnn_type)
        for model in self.sub_models():
            model.load_sub_model(d, *values)
            del values[:len(model.params)]  # Take next len(model.params) values
            if self.config.args.verbose <= 3:
                self.config.print(model.params_str)
        self.copy_shared_birnn(filename, d)
        assert not values, "Loaded values: %d more than expected" % len(values)
        if self.weight_decay and self.config.args.dynet_apply_weight_decay_on_load:
            t = tqdm(list(self.all_params(as_array=False).items()),
                     desc="Applying weight decay of %g" % self.weight_decay, unit="param", file=sys.stdout)
            for key, param in t:
                t.set_postfix(param=key)
                try:
                    value = param.as_array() * np.float_power(1 - self.weight_decay, self.updates)
                except AttributeError:
                    continue
                try:
                    param.set_value(value)
                except AttributeError:
                    param.init_from_array(value)
        self.config.print(self, level=1)

    def load_param_values(self, filename, d=None):
        return list(tqdm(dy.load_generator(filename, self.model), total=self.params_num(d) if d else None,
                         desc="Loading model from '%s'" % filename, unit="param", file=sys.stdout))

    def copy_shared_birnn(self, filename, d):
        shared_values = None
        values = self.load_param_values(filename, d)  # Load parameter values again so that shared parameters are copied
        for model in self.sub_models():
            if model is self.birnn:
                shared_values = values[:len(model.params)]
            del values[:len(model.params)]  # Take next len(model.params) values
        for axis, model in self.axes.items():
            if model.birnn.copy_shared:
                model.birnn.load_sub_model(d, *shared_values, load_path=self.birnn.save_path)
                if self.config.args.verbose <= 3:
                    self.config.print(lambda: "Copied from %s to %s" %
                                      ("/".join(self.birnn.save_path), model.birnn.params_str()), level=1)
                self.init_axis_model(axis, init=False)  # Update input_dim

    def params_num(self, d):
        return sum(len(m.get_sub_dict(d).get("param_keys", ())) for m in self.sub_models())

    def all_params(self, as_array=True):
        d = super().all_params()
        for model in self.sub_models():
            for key, value in model.params.items():
                for name, param in ((key, value),) if isinstance(value, (dy.Parameters, dy.LookupParameters)) else [
                        ("%s%s%d%d%d" % (key, p, i, j, k), v) for i, (f, b) in
                        enumerate(value.builder_layers)
                        for p, r in (("f", f), ("b", b)) for j, l in enumerate(r.get_parameters())
                        for k, v in enumerate(l)] if isinstance(value, dy.BiRNNBuilder) else [
                        ("%s%d%d" % (key, j, k), v) for j, l in enumerate(value.get_parameters())
                        for k, v in enumerate(l)]:
                    d["_".join(model.save_path + (name,))] = param.as_array() if as_array else param
        return d

    def print_params(self, max_rows=10):
        for model in self.sub_models():
            for key, value in model.params.items():
                print("[%s] %s" % (model.params_str(), key))
                # noinspection PyBroadException
                try:
                    print(value.as_array()[:max_rows])
                except Exception:
                    pass
