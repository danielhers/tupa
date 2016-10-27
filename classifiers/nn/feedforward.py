import time
from collections import defaultdict

import numpy as np
import dynet as dy

from nn.neural_network import NeuralNetwork
from parsing import config


class FeedforwardNeuralNetwork(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(FeedforwardNeuralNetwork, self).__init__(*args, model_type=config.FEEDFORWARD_NN, **kwargs)
        self._samples = defaultdict(list)
        self._inputs = {}
        self._input_values = {}
        self._label = self._loss_value = self._trainer = None

    def init_model(self):
        if self.model is not None:
            return
        if config.Config().args.verbose:
            print("Input: " + self._input_params)
        self.model = dy.Model()
        input_dim = 0
        for suffix, param in self._input_params.items():
            if not param.numeric:  # index feature
                p = self.model.add_lookup_parameters((param.size, param.dim))
                if param.init:
                    for i, row in enumerate(param.init):
                        p.init_row(i, row)
                self._params[suffix] = p
            self._input_values[suffix] = dy.vecInput(param.num * param.dim)
            input_dim += param.num * param.dim
        for i in range(1, self._layers + 1):
            in_dim = input_dim if i == 1 else self._layer_dim
            out_dim = self._layer_dim if i < self._layers else self.max_num_labels
            self._params["W%d" % i] = self.model.add_parameters((out_dim, in_dim))
            self._params["b%d" % i] = self.model.add_parameters(out_dim)
        self._trainer = self._optimizer(self.model)

    def _generate_inputs(self):
        for suffix, param in self._input_params.items():
            xs = self._inputs[suffix]
            value = self._input_values[suffix]
            if not param.numeric:
                xs = dy.concatenate(list(self._params[suffix].batch(xs)))
            value.set(xs)
            yield value

    def _eval(self):
        dy.renew_cg()
        x = dy.concatenate(list(self._generate_inputs()))
        for i in range(1, self._layers + 1):
            W = dy.parameter(self._params["W%d" % i])
            b = dy.parameter(self._params["b%d" % i])
            f = self._activation if i < self._layers else dy.softmax
            x = f(W * x + b)
        self._loss_value = self._loss(x, dy.scalarInput(self._label))
        return x

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :return: array with score for each label
        """
        super(FeedforwardNeuralNetwork, self).score(features)
        if not self.is_frozen and self._iteration == 0:  # not fit yet
            return np.zeros(self.num_labels)
        self.init_model()
        for suffix, value in features.items():
            self._inputs[suffix].set(value)
        scores = self._eval()
        return scores[0, :self.num_labels]

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: add this many samples with the same features
        """
        super(FeedforwardNeuralNetwork, self).update(features, pred, true, importance)
        self.init_model()
        for _ in range(int(importance)):
            for suffix, value in features.items():
                self._inputs[suffix] = value
            self._label = true
            self._eval()
            self._loss_value.backward()
            self._trainer.update()

    def finish(self, train=False):
        """
        Mark the current item as finished.  Fit the model if reached the batch size.
        :param train: fit the model if batch size reached?
        """
        self._item_index += 1
        if train and self._batch_size is not None and self._item_index >= self._batch_size:
            self.finalize(freeze=False)

    def finalize(self, freeze=True):
        """
        Fit this model on collected samples, and return a frozen model
        :return new FeedforwardNeuralNetwork object with the same weights, after fitting
        """
        super(FeedforwardNeuralNetwork, self).finalize()
        if self._samples:
            started = time.time()
            print("Fitting model...", flush=True)
            self.init_model()
            self._trainer.update_epoch()
            self._samples = defaultdict(list)
            self._item_index = 0
            self._iteration += 1
            print("Done (%.3fs)." % (time.time() - started))
        if freeze:
            print("Labels: %d" % self.num_labels)
            print("Features: %d" % sum(f.num * (f.dim or 1) for f in self._input_params.values()))
            return FeedforwardNeuralNetwork(self.filename, list(self.labels), model=self.model)
        return None
