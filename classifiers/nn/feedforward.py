import sys

import numpy as np

import dynet as dy
from nn.neural_network import NeuralNetwork
from parsing import config


class FeedforwardNeuralNetwork(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(FeedforwardNeuralNetwork, self).__init__(*args, model_type=config.FEEDFORWARD_NN, **kwargs)
        self._trainer = None
        self._losses = []
        self._iteration = 0

    def init_model(self):
        self.model = dy.Model()
        self._trainer = self._optimizer(self.model)
        input_dim = 0
        for suffix, param in sorted(self._input_params.items()):
            if not param.numeric and param.dim > 0:  # index feature
                p = self.model.add_lookup_parameters((param.size, param.dim))
                if param.init is not None:
                    p.init_from_array(param.init)
                self._params[suffix] = p
            input_dim += param.num * param.dim
        for i in range(1, self._layers + 1):
            in_dim = input_dim if i == 1 else self._layer_dim
            out_dim = self._layer_dim if i < self._layers else self.max_num_labels
            self._params["W%d" % i] = self.model.add_parameters((out_dim, in_dim), init=self._init)
            self._params["b%d" % i] = self.model.add_parameters(out_dim, init=self._init)

    def _generate_inputs(self, features):
        for suffix, param in sorted(self._input_params.items()):
            xs = features[suffix]
            if param.numeric:
                yield dy.inputVector(xs)
            elif param.dim > 0:
                yield dy.reshape(self._params[suffix].batch(xs), (param.num * param.dim,))

    def _eval(self, features, train=False):
        if self.model is None:
            self.init_model()
        if not self._losses:
            dy.renew_cg()
        x = dy.concatenate(list(self._generate_inputs(features)))
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
        super(FeedforwardNeuralNetwork, self).score(features)
        return self._eval(features).npvalue()[:self.num_labels] if self._iteration > 0 else np.zeros(self.num_labels)

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: add this many samples with the same features
        """
        super(FeedforwardNeuralNetwork, self).update(features, pred, true, importance)
        for _ in range(int(importance)):
            self._losses.append(dy.pick(self._eval(features, train=True), true))
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
        super(FeedforwardNeuralNetwork, self).finalize()
        if self._losses:
            loss = -dy.esum(self._losses)
            loss.forward()
            loss.backward()
            self._trainer.update()
            self._losses = []
            self._iteration += 1
        return self
