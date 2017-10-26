from collections import OrderedDict

import dynet as dy
import numpy as np

from .constants import ACTIVATIONS, INITIALIZERS, RNNS, CategoricalParameter
from .mlp import MultilayerPerceptron
from ...features.feature_params import MISSING_VALUE


class BiRNN(object):
    def __init__(self, args, model, global_params, params=None, shared=False):
        self.args = args
        self.model = model
        self.params = global_params if shared else params or OrderedDict()  # string (param identifier) -> parameter
        self.global_params = global_params  # from parent network: string (param identifier) -> parameter
        self.dropout = self.args.dropout
        self.lstm_layers = self.args.lstm_layers
        self.lstm_layer_dim = self.args.lstm_layer_dim
        self.embedding_layers = self.args.embedding_layers
        self.embedding_layer_dim = self.args.embedding_layer_dim
        self.max_length = self.args.max_length
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.rnn_builder = CategoricalParameter(RNNS, self.args.rnn)
        self.input_reps = self.empty_rep = None
        self.mlp = MultilayerPerceptron(self.args, self.model, self.params, self.embedding_layers,
                                        self.embedding_layer_dim, self.lstm_layer_dim, suffix1="e", offset=1)

    @property
    def empty(self):
        return not (self.lstm_layer_dim and self.lstm_layers)

    def init_params(self, indexed_dim, indexed_num):
        """
        Initialize BiRNN builder
        :return: total output dimension of BiRNN
        """
        if self.empty:
            return 0
        self.mlp.init_params(indexed_dim)
        self.params["bilstm"] = dy.BiRNNBuilder(self.lstm_layers,
                                                self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                                                self.lstm_layer_dim, self.model, self.rnn_builder())
        return indexed_num * self.lstm_layer_dim

    def init_features(self, features, train=False):
        if self.empty or not self.params:
            return
        embeddings = [[self.global_params[s][k] for k in ks] for s, ks in sorted(features.items())]  # lists of vectors
        inputs = [self.mlp.evaluate(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
        bilstm = self.params["bilstm"]
        if train:
            bilstm.set_dropout(self.dropout)
        else:
            bilstm.disable_dropout()
        self.input_reps = bilstm.transduce(inputs[:self.max_length])
        self.empty_rep = dy.inputVector(np.zeros(self.lstm_layer_dim, dtype=float))

    def index_input(self, indices):
        """
        :param indices: indices of inputs
        :return: list of BiRNN outputs at given indices
        """
        return [] if self.empty or not self.params else \
            [self.empty_rep if i == MISSING_VALUE else self.input_reps[min(i, self.max_length - 1)] for i in indices]

    def save(self):
        return OrderedDict(
            lstm_layers=self.lstm_layers,
            lstm_layer_dim=self.lstm_layer_dim,
            embedding_layers=self.embedding_layers,
            embedding_layer_dim=self.embedding_layer_dim,
            max_length=self.max_length,
            activation=str(self.activation),
            init=str(self.init),
            rnn=str(self.rnn_builder),
        )

    def load(self, d):
        self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
        self.args.lstm_layer_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
        self.args.embedding_layers = self.embedding_layers = d["embedding_layers"]
        self.args.embedding_layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
        self.args.max_length = self.max_length = d["max_length"]
        self.args.rnn = self.rnn_builder.string = d.get("rnn", self.args.rnn)
        activation = d.get("activation")
        if activation:
            self.args.activation = self.activation.string = activation
        init = d.get("init")
        if init:
            self.args.init = self.init.string = init
