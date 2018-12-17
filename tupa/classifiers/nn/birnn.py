import numpy as np
import torch

from .constants import Activation, Initializer, RNN
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from .util import randomize_orthonormal
from ...model_util import MISSING_VALUE


class BiRNN(SubModel):
    def __init__(self, config, args, model, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.args = args
        self.model = model
        self.dropout = self.args.dropout
        self.lstm_layers = self.args.lstm_layers
        self.lstm_layer_dim = self.args.lstm_layer_dim
        self.embedding_layers = self.args.embedding_layers if self.lstm_layer_dim and self.lstm_layers else 0
        self.embedding_layer_dim = self.args.embedding_layer_dim
        self.max_length = self.args.max_length
        self.activation = Activation(self.args.activation)
        self.init = Initializer(self.args.init)
        self.rnn = RNN(self.args.rnn)
        self.input_reps = self.indexed_dim = self.indexed_num = None
        self.mlp = MultilayerPerceptron(self.config, self.args, self.model, params=self.params,
                                        layers=self.embedding_layers, layer_dim=self.embedding_layer_dim,
                                        output_dim=self.lstm_layer_dim)

    def init_params(self, indexed_dim, indexed_num):
        """
        Initialize BiRNN builder
        :return: total output dimension of BiRNN
        """
        if self.lstm_layer_dim and self.lstm_layers:
            if self.params:
                assert self.indexed_dim == indexed_dim, "Input dim changed: %d != %d" % (self.indexed_dim, indexed_dim)
                assert self.indexed_num == indexed_num, "Input num changed: %d != %d" % (self.indexed_num, indexed_num)
            else:
                self.indexed_dim = indexed_dim
                self.indexed_num = indexed_num
                randomize_orthonormal(**self.init_rnn_params(indexed_dim))
                self.config.print("Initializing BiRNN: %s" % self, level=4)

    def init_rnn_params(self, indexed_dim):
        self.params["birnn"] = rnn = self.rnn(self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                                              self.lstm_layer_dim, self.lstm_layers, bidirectional=True,
                                              dropout=self.dropout)
        return rnn.named_parameters()

    def init_features(self, embeddings, train=False):
        """
        Set the value of self.input_reps given embeddings for the whole input sequence
        :param embeddings: list of [(key, list of vectors embeddings per time step)] per feature
        :param train: are we training now?
        """
        if self.params:
            for module in self.params.values():
                module.train(mode=train)
            keys, embeddings = zip(*embeddings)
            inputs = [self.mlp.evaluate(zip(keys, es), train=train) for es in zip(*embeddings)]  # join each time step
            self.config.print("Transducing %d inputs with dropout %s" %
                              (len(inputs), self.dropout if train else "disabled"), level=4)
            self.input_reps = self.transduce(inputs, train)
            expected = min(len(inputs), self.max_length or np.iinfo(int).max)
            assert len(self.input_reps) == expected, \
                "transduce() returned incorrect number of elements: %d != %d" % (len(self.input_reps), expected)

    def transduce(self, inputs, train):
        birnn = self.params["birnn"]
        return birnn(torch.DoubleTensor(inputs[:self.max_length]))

    def evaluate(self, indices):
        """
        :param indices: indices of inputs
        :return: list of BiRNN outputs at given indices
        """
        if self.params:
            assert len(indices) == self.indexed_num, "Input size mismatch: %d != %d" % (len(indices), self.indexed_num)
            return [("/".join(self.save_path),
                     torch.zeros(self.lstm_layer_dim, dtype=torch.float64) if i == MISSING_VALUE
                     else self.get_representation(i)) for i in indices]
        return []

    def get_representation(self, i):
        return self.input_reps[min(i, self.max_length - 1)]

    def save_sub_model(self, d, *args):
        values = super().save_sub_model(
            d,
            ("lstm_layers", self.lstm_layers),
            ("lstm_layer_dim", self.lstm_layer_dim),
            ("embedding_layers", self.embedding_layers),
            ("embedding_layer_dim", self.embedding_layer_dim),
            ("max_length", self.max_length),
            ("activation", str(self.activation)),
            ("init", str(self.init)),
            ("dropout", self.dropout),
            ("rnn", str(self.rnn)),
            ("indexed_dim", self.indexed_dim),
            ("indexed_num", self.indexed_num),
            ("input_dim", self.mlp.input_dim),
            ("gated", self.mlp.gated),
        ) if self.lstm_layer_dim and self.lstm_layers else []
        self.config.print("Saving BiRNN: %s" % self, level=4)
        return values

    def load_sub_model(self, d, *args, **kwargs):
        d = super().load_sub_model(d, *args, **kwargs)
        if d:
            self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
            self.args.lstm_layer_dim = self.mlp.output_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
            self.args.embedding_layers = self.mlp.layers = self.embedding_layers = d["embedding_layers"]
            self.args.embedding_layer_dim = self.mlp.layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
            self.args.max_length = self.max_length = d["max_length"]
            self.args.rnn = self.rnn.string = d["rnn"]
            self.args.activation = self.mlp.activation.string = self.activation.string = d["activation"]
            self.args.init = self.mlp.init.string = self.init.string = d["init"]
            self.args.dropout = self.mlp.dropout = self.dropout = d["dropout"]
            self.indexed_dim = self.mlp.input_dim = d["indexed_dim"]
            self.indexed_num = d["indexed_num"]
            self.mlp.input_dim = d.get("input_dim")
            self.mlp.gated = d.get("gated")
        else:
            self.args.lstm_layers = self.lstm_layers = self.args.lstm_layer_dim = self.lstm_layer_dim = 0
        self.config.print("Loading BiRNN: %s" % self, level=4)

    def __str__(self):
        return "%s lstm_layers: %d, lstm_layer_dim: %d, embedding_layers: %d, embedding_layer_dim: %d, " \
               "max_length: %d, rnn: %s, activation: %s, init: %s, dropout: %f, indexed_dim: %s, indexed_num: %s, "\
               "params: %s" % (
                "/".join(self.save_path), self.lstm_layers, self.lstm_layer_dim, self.embedding_layers,
                self.embedding_layer_dim, self.max_length, self.rnn, self.activation, self.init, self.dropout,
                self.indexed_dim, self.indexed_num, list(self.params.keys()))

    def sub_models(self):
        return [self.mlp]


class EmptyRNN(BiRNN):
    def init_params(self, indexed_dim, indexed_num):
        pass

    def init_features(self, embeddings, train=False):
        pass

    def evaluate(self, indices):
        return []

    def save_sub_model(self, d, *args):
        return []

    def load_sub_model(self, d, *args, **kwargs):
        super().load_sub_model(d, *args)
        self.args.lstm_layers = self.lstm_layers = self.args.lstm_layer_dim = self.lstm_layer_dim = 0
