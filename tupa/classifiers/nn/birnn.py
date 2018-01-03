import dynet as dy
import numpy as np

from .constants import ACTIVATIONS, INITIALIZERS, RNNS, CategoricalParameter
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from ...model_util import MISSING_VALUE


class BiRNN(SubModel):
    def __init__(self, args, model, save_path=None, with_birnn=True):
        super().__init__(save_path=save_path)
        self.args = args
        self.model = model
        self.dropout = self.args.dropout
        self.lstm_layers = self.args.lstm_layers
        self.lstm_layer_dim = self.args.lstm_layer_dim
        self.embedding_layers = self.args.embedding_layers if self.lstm_layer_dim and self.lstm_layers else 0
        self.embedding_layer_dim = self.args.embedding_layer_dim
        self.max_length = self.args.max_length
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.rnn_builder = CategoricalParameter(RNNS, self.args.rnn)
        self.input_reps = self.empty_rep = self.indexed_dim = self.indexed_num = None
        self.mlp = MultilayerPerceptron(self.args, self.model, params=self.params, layers=self.embedding_layers,
                                        layer_dim=self.embedding_layer_dim, output_dim=self.lstm_layer_dim)
        self.save_path = save_path
        self.with_birnn = with_birnn

    def init_params(self, indexed_dim, indexed_num):
        """
        Initialize BiRNN builder
        :return: total output dimension of BiRNN
        """
        if self.with_birnn and self.lstm_layer_dim and self.lstm_layers:
            if self.params:
                assert self.indexed_dim == indexed_dim, "Input dim changed: %d != %d" % (self.indexed_dim, indexed_dim)
                assert self.indexed_num == indexed_num, "Input num changed: %d != %d" % (self.indexed_num, indexed_num)
            else:
                self.indexed_dim = indexed_dim
                self.indexed_num = indexed_num
                self.mlp.init_params(indexed_dim)
                self.params["birnn"] = dy.BiRNNBuilder(self.lstm_layers,
                                                       self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                                                       self.lstm_layer_dim, self.model, self.rnn_builder())
                if self.args.verbose > 3:
                    print("Initializing BiRNN: %s" % self)
            return indexed_num * self.lstm_layer_dim
        return 0

    def init_features(self, embeddings, train=False):
        if self.params:
            inputs = [self.mlp.evaluate(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
            if self.args.verbose > 3:
                print("Transducing %d inputs with dropout %s" % (len(inputs), self.dropout if train else "disabled"))
            birnn = self.params["birnn"]
            if train:
                birnn.set_dropout(self.dropout)
            else:
                birnn.disable_dropout()
            self.input_reps = birnn.transduce(inputs[:self.max_length])
            self.empty_rep = dy.inputVector(np.zeros(self.lstm_layer_dim, dtype=float))

    def evaluate(self, indices):
        """
        :param indices: indices of inputs
        :return: list of BiRNN outputs at given indices
        """
        if self.params:
            assert len(indices) == self.indexed_num, "Input size mismatch: %d != %d" % (len(indices), self.indexed_num)
            return [self.empty_rep if i == MISSING_VALUE else
                    self.input_reps[min(i, self.max_length - 1)] for i in indices]
        return []

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
            ("rnn", str(self.rnn_builder)),
            ("indexed_dim", self.indexed_dim),
            ("indexed_num", self.indexed_num),
        ) if self.with_birnn and self.lstm_layer_dim and self.lstm_layers else []
        if self.args.verbose > 3:
            print("Saving BiRNN: %s" % self)
        return values

    def load_sub_model(self, d, *args):
        d = super().load_sub_model(d, *args)
        if self.with_birnn and d:
            self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
            self.args.lstm_layer_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
            self.args.embedding_layers = self.embedding_layers = d["embedding_layers"]
            self.args.embedding_layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
            self.args.max_length = self.max_length = d["max_length"]
            self.args.rnn = self.rnn_builder.string = d["rnn"]
            self.args.activation = self.activation.string = d["activation"]
            self.args.init = self.init.string = d["init"]
            self.args.dropout = self.dropout = d["dropout"]
            self.indexed_dim = self.mlp.input_dim = d["indexed_dim"]
            self.indexed_num = d["indexed_num"]
        else:
            self.args.lstm_layers = self.lstm_layers = self.args.lstm_layer_dim = self.lstm_layer_dim = 0
        if self.args.verbose > 3:
            print("Loading BiRNN: %s" % self)

    def __str__(self):
        return "%s lstm_layers: %d, lstm_layer_dim: %d, embedding_layers: %d, embedding_layer_dim: %d, " \
               "max_length: %d, rnn: %s, activation: %s, init: %s, dropout: %f, indexed_dim: %s, indexed_num: %s, "\
               "params: %s" % (
                "/".join(self.save_path), self.lstm_layers, self.lstm_layer_dim, self.embedding_layers,
                self.embedding_layer_dim, self.max_length, self.rnn_builder, self.activation, self.init, self.dropout,
                self.indexed_dim, self.indexed_num, self.params.keys())
