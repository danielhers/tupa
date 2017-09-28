import dynet as dy
import numpy as np

from tupa.features.feature_params import MISSING_VALUE

RNNS = {
    "simple": dy.SimpleRNNBuilder,
    "gru": dy.GRUBuilder,
    "lstm": dy.LSTMBuilder,
    "vanilla_lstm": dy.VanillaLSTMBuilder,
    "compact_vanilla_lstm": dy.CompactVanillaLSTMBuilder,
    "coupled_lstm": dy.CoupledLSTMBuilder,
    "fast_lstm": dy.FastLSTMBuilder,
}


class BiRNN(object):
    def __init__(self, args, params, init, dropout, activation):
        self.args = args
        self.params = params
        self.init = init
        self.dropout = dropout
        self.activation = activation
        self.lstm_layers = self.args.lstm_layers
        self.lstm_layer_dim = self.args.lstm_layer_dim
        self.embedding_layers = self.args.embedding_layers
        self.embedding_layer_dim = self.args.embedding_layer_dim
        self.max_length = self.args.max_length
        self.rnn_str = self.args.rnn
        self.rnn_builder = RNNS[self.rnn_str]
        self.model = self.input_reps = self.empty_rep = None

    @property
    def empty(self):
        return not (self.lstm_layer_dim and self.lstm_layers)

    def init_indexed_input_params(self, indexed_dim, indexed_num):
        """
        Initialize BiRNN builder
        :return: total output dimension of BiRNN
        """
        if self.empty:
            return indexed_dim * indexed_num
        for i in range(1, self.embedding_layers + 1):
            in_dim = indexed_dim if i == 1 else self.embedding_layer_dim
            out_dim = self.embedding_layer_dim if i < self.embedding_layers else self.lstm_layer_dim
            self.params[("We", i)] = self.model.add_parameters((out_dim, in_dim), init=self.init)
            self.params[("be", i)] = self.model.add_parameters(out_dim, init=self.init)
        self.params["bilstm"] = dy.BiRNNBuilder(self.lstm_layers,
                                                self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                                                self.lstm_layer_dim, self.model, self.rnn_builder)
        return indexed_num * self.lstm_layer_dim

    def init_features(self, features, train=False):
        if self.empty:
            return
        embeddings = [[self.params[s][k] for k in ks] for s, ks in sorted(features.items())]  # time-lists of vectors
        inputs = [self.evaluate_embeddings(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
        bilstm = self.params["bilstm"]
        if train:
            bilstm.set_dropout(self.dropout)
        else:
            bilstm.disable_dropout()
        self.input_reps = bilstm.transduce(inputs[:self.max_length])
        self.empty_rep = dy.inputVector(np.zeros(self.lstm_layer_dim, dtype=float))

    def evaluate_embeddings(self, embeddings, train=False):
        """
        Apply MLP to process a single time-step of embeddings to prepare input for LSTM
        :param embeddings: list of embedding features for a single time step
        :param train: whether to apply dropout
        :return: expression corresponding to MLP output
        """
        x = dy.concatenate(list(embeddings))
        for i in range(1, self.embedding_layers + 1):
            W = dy.parameter(self.params[("We", i)])
            b = dy.parameter(self.params[("be", i)])
            if train and self.dropout:
                x = dy.dropout(x, self.dropout)
            x = self.activation(W * x + b)
        return x

    def index_input(self, indices):
        """
        :param indices: indices of inputs
        :return: BiRNN outputs at given indices
        """
        if self.empty:
            raise ValueError("Input representations not initialized, cannot evaluate indexed features")
        return dy.concatenate([self.empty_rep if i == MISSING_VALUE else self.input_reps[min(i, self.max_length - 1)]
                               for i in indices])

    def save(self):
        return {
            "lstm_layers": self.lstm_layers,
            "lstm_layer_dim": self.lstm_layer_dim,
            "embedding_layers": self.embedding_layers,
            "embedding_layer_dim": self.embedding_layer_dim,
            "max_length": self.max_length,
            "rnn": self.rnn_str,
        }

    def load(self, d):
        self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
        self.args.lstm_layer_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
        self.args.embedding_layers = self.embedding_layers = d["embedding_layers"]
        self.args.embedding_layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
        self.args.max_length = self.max_length = d["max_length"]
        self.args.rnn = self.rnn_str = d.get("rnn", self.args.rnn)
