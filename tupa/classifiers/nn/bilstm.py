import dynet as dy

from tupa.config import BILSTM_NN
from tupa.features.feature_params import MISSING_VALUE
from .neural_network import NeuralNetwork


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(BILSTM_NN, *args, **kwargs)
        self.lstm_layers = self.args.lstm_layers
        self.lstm_layer_dim = self.args.lstm_layer_dim
        self.embedding_layers = self.args.embedding_layers
        self.embedding_layer_dim = self.args.embedding_layer_dim
        self.max_length = self.args.max_length
        self.input_reps = None
        self.empty_rep = None

    def init_indexed_input_params(self):
        """
        Initialize BiLSTM builder
        :return: total output dimension of BiLSTM
        """
        for i in range(1, self.embedding_layers + 1):
            in_dim = self.indexed_dim if i == 1 else self.embedding_layer_dim
            out_dim = self.embedding_layer_dim if i < self.embedding_layers else self.lstm_layer_dim
            self.params[("We", i)] = self.model.add_parameters((out_dim, in_dim), init=self.init)
            self.params[("be", i)] = self.model.add_parameters(out_dim, init=self.init)
        self.params["bilstm"] = dy.BiRNNBuilder(self.lstm_layers,
                                                self.lstm_layer_dim if self.embedding_layers else self.indexed_dim,
                                                self.lstm_layer_dim, self.model, dy.LSTMBuilder)
        return self.indexed_num * self.lstm_layer_dim

    def init_features(self, features, train=False):
        self.init_model()
        embeddings = [[self.params[s][k] for k in ks] for s, ks in sorted(features.items())]  # time-lists of vectors
        inputs = [self.evaluate_embeddings(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
        bilstm = self.params["bilstm"]
        if train:
            bilstm.set_dropout(self.dropout)
        else:
            bilstm.disable_dropout()
        self.input_reps = bilstm.transduce(inputs[:self.max_length])
        self.empty_rep = self.zero_input(self.lstm_layer_dim)

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
        :return: BiLSTM outputs at given indices
        """
        return dy.concatenate([self.empty_rep if i == MISSING_VALUE else self.input_reps[min(i, self.max_length - 1)]
                               for i in indices])

    def save_extra(self):
        return {
            "lstm_layers": self.lstm_layers,
            "lstm_layer_dim": self.lstm_layer_dim,
            "embedding_layers": self.embedding_layers,
            "embedding_layer_dim": self.embedding_layer_dim,
            "max_length": self.max_length,
        }

    def load_extra(self, d):
        self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
        self.args.lstm_layer_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
        self.args.embedding_layers = self.embedding_layers = d["embedding_layers"]
        self.args.embedding_layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
        self.args.max_length = self.max_length = d["max_length"]
