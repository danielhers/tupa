import dynet as dy
from classifiers.classifier import ClassifierProperty
from features.feature_params import MISSING_VALUE
from nn.neural_network import NeuralNetwork
from tupa.config import Config, BILSTM_NN


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(BILSTM_NN, *args, **kwargs)
        self._lstm_layers = Config().args.lstm_layers
        self._lstm_layer_dim = Config().args.lstm_layer_dim
        self._embedding_layers = Config().args.embedding_layers
        self._embedding_layer_dim = Config().args.embedding_layer_dim
        self._input_reps = None
        self._empty_rep = None

    def init_indexed_input_params(self):
        """
        Initialize BiLSTM builder
        :return: total output dimension of BiLSTM
        """
        for i in range(1, self._embedding_layers + 1):
            in_dim = self._indexed_dim if i == 1 else self._embedding_layer_dim
            out_dim = self._embedding_layer_dim if i < self._embedding_layers else self._lstm_layer_dim
            self._params["W%de" % i] = self.model.add_parameters((out_dim, in_dim), init=self._init)
            self._params["b%de" % i] = self.model.add_parameters(out_dim, init=self._init)
        self._params["bilstm"] = dy.BiRNNBuilder(self._lstm_layers,
                                                 self._lstm_layer_dim if self._embedding_layers else self._indexed_dim,
                                                 self._lstm_layer_dim, self.model, dy.LSTMBuilder)
        return self._indexed_num * self._lstm_layer_dim

    def init_features(self, features, train=False):
        if self.model is None:
            self.init_model()
        embeddings = [[self._params[s][k] for k in ks] for s, ks in sorted(features.items())]  # time-lists of vectors
        inputs = [self.evaluate_embeddings(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
        bilstm = self._params["bilstm"]
        if train:
            bilstm.set_dropout(self._dropout)
        else:
            bilstm.disable_dropout()
        self._input_reps = bilstm.transduce(inputs)
        self._empty_rep = self.zero_input(self._lstm_layer_dim)

    def evaluate_embeddings(self, embeddings, train=False):
        """
        Apply MLP to process a single time-step of embeddings to prepare input for LSTM
        :param embeddings: list of embedding features for a single time step
        :param train: whether to apply dropout
        :return: expression corresponding to MLP output
        """
        x = dy.concatenate(list(embeddings))
        for i in range(1, self._embedding_layers + 1):
            W = dy.parameter(self._params["W%de" % i])
            b = dy.parameter(self._params["b%de" % i])
            if train and self._dropout:
                x = dy.dropout(x, self._dropout)
            x = self._activation(W * x + b)
        return x

    def index_input(self, indices):
        """
        :param indices: indices of inputs
        :return: BiLSTM outputs at given indices
        """
        return dy.concatenate([self._empty_rep if i == MISSING_VALUE else self._input_reps[i] for i in indices])

    def save_extra(self):
        return {
            "lstm_layers": self._lstm_layers,
            "lstm_layer_dim": self._lstm_layer_dim,
        }

    def load_extra(self, d):
        self._lstm_layers = d["lstm_layers"]
        self._lstm_layer_dim = d["lstm_layer_dim"]

    def get_classifier_properties(self):
        return super(BiLSTM, self).get_classifier_properties() + \
               (ClassifierProperty.require_init_features,)
