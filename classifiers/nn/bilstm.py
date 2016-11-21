import dynet as dy
from classifiers.classifier import ClassifierProperty
from features.feature_params import MISSING_VALUE
from nn.neural_network import NeuralNetwork
from parsing.config import Config, BILSTM_NN


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(BILSTM_NN, *args, **kwargs)
        self._lstm_layers = Config().args.lstmlayers
        self._lstm_layer_dim = Config().args.lstmlayerdim
        self._input_reps = None
        self._empty_rep = None

    def init_indexed_input_params(self):
        """
        Initialize BiLSTM builder
        :return: total output dimension of BiLSTM
        """
        self._params["bilstm"] = dy.BiRNNBuilder(self._lstm_layers, self._indexed_dim, self._lstm_layer_dim, self.model,
                                                 dy.LSTMBuilder)
        return self._indexed_num * self._lstm_layer_dim

    def init_features(self, features, train=False):
        if self.model is None:
            self.init_model()
        embeddings = [[self._params[s][x] for x in xs] for s, xs in sorted(features.items())]  # time-lists of vectors
        embeddings = [dy.concatenate(list(e)) for e in zip(*embeddings)]  # one list: join each time step to one vector
        bilstm = self._params["bilstm"]
        if train:
            bilstm.set_dropout(self._dropout)
        self._input_reps = bilstm.transduce(embeddings)
        self._empty_rep = self.empty_rep(self._lstm_layer_dim)

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
        return ClassifierProperty.require_init_features,
