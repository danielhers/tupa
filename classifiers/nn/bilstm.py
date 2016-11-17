import dynet as dy
from classifiers.classifier import ClassifierProperty

from nn.neural_network import NeuralNetwork
from parsing import config
from parsing.config import Config


EMPTY_INDEX = 1  # used as index into lookup table for "padding" when a feature is missing


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(config.BILSTM_NN, *args, **kwargs)
        self._lstm_layers = Config().args.lstmlayers
        self._lstm_layer_dim = Config().args.lstmlayerdim
        self._input_reps = None
        self._empty_rep = None

    def init_extra_inputs(self, dim, num):
        self._params["bilstm"] = dy.BiRNNBuilder(self._lstm_layers, dim, self._lstm_layer_dim, self.model,
                                                 dy.LSTMBuilder)
        return num * self._lstm_layer_dim

    def init_features(self, features, train=False):
        self.init_cg()
        # TODO add e.g. 2 to x so as to skip the unknown and EMPTY_INDEX?
        embeddings = [[self._params[s][x] for s, xs in sorted(features.items()) for x in xs]]  # time-lists of vectors
        embeddings = [dy.concatenate(list(e)) for e in zip(*embeddings)]  # one list: join each time step to one vector
        bilstm = self._params["bilstm"]
        if train:
            bilstm.set_dropout(self._dropout)
        self._input_reps = bilstm.transduce(embeddings)
        self._empty_rep = dy.concatenate([self._params[s][EMPTY_INDEX] for s in sorted(features.keys())])

    def index_input(self, indices):
        return dy.concatenate([self._input_reps[i] if i > 0 else self._empty_rep for i in indices])

    def evaluate(self, features, train=False):
        return self.evaluate_mlp(features, train)

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
