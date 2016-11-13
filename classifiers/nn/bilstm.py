import dynet as dy

from nn.neural_network import NeuralNetwork
from parsing import config


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, lstm_layers=1, lstm_layer_dim=100, **kwargs):
        """
        :param layers: number of LSTM hidden layers
        :param layer_dim: size of LSTM hidden layer
        """
        super(BiLSTM, self).__init__(*args, model_type=config.BILSTM_NN, **kwargs)
        self._lstm_layers = lstm_layers
        self._lstm_layer_dim = lstm_layer_dim
        self._input_reps = {}

    def init_extra_inputs(self, suffix, param):
        if param.indexed:
            self._params["bilstm_%s" % suffix] = dy.BiRNNBuilder(self._lstm_layers, param.dim, self._lstm_layer_dim,
                                                                 self.model, dy.LSTMBuilder)
        return param.num * (self._lstm_layer_dim if param.indexed else param.dim)

    def init_features(self, features, train=False):
        self.init_cg()
        for suffix, values in sorted(features.items()):
            bilstm = self._params["bilstm_%s" % suffix]
            if train:
                bilstm.set_dropout(self._dropout)
            embeddings = [self._params[suffix][v] for v in values]
            self._input_reps[suffix] = bilstm.transduce(embeddings)

    def index_input(self, suffix, param, values):
        return dy.concatenate([self._input_reps[suffix][v] for v in values]) if param.indexed else None

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
