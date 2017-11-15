import dynet as dy
import numpy as np

from .constants import ACTIVATIONS, INITIALIZERS, RNNS, CategoricalParameter
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from ...features.feature_params import MISSING_VALUE


class BiRNN(SubModel):
    def __init__(self, args, model, save_path=None):
        super().__init__(save_path=save_path)
        self.args = args
        self.model = model
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
        self.mlp = MultilayerPerceptron(self.args, self.model, params=self.params, layers=self.embedding_layers,
                                        layer_dim=self.embedding_layer_dim, output_dim=self.lstm_layer_dim)
        self.save_path = save_path

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
        self.params["birnn"] = dy.BiRNNBuilder(self.lstm_layers,
                                               self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                                               self.lstm_layer_dim, self.model, self.rnn_builder())
        return indexed_num * self.lstm_layer_dim

    def init_features(self, embeddings, train=False):
        if self.empty or not self.params:
            return
        inputs = [self.mlp.evaluate(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
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
        return [] if self.empty or not self.params else \
            [self.empty_rep if i == MISSING_VALUE else self.input_reps[min(i, self.max_length - 1)] for i in indices]

    def save_sub_model(self, d, *args):
        super().save_sub_model(
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
        )

    def load_sub_model(self, d):
        d, param_keys = super().load_sub_model(d)
        self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
        self.args.lstm_layer_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
        self.args.embedding_layers = self.embedding_layers = d["embedding_layers"]
        self.args.embedding_layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
        self.args.max_length = self.max_length = d["max_length"]
        self.args.rnn = self.rnn_builder.string = d["rnn"]
        self.args.activation = self.activation.string = d["activation"]
        self.args.init = self.init.string = d["init"]
        self.args.dropout = self.dropout = d["dropout"]
        return param_keys
