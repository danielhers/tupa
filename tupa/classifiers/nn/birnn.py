import dynet as dy
import numpy as np

from .constants import ACTIVATIONS, INITIALIZERS, RNNS, CategoricalParameter
from .mlp import MultilayerPerceptron
from .sub_model import SubModel
from .util import randomize_orthonormal
from ...model_util import MISSING_VALUE


class BiRNN(SubModel):
    def __init__(self, args, model, **kwargs):
        super().__init__(**kwargs)
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
                self.mlp.init_params(indexed_dim)
                randomize_orthonormal(*self.init_rnn_params(indexed_dim), activation=self.activation)
                if self.args.verbose > 3:
                    print("Initializing BiRNN: %s" % self)
            return indexed_num * self.lstm_layer_dim
        return 0

    def init_rnn_params(self, indexed_dim):
        rnn = dy.BiRNNBuilder(self.lstm_layers, self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                              self.lstm_layer_dim, self.model, self.rnn_builder())
        self.params["birnn"] = rnn
        return [p for f, b in rnn.builder_layers for r in (f, b) for l in r.get_parameters() for p in l]

    def init_features(self, embeddings, train=False):
        if self.params:
            inputs = [self.mlp.evaluate(e, train=train) for e in zip(*embeddings)]  # join each time step to a vector
            if self.args.verbose > 3:
                print("Transducing %d inputs with dropout %s" % (len(inputs), self.dropout if train else "disabled"))
            self.input_reps = self.transduce(inputs, train)
            self.empty_rep = dy.inputVector(np.zeros(self.lstm_layer_dim, dtype=float))

    def transduce(self, inputs, train):
        birnn = self.params["birnn"]
        if train:
            birnn.set_dropout(self.dropout)
        else:
            birnn.disable_dropout()
        return birnn.transduce(inputs[:self.max_length])

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
        ) if self.lstm_layer_dim and self.lstm_layers else []
        if self.args.verbose > 3:
            print("Saving BiRNN: %s" % self)
        return values

    def load_sub_model(self, d, *args, **kwargs):
        d = super().load_sub_model(d, *args, **kwargs)
        if d:
            self.args.lstm_layers = self.lstm_layers = d["lstm_layers"]
            self.args.lstm_layer_dim = self.mlp.output_dim = self.lstm_layer_dim = d["lstm_layer_dim"]
            self.args.embedding_layers = self.mlp.layers = self.embedding_layers = d["embedding_layers"]
            self.args.embedding_layer_dim = self.mlp.layer_dim = self.embedding_layer_dim = d["embedding_layer_dim"]
            self.args.max_length = self.max_length = d["max_length"]
            self.args.rnn = self.rnn_builder.string = d["rnn"]
            self.args.activation = self.mlp.activation.string = self.activation.string = d["activation"]
            self.args.init = self.mlp.init.string = self.init.string = d["init"]
            self.args.dropout = self.mlp.dropout = self.dropout = d["dropout"]
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
                self.indexed_dim, self.indexed_num, list(self.params.keys()))

    def sub_models(self):
        return [self.mlp]


class EmptyRNN(BiRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_params(self, indexed_dim, indexed_num):
        return 0

    def init_features(self, embeddings, train=False):
        pass

    def evaluate(self, indices):
        return []

    def save_sub_model(self, d, *args):
        return []

    def load_sub_model(self, d, *args, **kwargs):
        super().load_sub_model(d, *args)
        self.args.lstm_layers = self.lstm_layers = self.args.lstm_layer_dim = self.lstm_layer_dim = 0


class HighwayRNN(BiRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_rnn_params(self, indexed_dim):
        params = []
        for i in range(self.lstm_layers):
            for n in "f", "b":
                input_dim = self.lstm_layer_dim if i or self.embedding_layers else indexed_dim
                rnn = self.rnn_builder()(1, input_dim, self.lstm_layer_dim, self.model)
                self.params["rnn%d%s" % (i, n)] = rnn
                params += [p for l in rnn.get_parameters() for p in l]
                for p, dim in (("Wr", (self.lstm_layer_dim, input_dim + self.lstm_layer_dim)),
                               ("br", self.lstm_layer_dim), ("Wh", (self.lstm_layer_dim, input_dim))):
                    param = self.model.add_parameters(dim, init=self.init()())
                    self.params["%s%d%s" % (p, i, n)] = param
                    params.append(param)
        return params

    def transduce(self, inputs, train):
        xs = inputs[:self.max_length]
        if not xs:
            return []
        for i in range(self.lstm_layers):
            for n, d in ("f", 1), ("b", -1):
                hs = self.params["rnn%d%s" % (i, n)].initial_state().transduce(xs[::d])
                Wr, br, Wh = [dy.parameter(self.params["%s%d%s" % (p, i, n)]) for p in ("Wr", "br", "Wh")]
                rs = [dy.logistic(Wr * dy.concatenate([h, x]) + br) for h, x in zip(hs[:-1], xs[::d][1:])]
                xs = [hs[0]] + [dy.cmult(r, h) + dy.cmult(1 - r, Wh * x) for r, h, x in zip(rs, hs[1:], xs[::d][1:])]
        if train:
            x = dy.dropout_dim(dy.concatenate(xs, 1), 1, self.dropout)
            xs = [dy.pick(x, i, 1) for i in range(len(xs))]
        return xs
