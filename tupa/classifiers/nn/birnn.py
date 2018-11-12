import dynet as dy
import numpy as np

from .constants import ACTIVATIONS, INITIALIZERS, RNNS, CategoricalParameter
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
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.rnn_builder = CategoricalParameter(RNNS, self.args.rnn)
        self.input_reps = self.empty_rep = self.indexed_dim = self.indexed_num = None
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
                randomize_orthonormal(*self.init_rnn_params(indexed_dim), activation=self.activation)
                self.config.print("Initializing BiRNN: %s" % self, level=4)

    def init_rnn_params(self, indexed_dim):
        rnn = dy.BiRNNBuilder(self.lstm_layers, self.lstm_layer_dim if self.embedding_layers else indexed_dim,
                              self.lstm_layer_dim, self.model, self.rnn_builder())
        self.params["birnn"] = rnn
        return [p for f, b in rnn.builder_layers for r in (f, b) for l in r.get_parameters() for p in l]

    def init_features(self, embeddings, train=False):
        """
        Set the value of self.input_reps (and self.empty_rep) given embeddings for the whole input sequence
        :param embeddings: list of [(key, list of vectors embeddings per time step)] per feature
        :param train: are we training now?
        """
        if self.params:
            keys, embeddings = zip(*embeddings)
            inputs = [self.mlp.evaluate(zip(keys, es), train=train) for es in zip(*embeddings)]  # join each time step
            self.config.print("Transducing %d inputs with dropout %s" %
                              (len(inputs), self.dropout if train else "disabled"), level=4)
            self.input_reps = self.transduce(inputs, train)
            expected = min(len(inputs), self.max_length or np.iinfo(int).max)
            assert len(self.input_reps) == expected, \
                "transduce() returned incorrect number of elements: %d != %d" % (len(self.input_reps), expected)
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
            return [("/".join(self.save_path),
                     self.empty_rep if i == MISSING_VALUE else self.get_representation(i)) for i in indices]
        return []

    def get_representation(self, i):
        return self.input_reps[min(i, self.max_length - 1)]

    def transition(self, action):
        pass

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
            self.args.rnn = self.rnn_builder.string = d["rnn"]
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
                self.embedding_layer_dim, self.max_length, self.rnn_builder, self.activation, self.init, self.dropout,
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


class HighwayRNN(BiRNN):
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
                Wr, br, Wh = [self.params["%s%d%s" % (p, i, n)] for p in ("Wr", "br", "Wh")]
                hs_ = self.params["rnn%d%s" % (i, n)].initial_state().transduce(xs[::d])
                hs = [hs_[0]]
                for t in range(1, len(hs_)):
                    r = dy.logistic(Wr * dy.concatenate([hs[t - 1], xs[t]]) + br)
                    hs.append(dy.cmult(r, hs_[t]) + dy.cmult(1 - r, Wh * xs[t]))
                xs = hs
                if train:
                    x = dy.dropout_dim(dy.concatenate(xs, 1), 1, self.dropout)
                    xs = [dy.pick(x, i, 1) for i in range(len(xs))]
        return xs


class HierarchicalBiRNN(BiRNN):
    RNN_NAMES = "lrnn", "rrnn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_reps = {}

    def add_node(self, i):
        self.internal_reps[i] = [self.params[n].initial_state() for n in self.RNN_NAMES]

    def add_edge(self, i, j, direction):
        self.internal_reps[i][direction] = self.internal_reps[i][direction].add_input(self.get_representation(j))

    def transition(self, action):
        if self.params:
            if action.node:
                self.add_node(action.node.index)
            if action.edge:
                self.add_edge(action.edge.parent.index, action.edge.child.index, "RIGHT" in action.type)

    def init_features(self, embeddings, train=False):
        super().init_features(embeddings, train)
        self.internal_reps.clear()
        if self.params:
            self.add_node(0)  # Root

    def init_rnn_params(self, indexed_dim):
        params = super().init_rnn_params(indexed_dim)
        for name in self.RNN_NAMES:
            rnn = self.rnn_builder()(1, self.lstm_layer_dim, self.lstm_layer_dim / 2, self.model)
            self.params[name] = rnn
            params += [p for l in rnn.get_parameters() for p in l]
        return params

    def get_representation(self, i):
        states = self.internal_reps.get(i)
        if states:
            return dy.concatenate([s.output() or self.empty_rep[:self.lstm_layer_dim / 2] for s in states])
        return super().get_representation(i - 1)  # Terminal index is (node index-1)
