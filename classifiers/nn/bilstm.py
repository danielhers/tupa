import dynet as dy

from nn.neural_network import NeuralNetwork
from parsing import config


class BiLSTM(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(*args, model_type=config.BILSTM_NN, **kwargs)
        self._input_reps = {}

    def init_features(self, features):
        if self.model is None:
            self.init_model()
        if not self._losses:
            dy.renew_cg()
        for suffix, values in sorted(features.items()):
            # TODO feed through BiLSTM
            self._input_reps[suffix] = [self._params[suffix][v] for v in values]

    def init_model(self):
        self.model = dy.Model()
        self._trainer = self._optimizer(self.model)
        input_dim = 0
        for suffix, param in sorted(self._input_params.items()):
            if not param.numeric and param.dim > 0:  # index feature
                p = self.model.add_lookup_parameters((param.size, param.dim))
                if param.init is not None:
                    p.init_from_array(param.init)
                self._params[suffix] = p
            input_dim += param.num * param.dim
        for i in range(1, self._layers + 1):
            in_dim = input_dim if i == 1 else self._layer_dim
            out_dim = self._layer_dim if i < self._layers else self.max_num_labels
            self._params["W%d" % i] = self.model.add_parameters((out_dim, in_dim), init=self._init)
            self._params["b%d" % i] = self.model.add_parameters(out_dim, init=self._init)

    def _generate_inputs(self, features):
        for suffix, values in sorted(features.items()):
            param = self._input_params[suffix]
            if param.numeric:
                yield dy.inputVector(values)
            elif param.indexed:
                yield dy.concatenate([self._input_reps[suffix][v] for v in values])
            elif param.dim > 0:
                yield dy.reshape(self._params[suffix].batch(values), (param.num * param.dim,))

    def evaluate(self, features, train=False):
        x = dy.concatenate(list(self._generate_inputs(features)))
        for i in range(1, self._layers + 1):
            W = dy.parameter(self._params["W%d" % i])
            b = dy.parameter(self._params["b%d" % i])
            if train and self._dropout:
                x = dy.dropout(x, self._dropout)
            x = self._activation(W * x + b)
        return dy.log_softmax(x, restrict=list(range(self.num_labels)))
