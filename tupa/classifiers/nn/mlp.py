import dynet as dy

from .constants import ACTIVATIONS, INITIALIZERS, CategoricalParameter


class MultilayerPerceptron(object):
    def __init__(self, args, model, global_params, layers=None, layer_dim=None, output_dim=None,
                 num_labels=None, params=None, **kwargs):
        del params
        self.args = args
        self.model = model
        self.params = global_params
        self.total_layers = self.layers = self.args.layers if layers is None else layers
        self.layer_dim = self.args.layer_dim if layer_dim is None else layer_dim
        self.output_dim = self.args.output_dim if output_dim is None else output_dim
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.dropout = self.args.dropout
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.num_labels = num_labels
        if self.num_labels:
            self.total_layers += 1
        self.key_args = kwargs

    def init_params(self, input_dim):
        hidden_dim = (self.layers - 1) * [self.layer_dim]
        i_dim = [input_dim] + hidden_dim
        o_dim = hidden_dim + [self.output_dim]
        if self.num_labels:  # Adding another layer at the top
            i_dim.append(self.output_dim)
            o_dim.append(self.num_labels)
        self.params.update((key(prefix, i, **self.key_args), self.model.add_parameters(dims[i], init=self.init()()))
                           for prefix, dims in (("W", list(zip(o_dim, i_dim))), ("b", o_dim))
                           for i, dim in enumerate(dims))

    def evaluate(self, inputs, train=False):
        x = dy.concatenate(list(inputs))
        for i in range(self.total_layers):
            try:
                if train and self.dropout:
                    x = dy.dropout(x, self.dropout)
                W, b = [dy.parameter(self.params[key(prefix, i, **self.key_args)]) for prefix in ("W", "b")]
                x = self.activation()(W * x + b)
            except ValueError as e:
                raise ValueError("Error in evaluating layer %d of %d" % (i + 1, self.total_layers)) from e
        return x


def key(prefix, i, offset=0, suffix1="", suffix2=()):
    return (prefix + suffix1, i + offset) + suffix2
