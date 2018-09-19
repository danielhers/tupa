import dynet as dy

from .constants import ACTIVATIONS, INITIALIZERS, CategoricalParameter
from .sub_model import SubModel
from .util import randomize_orthonormal


class MultilayerPerceptron(SubModel):
    def __init__(self, config, args, model, layers=None, layer_dim=None, output_dim=None, num_labels=None, params=None,
                 **kwargs):
        super().__init__(params=params, **kwargs)
        self.config = config
        self.args = args
        self.model = model
        self.layers = self.args.layers if layers is None else layers
        self.layer_dim = self.args.layer_dim if layer_dim is None else layer_dim
        self.output_dim = self.args.output_dim if output_dim is None else output_dim
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.input_dim = self.weights = None

    @property
    def total_layers(self):
        return self.layers + (1 if self.num_labels else 0)

    def init_params(self, input_dim):
        assert input_dim, "Zero input dimension to MLP"
        self.input_dim = input_dim
        if self.layers > 0:
            if self.params:
                W0 = self.params["W0"]
                value = W0.as_array()
                old_input_dim = value.shape[1]
                extra_dim = self.input_dim - old_input_dim
                if extra_dim:  # Duplicate suffix of input dimension to accommodate extra input
                    assert extra_dim > 0, "%s: input dimension reduced from %d to %d" % (
                        "/".join(self.save_path), old_input_dim, self.input_dim)
                    value[:, -extra_dim:] /= 2
                    W0.set_value(value)
                    extra_value = value[:, -extra_dim:]
                    self.params["W0+"] = extra = self.model.add_parameters(extra_value.shape)
                    extra.set_value(extra_value)
            else:
                hidden_dim = (self.layers - 1) * [self.layer_dim]
                i_dim = [input_dim] + hidden_dim
                o_dim = hidden_dim + [self.output_dim]
                if self.num_labels:  # Adding another layer at the top
                    i_dim.append(self.output_dim)
                    o_dim.append(self.num_labels)
                self.params.update((prefix + str(i), self.model.add_parameters(dims[i], init=self.init()()))
                                   for prefix, dims in (("W", list(zip(o_dim, i_dim))), ("b", o_dim))
                                   for i, dim in enumerate(dims))
                self.verify_dims()
                randomize_orthonormal(*self.params.values(), activation=self.activation)
                self.config.print("Initializing MLP: %s" % self, level=4)

    def evaluate(self, inputs, train=False):
        x = dy.concatenate(list(inputs))  # TODO add interaction terms for biaffine attention
        dim = x.dim()[0][0]
        assert dim == self.input_dim, "Input dim mismatch: %d != %d" % (dim, self.input_dim)
        self.config.print(self, level=4)
        if self.total_layers:
            if self.weights is None:
                self.weights = [[self.params[prefix + str(i)] for prefix in ("W", "b")]
                                for i in range(self.total_layers)]
                if self.weights[0][0].dim()[0][1] < dim:  # number of columns in W0
                    self.weights[0][0] = dy.concatenate_cols([self.weights[0][0], self.params["W0+"]])
            for i, (W, b) in enumerate(self.weights):
                self.config.print(lambda: x.npvalue().tolist(), level=4)
                try:
                    if train and self.dropout:
                        x = dy.dropout(x, self.dropout)
                    x = self.activation()(W * x + b)
                except ValueError as e:
                    raise ValueError("Error in evaluating layer %d of %d" % (i + 1, self.total_layers)) from e
        self.config.print(lambda: x.npvalue().tolist(), level=4)
        return x

    def save_sub_model(self, d, *args):
        self.verify_dims()
        values = super().save_sub_model(
            d,
            ("layers", self.layers),
            ("total_layers", self.total_layers),
            ("layer_dim", self.layer_dim),
            ("output_dim", self.output_dim),
            ("activation", str(self.activation)),
            ("init", str(self.init)),
            ("dropout", self.dropout),
            ("num_labels", self.num_labels),
            ("input_dim", self.input_dim),
        )
        self.config.print("Saving MLP: %s" % self, level=4)
        return values

    def load_sub_model(self, d, *args, **kwargs):
        d = super().load_sub_model(d, *args, **kwargs)
        self.args.layers = self.layers = d["layers"]
        self.args.layer_dim = self.layer_dim = d["layer_dim"]
        self.args.output_dim = self.output_dim = d["output_dim"]
        self.args.activation = self.activation.string = d["activation"]
        self.args.init = self.init.string = d["init"]
        self.args.dropout = self.dropout = d["dropout"]
        self.num_labels = d["num_labels"]
        self.input_dim = d["input_dim"]
        self.verify_dims()
        self.config.print("Loading MLP: %s" % self, level=4)

    def verify_dims(self):
        if self.layers > 0:
            self.verify_dim("input_dim", sum(W.as_array().shape[1] for W in map(self.params.get, ("W0", "W0+")) if W))
            self.verify_dim("output_dim", self.params["W" + str(self.layers - 1)].as_array().shape[0])
            if self.num_labels:
                self.verify_dim("num_labels", self.params["W" + str(self.layers)].as_array().shape[0])
    
    def verify_dim(self, attr, val):
        expected = getattr(self, attr)
        assert val == expected, "%s %s: %d, expected: %d" % ("/".join(self.save_path), attr, val, expected)

    def invalidate_caches(self):
        self.weights = None

    def __str__(self):
        return "%s layers: %d, total_layers: %d, layer_dim: %d, output_dim: %d, activation: %s, init: %s, " \
               "dropout: %f, num_labels: %s, input_dim: %d, params: %s" % (
                "/".join(self.save_path), self.layers, self.total_layers, self.layer_dim, self.output_dim,
                self.activation, self.init, self.dropout, self.num_labels, self.input_dim, list(self.params.keys()))
