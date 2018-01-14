import dynet as dy

from .constants import ACTIVATIONS, INITIALIZERS, CategoricalParameter
from .sub_model import SubModel


class MultilayerPerceptron(SubModel):
    def __init__(self, args, model, layers=None, layer_dim=None, output_dim=None, num_labels=None, params=None,
                 save_path=()):
        super().__init__(params=params, save_path=save_path)
        self.args = args
        self.model = model
        self.total_layers = self.layers = self.args.layers if layers is None else layers
        self.layer_dim = self.args.layer_dim if layer_dim is None else layer_dim
        self.output_dim = self.args.output_dim if output_dim is None else output_dim
        self.activation = CategoricalParameter(ACTIVATIONS, self.args.activation)
        self.init = CategoricalParameter(INITIALIZERS, self.args.init)
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        if self.num_labels:
            self.total_layers += 1
        self.input_dim = None

    def init_params(self, input_dim):
        self.input_dim = input_dim
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
        if self.args.verbose > 3:
            print("Initializing MLP: %s" % self)

    def evaluate(self, inputs, train=False):
        x = dy.concatenate(list(inputs))
        dim = x.dim()[0][0]
        assert dim == self.input_dim, "Input dim mismatch: %d != %d" % (dim, self.input_dim)
        if self.args.verbose > 3:
            print(self)
        for i in range(self.total_layers):
            if self.args.verbose > 3:
                print(x.npvalue().tolist())
            try:
                if train and self.dropout:
                    x = dy.dropout(x, self.dropout)
                W, b = [dy.parameter(self.params[prefix + str(i)]) for prefix in ("W", "b")]
                x = self.activation()(W * x + b)
            except ValueError as e:
                raise ValueError("Error in evaluating layer %d of %d" % (i + 1, self.total_layers)) from e
        if self.args.verbose > 3:
            print(x.npvalue().tolist())
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
        if self.args.verbose > 3:
            print("Saving MLP: %s" % self)
        return values

    def load_sub_model(self, d, *args):
        d = super().load_sub_model(d, *args)
        self.args.layers = self.layers = d["layers"]
        self.total_layers = d["total_layers"]
        self.args.layer_dim = self.layer_dim = d["layer_dim"]
        self.args.output_dim = self.output_dim = d["output_dim"]
        self.args.activation = self.activation.string = d["activation"]
        self.args.init = self.init.string = d["init"]
        self.args.dropout = self.dropout = d["dropout"]
        self.num_labels = d["num_labels"]
        self.input_dim = d["input_dim"]
        self.verify_dims()
        if self.args.verbose > 3:
            print("Loading MLP: %s" % self)

    def verify_dims(self):
        self.verify_dim("input_dim", self.params["W0"].as_array().shape[1])
        self.verify_dim("output_dim", self.params["W" + str(self.layers - 1)].as_array().shape[0])
        if self.num_labels:
            self.verify_dim("num_labels", self.params["W" + str(self.layers)].as_array().shape[0])
    
    def verify_dim(self, attr, val):
        expected = getattr(self, attr)
        assert val == expected, "%s %s: %d, expected: %d" % ("/".join(self.save_path), attr, val, expected)

    def __str__(self):
        return "%s layers: %d, total_layers: %d, layer_dim: %d, output_dim: %d, activation: %s, init: %s, " \
               "dropout: %f, num_labels: %s, input_dim: %d, params: %s" % (
                "/".join(self.save_path), self.layers, self.total_layers, self.layer_dim, self.output_dim,
                self.activation, self.init, self.dropout, self.num_labels, self.input_dim, list(self.params.keys()))
