import re
from itertools import groupby

import torch
import torch.nn as nn

from .constants import Activation, Initializer
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
        self.activation = Activation(self.args.activation)
        self.init = Initializer(self.args.init)
        self.dropout = self.args.dropout
        self.gated = 1 if self.args.gated is None else self.args.gated  # None means --gated was given with no argument
        self.num_labels = num_labels
        self.input_dim = self.input_keys = self.weights = None

    @property
    def total_layers(self):
        return self.layers + (1 if self.num_labels else 0)

    def init_params(self, input_dim):
        assert input_dim, "Zero input dimension to MLP"
        self.input_dim = input_dim
        if self.layers > 0 and "W0" not in self.params:
            hidden_dim = (self.layers - 1) * [self.layer_dim]
            i_dim = [input_dim] + hidden_dim
            o_dim = hidden_dim + [self.output_dim]
            if self.num_labels:  # Adding another layer at the top
                i_dim.append(self.output_dim)
                o_dim.append(self.num_labels)
            self.params.update((prefix + str(i), self.init_layer(dims[i]))
                               for prefix, dims in (("W", list(zip(o_dim, i_dim))), ("b", o_dim))
                               for i, dim in enumerate(dims))
            if self.dropout:
                self.params.update(("d%d" % i, nn.Dropout(self.dropout)) for i in range(len(o_dim)))
            self.verify_dims()
            randomize_orthonormal(**self.params)
            self.config.print("Initializing MLP: %s" % self, level=4)

    def init_layer(self, dims):
        m = nn.Linear(*dims)
        self.init()(m)
        return m

    def evaluate(self, inputs, train=False):
        """
        Apply all MLP layers to concatenated input
        :param inputs: (key, vector) per feature type
        :param train: are we training now?
        :return: output vector of size self.output_dim
        """
        for module in self.params.values():
            module.train(mode=train)
        input_keys, inputs = list(map(list, zip(*list(inputs))))
        if self.input_keys:
            assert input_keys == self.input_keys, "Got:     %s\nBut expected input keys: %s" % (
                self.input_keys_str(input_keys), self.input_keys_str(self.input_keys))
        else:
            self.input_keys = input_keys
        if self.gated:
            gates = self.params.get("gates")
            if gates is None:  # FIXME attention weights should not be just parameters, but based on biaffine product?
                gates = self.params["gates"] = nn.Linear(len(inputs), self.gated)
                nn.init.constant(gates, 1.0)
            input_dims = [i.size()[0] for i in inputs]
            max_dim = max(input_dims)
            # Pad with zeros to get uniform dim
            x = torch.cat([torch.cat([i, torch.zeros(max_dim - d, dtype=torch.float64)])
                           if d < max_dim else i for i, d in zip(inputs, input_dims)], 1) * gates
            # Possibly multiple "attention heads" -- concatenate outputs to one vector
            inputs = [x.view(-1, x.size()[0] * x.size()[1],)]
        x = torch.cat(inputs)
        assert len(x.size()[0]) == 1, "Input should be a vector, but has dimension " + str(x.size())
        dim = x.size()[0]
        if self.input_dim:
            assert dim == self.input_dim, "Input dim mismatch: %d != %d" % (dim, self.input_dim)
        else:
            self.init_params(dim)
        self.config.print(self, level=4)
        if self.total_layers:
            if self.weights is None:
                self.weights = [[self.params[prefix + str(i)] for prefix in ("W", "b")]
                                for i in range(self.total_layers)]
            for i, (W, b) in enumerate(self.weights):
                self.config.print(lambda: x.npvalue().tolist(), level=4)
                try:
                    dropout = self.params.get("d%d" % i)
                    if dropout:
                        x = dropout(x)
                    x = self.activation()(W * x + b)
                except ValueError as e:
                    raise ValueError("Error in evaluating layer %d of %d" % (i + 1, self.total_layers)) from e
        self.config.print(x, level=4)
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
            ("gated", self.gated),
            ("num_labels", self.num_labels),
            ("input_dim", self.input_dim),
            ("input_keys", self.input_keys),
        )
        self.config.print("Saving MLP: %s" % self, level=4)
        return values

    def load_sub_model(self, d, *args, **kwargs):
        d = super().load_sub_model(d, *args, **kwargs)
        if not d:
            self.config.print("Skipped empty MLP: %s" % self, level=4)
            return
        self.args.layers = self.layers = d["layers"]
        self.args.layer_dim = self.layer_dim = d["layer_dim"]
        self.args.output_dim = self.output_dim = d["output_dim"]
        self.args.activation = self.activation.string = d["activation"]
        self.args.init = self.init.string = d["init"]
        self.args.dropout = self.dropout = d["dropout"]
        self.args.gated = self.gated = d.get("gated", 0)
        self.num_labels = d["num_labels"]
        self.input_dim = d["input_dim"]
        self.input_keys = d.get("input_keys")
        self.verify_dims()
        self.config.print("Loading MLP: %s" % self, level=4)

    def verify_dims(self):
        assert self.params, "No MLP parameters found for %s (classifier never used)" % ("/".join(self.save_path))
        if self.layers > 0:
            self.verify_dim("input_dim", self.params["W0"].size()[1])
            self.verify_dim("output_dim", self.params["W" + str(self.layers - 1)].size()[0])
            if self.num_labels:
                self.verify_dim("num_labels", self.params["W" + str(self.layers)].size()[0])
    
    def verify_dim(self, attr, val):
        expected = getattr(self, attr)
        assert val == expected, "%s %s: %d, expected: %d" % ("/".join(self.save_path), attr, val, expected)

    def invalidate_caches(self):
        self.weights = None

    @staticmethod
    def input_keys_str(input_keys):
        return None if input_keys is None else " ".join("%s:%d" % (k, len(list(l))) for k, l in groupby(input_keys))

    def __str__(self):
        try:
            return "%s layers: %d, total_layers: %d, layer_dim: %d, output_dim: %d, activation: %s, init: %s, " \
                   "dropout: %f, gated: %d, num_labels: %s, input_dim: %d, input_keys: %s, params: %s" % (
                    "/".join(self.save_path), self.layers, self.total_layers, self.layer_dim, self.output_dim,
                    self.activation, self.init, self.dropout, self.gated, self.num_labels, self.input_dim,
                    self.input_keys_str(self.input_keys), list(self.params.keys()))
        except TypeError:
            return "not initialized"
