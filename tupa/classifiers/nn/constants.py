LOSSES = ("softmax", "max_margin")
DEFAULT_LOSS = "softmax"

TRAINERS = {
    "sgd": "SGD",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "rmsprop": "RMSprop",
    "adam": "Adam",
    "sparseadam": "SparseAdam",
    "adamax": "Adamax",
    "asgd": "ASGD",
    "rprop": "Rprop",
}
DEFAULT_TRAINER = "sgd"
EXTRA_TRAINER = "adam"  # TODO amsgrad=True
TRAINER_LEARNING_RATE_PARAM_NAMES = {k: "lr" for k in TRAINERS}
TRAINER_KWARGS = {}  # "adam": dict(beta_2=0.9)}

INITIALIZERS = {
    "orthogonal": "orthogonal",
    "glorot_uniform": "xavier_uniform",
    "normal": "normal",
}
DEFAULT_INITIALIZER = "glorot_uniform"

ACTIVATIONS = {
    "tanh": "tanh",
    "hardtanh": "hardtanh",
    "sigmoid": "sigmoid",
    "logsigmoid": "logsigmoid",
    "relu": "relu",
}
DEFAULT_ACTIVATION = "relu"

RNNS = {
    "gru": "GRU",
    "lstm": "LSTM",
}
DEFAULT_RNN = "lstm"


class CategoricalParameter:
    def __init__(self, values, string, module):
        self._value = self._string = None
        self.values = values
        self.string = string
        self.module = module

    def __call__(self):
        return getattr(self.module, self._value)

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, s):
        self._string = s
        self._value = self.values.get(s)

    def __str__(self):
        return self._string


class Trainer(CategoricalParameter):
    def __init__(self, string):
        import torch.optim as optim
        super().__init__(values=TRAINERS, string=string, module=optim)
        self.string = string


class Activation(CategoricalParameter):
    def __init__(self, string):
        import torch.nn as nn
        super().__init__(values=ACTIVATIONS, string=string, module=nn)
        self.string = string


class Initializer(CategoricalParameter):
    def __init__(self, string):
        import torch.nn.init as init
        super().__init__(values=INITIALIZERS, string=string, module=init)
        self.string = string


class RNN(CategoricalParameter):
    def __init__(self, string):
        import torch.nn as nn
        super().__init__(values=RNNS, string=string, module=nn)
        self.string = string
