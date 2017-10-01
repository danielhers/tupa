from functools import partial

import dynet as dy

TRAINERS = {
    "sgd": (dy.SimpleSGDTrainer, "learning_rate"),
    "cyclic": (dy.CyclicalSGDTrainer, "learning_rate_max"),
    "momentum": (dy.MomentumSGDTrainer, "learning_rate"),
    "adagrad": (dy.AdagradTrainer, "learning_rate"),
    "adadelta": (dy.AdadeltaTrainer, None),
    "rmsprop": (dy.RMSPropTrainer, "learning_rate"),
    "adam": (partial(dy.AdamTrainer, beta_2=0.9), "alpha"),
}
DEFAULT_TRAINER = "adam"

INITIALIZERS = {
    "glorot_uniform": dy.GlorotInitializer(),
    "normal": dy.NormalInitializer(),
    # "uniform": dy.UniformInitializer(1),
    # "const": dy.ConstInitializer(0),
}
DEFAULT_INITIALIZER = "glorot_uniform"

ACTIVATIONS = {
    # "square": dy.square,
    "cube": dy.cube,
    "tanh": dy.tanh,
    "sigmoid": dy.logistic,
    "relu": dy.rectify,
}
DEFAULT_ACTIVATION = "relu"

RNNS = {
    # "simple": dy.SimpleRNNBuilder,
    "gru": dy.GRUBuilder,
    "lstm": dy.LSTMBuilder,
    "vanilla_lstm": dy.VanillaLSTMBuilder,
    "compact_vanilla_lstm": dy.CompactVanillaLSTMBuilder,
    "coupled_lstm": dy.CoupledLSTMBuilder,
    "fast_lstm": dy.FastLSTMBuilder,
}
DEFAULT_RNN = "lstm"


class CategoricalParameter(object):
    def __init__(self, values, string):
        self._value = self._string = None
        self.values = values
        self.string = string

    def __call__(self):
        return self._value

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, s):
        self._string = s
        self._value = self.values.get(s)

    def __str__(self):
        return self._string
