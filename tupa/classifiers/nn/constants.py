LOSSES = ("softmax", "max_margin")
DEFAULT_LOSS = "softmax"

TRAINERS = {
    "sgd": "SimpleSGDTrainer",
    "cyclic": "CyclicalSGDTrainer",
    "momentum": "MomentumSGDTrainer",
    "adagrad": "AdagradTrainer",
    "adadelta": "AdadeltaTrainer",
    "rmsprop": "RMSPropTrainer",
    "adam": "AdamTrainer",
    "amsgrad": "AmsgradTrainer",
}
DEFAULT_TRAINER = "sgd"
EXTRA_TRAINER = "amsgrad"
TRAINER_LEARNING_RATE_PARAM_NAMES = {k: "learning_rate" for k in TRAINERS}
TRAINER_LEARNING_RATE_PARAM_NAMES.update(cyclic="learning_rate_max")
TRAINER_KWARGS = {}  # "adam": dict(beta_2=0.9)}

INITIALIZERS = {
    # "saxe": "SaxeInitializer",
    "glorot_uniform": "GlorotInitializer",
    "normal": "NormalInitializer",
}
DEFAULT_INITIALIZER = "glorot_uniform"

ACTIVATIONS = {
    "cube": "cube",
    "tanh": "tanh",
    "sigmoid": "logistic",
    "relu": "rectify",
}
DEFAULT_ACTIVATION = "relu"

RNNS = {
    # "simple": "SimpleRNNBuilder",
    "gru": "GRUBuilder",
    "lstm": "LSTMBuilder",
    "vanilla_lstm": "VanillaLSTMBuilder",
    "compact_vanilla_lstm": "CompactVanillaLSTMBuilder",
    "coupled_lstm": "CoupledLSTMBuilder",
    "fast_lstm": "FastLSTMBuilder",
}
DEFAULT_RNN = "lstm"


class CategoricalParameter:
    def __init__(self, values, string):
        self._value = self._string = None
        self.values = values
        self.string = string

    def __call__(self):
        import dynet as dy
        return getattr(dy, self._value)

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, s):
        self._string = s
        self._value = self.values.get(s)

    def __str__(self):
        return self._string
