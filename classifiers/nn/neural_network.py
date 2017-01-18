from keras import backend as K
from keras import regularizers
from keras.models import model_from_json

from classifiers.classifier import Classifier, ClassifierProperty
from features.feature_params import MISSING_VALUE
from parsing.config import Config
from parsing.model_util import load_dict, save_dict


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureEnumerator.
    """

    def __init__(self, *args, input_params=None, model=None):
        """
        Create a new untrained NN
        :param input_params: dict of feature type name -> FeatureInformation
        """
        super(NeuralNetwork, self).__init__(*args)
        assert input_params is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self.max_num_labels = Config().args.max_labels
            self._layers = Config().args.layers
            self._layer_dim = Config().args.layer_dim
            self._activation = (lambda x: x*x*x) if Config().args.activation == "cube" else Config().args.activation
            self._init = Config().args.init
            self._num_labels = self.num_labels
            self._minibatch_size = Config().args.minibatch_size
            self._nb_epochs = Config().args.epochs
            self._dropout = Config().args.dropout
            self._optimizer = Config().args.optimizer
            self._loss = (lambda t, p: K.sum(K.maximum(0., 1.-p*t+p*(1.-t)))) if Config().args.loss == "max_margin" else Config().args.loss
            self._regularizer = (lambda: None) if Config().args.regularizer is None else \
                (lambda: regularizers.l1l2(Config().args.regularization, Config().args.regularization)) if Config().args.regularizer == "l1l2" else \
                (lambda: regularizers.get(Config().args.regularizer, {"l": Config().args.regularization}))
            self.input_params = input_params
            self.model = None
        self._batch_size = Config().args.batch_size
        self._item_index = 0
        self._iteration = 0

    def init_model(self):
        raise NotImplementedError()

    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss={"out": self._loss})

    @property
    def input_dim(self):
        return sum(f.num * f.dim for f in self.input_params.values())

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def save(self):
        """
        Save all parameters to file
        :param filename: file to save to
        """
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "is_frozen": self.is_frozen,
            "iteration": self._iteration,
        }
        save_dict(self.filename, d)
        self.init_model()
        with open(self.filename + ".json", "w") as f:
            f.write(self.model.to_json())
        try:
            self.model.save_weights(self.filename + ".h5", overwrite=True)
        except ValueError as e:
            print("Failed saving model weights: %s" % e)

    def load(self):
        """
        Load all parameters from file
        """
        d = load_dict(self.filename)
        model_type = d.get("type")
        assert model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.is_frozen = d["is_frozen"]
        self._iteration = d.get("iteration", 0)
        with open(self.filename + ".json") as f:
            self.model = model_from_json(f.read())
        try:
            self.model.load_weights(self.filename + ".h5")
        except KeyError as e:
            print("Failed loading model weights: %s" % e)
        self.compile()

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)

    def get_classifier_properties(self):
        return super(NeuralNetwork, self).get_classifier_properties() + \
               (ClassifierProperty.trainable_after_saving,)
