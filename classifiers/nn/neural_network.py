from keras import backend as K
from keras import regularizers
from keras.models import model_from_json

from classifiers.classifier import Classifier
from parsing.model_util import load_dict, save_dict


class NeuralNetwork(Classifier):
    """
    Neural network to be used by the parser for action classification. Uses dense features.
    Keeps weights in constant-size matrices. Does not allow adding new features on-the-fly.
    Allows adding new labels on-the-fly, but requires pre-setting maximum number of labels.
    Expects features from FeatureEnumerator.
    """

    def __init__(self, labels, model_type, feature_params=None, model=None,
                 layers=1, layer_dim=100, activation="tanh", normalize=False,
                 init="glorot_normal", max_num_labels=100, batch_size=10,
                 minibatch_size=200, nb_epochs=5, dropout=0,
                 optimizer="adam", loss="categorical_crossentropy",
                 regularizer="l2", regularization=1e-8):
        """
        Create a new untrained NN or copy the weights from an existing one
        :param labels: a list of labels that can be updated later to add a new label
        :param feature_params: dict of feature type name -> FeatureInformation
        :param model: if given, copy the weights (from a trained model)
        :param layers: number of hidden layers
        :param layer_dim: size of hidden layer
        :param activation: activation function at hidden layers
        :param normalize: perform batch normalization after each layer?
        :param init: initialization type for hidden layers
        :param max_num_labels: since model size is fixed, set maximum output size
        :param batch_size: fit model every this many items
        :param minibatch_size: batch size for SGD
        :param nb_epochs: number of epochs for SGD
        :param dropout: dropout to apply to input layer
        :param optimizer: algorithm to use for optimization
        :param loss: objective function to use for optimization
        :param regularizer: regularization type (None, l1, l2 or l1l2)
        :param regularization: regularization parameter lambda
        """
        super(NeuralNetwork, self).__init__(model_type=model_type, labels=labels, model=model)
        assert feature_params is not None or model is not None
        if self.is_frozen:
            self.model = model
        else:
            self.max_num_labels = max_num_labels
            self._layers = layers
            self._layer_dim = layer_dim
            self._activation = activation
            self._normalize = normalize
            self._init = init
            self._num_labels = self.num_labels
            self._minibatch_size = minibatch_size
            self._nb_epochs = nb_epochs
            self._dropout = dropout
            self._optimizer = optimizer
            if loss == "max_margin":
                self._loss = lambda true, pred: K.sum(K.maximum(0., 1. - pred*true + pred*(1. - true)))
            else:
                self._loss = loss
            if regularizer is None:
                self._regularizer = None
            elif regularizer == "l1l2":
                self._regularizer = regularizers.l1l2(regularization, regularization)
            else:
                self._regularizer = regularizers.get(regularizer, {"l": regularization})
            self.feature_params = feature_params
            self.model = None
        self._batch_size = batch_size
        self._item_index = 0
        self._iteration = 0

    def init_model(self):
        raise NotImplementedError()

    def compile(self):
        self.model.compile(optimizer=self._optimizer, loss={"out": self._loss})

    @property
    def input_dim(self):
        return sum(f.num * f.dim for f in self.feature_params.values())

    def resize(self):
        assert self.num_labels <= self.max_num_labels, "Exceeded maximum number of labels"

    def save(self, filename):
        """
        Save all parameters to file
        :param filename: file to save to
        """
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "is_frozen": self.is_frozen,
        }
        save_dict(filename, d)
        self.init_model()
        with open(filename + ".json", "w") as f:
            f.write(self.model.to_json())
        try:
            self.model.save_weights(filename + ".h5", overwrite=True)
        except ValueError as e:
            print("Failed saving model weights: %s" % e)

    def load(self, filename):
        """
        Load all parameters from file
        :param filename: file to load from
        """
        d = load_dict(filename)
        model_type = d.get("type")
        assert model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.is_frozen = d["is_frozen"]
        with open(filename + ".json") as f:
            self.model = model_from_json(f.read())
        try:
            self.model.load_weights(filename + ".h5")
        except KeyError as e:
            print("Failed loading model weights: %s" % e)
        self.compile()

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)
