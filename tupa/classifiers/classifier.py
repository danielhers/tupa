from collections import OrderedDict

from ..config import Config
from ..model_util import load_json, save_json


class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, model_type, filename, labels, input_params=None):
        """
        :param labels: dict of axis (string) -> Labels object, can be updated later to add new axes and labels
        :param input_params: dict of feature type name -> FeatureInformation
        """
        self.args = Config().args
        self.model_type = model_type
        self.filename = filename
        self.labels = labels
        self.input_params = input_params
        self.learning_rate = self.args.learning_rate
        self.learning_rate_decay = self.args.learning_rate_decay
        self._num_labels = self.num_labels
        self.model = self.labels_t = None
        self.is_frozen = False
        self.updates = self.epoch = self.best_score = 0

    @property
    def num_labels(self):
        return {a: len(l.all) for a, l in self.labels.items()}
    
    @property
    def input_dim(self):
        raise NotImplementedError()

    def score(self, features, axis):
        if not self.is_frozen:
            self._update_num_labels()

    def init_features(self, features, axes, train=False):
        pass

    def update(self, features, axis, pred, true, importance=None):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values
        :param axis: axis of the label we are predicting
        :param pred: label predicted by the classifier (non-negative integer bounded by num_labels[axis])
        :param true: true labels (non-negative integers bounded by num_labels[axis])
        :param importance: how much to scale the update for the weight update for each true label
        """
        assert not self.is_frozen, "Cannot update a frozen model"
        self._update_num_labels()

    def _update_num_labels(self):
        """
        self.num_labels is a property, and so updated automatically when a label is added to self.labels,
        but we may need to resize the weight matrices whenever that happens
        """
        if self._num_labels != self.num_labels:
            self._num_labels = self.num_labels
            self.resize()

    def resize(self):
        raise NotImplementedError()

    def finalize(self, *args, **kwargs):
        assert not self.is_frozen, "Cannot freeze a frozen model"
        self._update_num_labels()

    def finished_step(self, train=False):
        """
        Called by the parser when a single step is finished
        """
        pass

    def finished_item(self, train=False):
        """
        Called by the parser when a whole item is finished
        """
        pass

    def save(self, skip_labels=()):
        """
        Save all parameters to file
        """
        d = OrderedDict((
            ("type", self.model_type),
            ("axes", OrderedDict(
                (a, OrderedDict(labels=l.save(skip=a in skip_labels))) for a, l in self.labels.items()  # (all, size)
            )),
            ("is_frozen", self.is_frozen),
            ("learning_rate", self.learning_rate),
            ("learning_rate_decay", self.learning_rate_decay),
            ("updates", self.updates),
            ("epoch", self.epoch),
            ("best_score", self.best_score),
        ))
        self.save_model(d)
        save_json(self.filename + ".json", d)

    def save_model(self, d):
        """
        Save all parameters to file
        """
        pass

    def load(self):
        """
        Load all parameters from file
        """
        d = self.load_file(self.filename, clear=True)
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels_t = OrderedDict((a, l["labels"]) for a, l in d["axes"].items())  # labels to be corrected by Model
        self.is_frozen = d["is_frozen"]
        self.args.learning_rate = self.learning_rate = d["learning_rate"]
        self.args.learning_rate_decay = self.learning_rate_decay = d["learning_rate_decay"]
        self.updates = d["updates"]
        self.epoch = d["epoch"]
        self.best_score = d.get("best_score", 0)
        self.load_model(d)

    def load_model(self, d):
        pass

    @classmethod
    def get_model_type(cls, filename):
        return cls.load_file(filename).get("type")

    LOADED = {}  # Cache for loaded JSON files

    @classmethod
    def load_file(cls, filename, clear=False):
        d = cls.LOADED.get(filename)
        if d is None:
            d = load_json(filename + ".json")
            cls.LOADED[filename] = d
        if clear:
            cls.LOADED.clear()
        return d

    def get_all_params(self):
        return OrderedDict((k, v.all) for k, v in self.labels.items())

    def __str__(self):
        return "Labels: %s, features: %s" % tuple(map(dict_value, (self.num_labels, self.input_dim)))


def dict_value(d):
    return next(iter(d.values())) if len(d) == 1 else d
