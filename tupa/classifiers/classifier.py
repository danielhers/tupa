from collections import OrderedDict

from ..model_util import load_json, save_json


class Classifier:
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, config, labels, input_params=None):
        """
        :param config: Config to get hyperparameters from
        :param labels: dict of axis (string) -> Labels object, can be updated later to add new axes and labels
        :param input_params: dict of feature type name -> FeatureInformation
        """
        self.config = config
        self.labels = labels
        self.input_params = input_params
        self.model_type = self.config.args.classifier
        self.learning_rate = self.config.args.learning_rate
        self.learning_rate_decay = self.config.args.learning_rate_decay
        self.model = self.labels_t = None
        self.is_frozen = False
        self.updates = self.epoch = self.best_score = 0
        self._num_labels = self.num_labels

    @property
    def num_labels(self):
        return OrderedDict((a, len(l.all)) for a, l in self.labels.items()) if self.labels \
            else OrderedDict((a, len(l) if l else s) for a, (l, s) in self.labels_t.items()) if self.labels_t else {}
    
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

    def finalize(self, *args, finished_epoch=False, **kwargs):
        assert not self.is_frozen, "Cannot freeze a frozen model"
        self._update_num_labels()
        if finished_epoch:
            self.epoch += 1

    def finished_step(self, train=False):
        """
        Called by the parser when a single step is finished
        """
        pass

    def finished_item(self, train=False, renew=True):
        """
        Called by the parser when a whole item is finished
        """
        pass

    def transition(self, action, axis):
        pass

    def save(self, filename, skip_labels=(), **kwargs):
        """
        Save all parameters to file
        """
        d = OrderedDict((
            ("type", self.model_type),
            ("axes", OrderedDict(
                (a, OrderedDict((("index", i), ("labels", l.save(skip=a in skip_labels)))))  # (all, size)
                for i, (a, l) in enumerate(self.labels.items())
            )),
            ("is_frozen", self.is_frozen),
            ("learning_rate", self.learning_rate),
            ("learning_rate_decay", self.learning_rate_decay),
            ("updates", self.updates),
            ("epoch", self.epoch),
            ("best_score", self.best_score),
        ) + tuple(kwargs.items()))
        self.save_model(filename, d)
        save_json(filename + ".json", d)

    def save_model(self, filename, d):
        """
        Save all parameters to file
        """
        pass

    def load(self, filename):
        """
        Load all parameters from file
        """
        d = self.load_file(filename, clear=True)
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels_t = OrderedDict((a, l["labels"]) for a, l in sorted(d["axes"].items(), key=lambda x: x[1]["index"]))
        self.is_frozen = d["is_frozen"]
        self.config.args.learning_rate = self.learning_rate = d["learning_rate"]
        self.config.args.learning_rate_decay = self.learning_rate_decay = d["learning_rate_decay"]
        self.updates = d.get("updates", 0)
        self.epoch = d.get("epoch", 0)
        self.best_score = d.get("best_score", 0)
        self.load_model(filename, d)

    def load_model(self, filename, d):
        pass

    @classmethod
    def get_property(cls, filename, prop):
        return cls.load_file(filename).get(prop, None)

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

    def all_params(self):
        return OrderedDict((k, v.all) for k, v in self.labels.items())

    def __str__(self):
        return "Labels: %s, features: %s" % tuple(map(dict_value, (self.num_labels, self.input_dim)))

    def print_params(self, max_rows=10):
        pass


def dict_value(d):
    return next(iter(d.values())) if len(d) == 1 else ", ".join("%s: %d" % i for i in d.items())
