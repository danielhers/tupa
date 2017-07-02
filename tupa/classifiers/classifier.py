from enum import Enum

from tupa.config import Config
from tupa.model_util import load_dict, save_dict


class ClassifierProperty(Enum):
    update_only_on_error = 1
    require_init_features = 2
    trainable_after_saving = 3


class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, model_type, filename, labels, input_params=None, model=None):
        """
        :param labels: tuple of lists of labels that can be updated later to add new labels
        :param input_params: dict of feature type name -> FeatureInformation
        :param model: if given, copy the weights (from a trained model)
        """
        self.model = None
        self.model_type = model_type
        self.filename = filename
        self.labels = tuple(labels)
        self.input_params = input_params
        self._num_labels = self.num_labels
        self.input_dim = None
        self.is_frozen = model is not None
        self.updates = 0
        self.epoch = 0
        self.learning_rate = Config().args.learning_rate
        self.learning_rate_decay = Config().args.learning_rate_decay

    @property
    def num_labels(self):
        return tuple(map(len, self.labels))

    def num_labels_str(self):
        return "x".join(map(str, self.num_labels))

    def score(self, features, axis):
        if not self.is_frozen:
            self._update_num_labels(axis)

    def init_features(self, features, train=False):
        pass

    def update(self, features, axis, pred, true, importance=1):
        assert not self.is_frozen, "Cannot update a frozen model"
        self._update_num_labels(axis)

    def _update_num_labels(self, axis=None):
        """
        self.num_labels is updated automatically when a label is added to self.labels,
        but we need to update the weights whenever that happens
        """
        if self.num_labels != self._num_labels:
            self._num_labels = self.num_labels
            self.resize(axis)

    def resize(self, axis=None):
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

    def save(self):
        """
        Save all parameters to file
        """
        d = {
            "type": self.model_type,
            "labels": self.save_labels(),
            "is_frozen": self.is_frozen,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "updates": self.updates,
            "epoch": self.epoch,
        }
        d.update(self.save_model())
        save_dict(self.filename + ".dict", d)

    def save_labels(self):
        return self.labels

    def save_model(self):
        return self.save_extra()

    def save_extra(self):
        return {}

    def load(self):
        """
        Load all parameters from file
        """
        d = load_dict(self.filename + ".dict")
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels = tuple(map(list, d["labels"]))
        self.is_frozen = d["is_frozen"]
        self.updates = d.get("updates", d.get("_update_index", 0))
        self.epoch = d.get("epoch", 0)
        Config().args.learning_rate = self.learning_rate = d["learning_rate"]
        Config().args.learning_rate_decay = self.learning_rate_decay = d["learning_rate_decay"]
        self.load_model(d)

    def load_model(self, d):
        self.load_extra(d)

    def load_extra(self, d):
        pass

    def get_classifier_properties(self):
        return ()

    def __str__(self):
        return "%s labels, %d features" % (self.num_labels_str(), self.input_dim)
