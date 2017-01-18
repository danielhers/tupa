from enum import Enum

from parsing.config import Config
from parsing.model_util import load_dict, save_dict


class ClassifierProperty(Enum):
    update_only_on_error = 1
    require_init_features = 2
    trainable_after_saving = 3


class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, model_type, filename, labels, model=None):
        """
        :param labels: a list of labels that can be updated later to add a new label
        :param model: if given, copy the weights (from a trained model)
        """
        self.model = None
        self.model_type = model_type
        self.filename = filename
        self.labels = labels or []
        self._num_labels = len(self.labels)
        self.is_frozen = model is not None
        self.learning_rate = Config().args.learning_rate
        self.learning_rate_decay = Config().args.learning_rate_decay

    @property
    def num_labels(self):
        return len(self.labels)

    def score(self, features):
        if not self.is_frozen:
            self._update_num_labels()

    def update(self, features, pred, true, importance=1):
        assert not self.is_frozen, "Cannot update a frozen model"
        self._update_num_labels()

    def _update_num_labels(self):
        """
        self.num_labels is updated automatically when a label is added to self.labels,
        but we need to update the weights whenever that happens
        """
        if self.num_labels > self._num_labels:
            self._num_labels = self.num_labels
            self.resize()

    def finished_item(self, train=False):
        """
        Mark the current item as finished.  Fit the model if reached the batch size.
        :param train: fit the model if batch size reached?
        """
        pass

    def finished_step(self, train=False):
        """
        Mark the current time step as finished.
        """
        pass

    def resize(self):
        raise NotImplementedError()

    def finalize(self, *args, **kwargs):
        assert not self.is_frozen, "Cannot freeze a frozen model"
        self._update_num_labels()

    def save(self):
        """
        Save all parameters to file
        """
        d = {
            "type": self.model_type,
            "labels": self.labels,
            "is_frozen": self.is_frozen,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
        }
        d.update(self.save_model())
        save_dict(self.filename, d)

    def save_model(self):
        return self.save_extra()

    def save_extra(self):
        return {}

    def load(self):
        """
        Load all parameters from file
        """
        d = load_dict(self.filename)
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, \
            "Model type does not match: %s" % model_type
        self.labels = list(d["labels"])
        self.is_frozen = d["is_frozen"]
        self.learning_rate = d["learning_rate"]
        self.learning_rate_decay = d["learning_rate_decay"]
        self.load_model(d)

    def load_model(self, d):
        self.load_extra(d)

    def load_extra(self, d):
        pass

    def get_classifier_properties(self):
        return ()

    def __str__(self):
        return ("%d labels, " % self.num_labels) + (
                "%d features" % self.input_dim)
