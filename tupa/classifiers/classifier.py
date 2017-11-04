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
        self.model = None
        self.model_type = model_type
        self.filename = filename
        self.labels = labels
        self.input_params = input_params
        self._num_labels = self.num_labels
        self.input_dim = self.labels_t = None
        self.is_frozen = False
        self.updates = 0
        self.epoch = 0
        self.learning_rate = self.args.learning_rate
        self.learning_rate_decay = self.args.learning_rate_decay

    @property
    def num_labels(self):
        return {a: len(l.all) for a, l in self.labels.items()}

    def score(self, features, axis):
        if not self.is_frozen:
            self._update_num_labels()

    def init_features(self, features, axes, train=False):
        pass

    def update(self, features, axis, pred, true, importance=1):
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
        d = OrderedDict(
            type=self.model_type,
            labels={a: l.save(skip=a in skip_labels) for a, l in self.labels.items()},  # (all, size) for each
            is_frozen=self.is_frozen,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            updates=self.updates,
            epoch=self.epoch,
        )
        d.update(self.save_model())
        save_json(self.filename + ".json", d)

    def save_model(self):
        """
        Save all parameters to file
        """
        return {}

    def load(self):
        """
        Load all parameters from file
        """
        d = load_json(self.filename + ".json")
        model_type = d.get("type")
        assert model_type is None or model_type == self.model_type, "Model type does not match: %s" % model_type
        self.labels_t = d["labels"]  # Just a dict of (all, size) pairs, to be corrected by Model to Actions and Labels
        self.is_frozen = d["is_frozen"]
        self.updates = d.get("updates", d.get("_update_index", 0))
        self.epoch = d.get("epoch", 0)
        self.args.learning_rate = self.learning_rate = d["learning_rate"]
        self.args.learning_rate_decay = self.learning_rate_decay = d["learning_rate_decay"]
        self.load_model(d)

    def load_model(self, d):
        pass

    def __str__(self):
        return "Labels: %s, %d features" % (next(iter(self.num_labels.values())) if len(self.num_labels) == 1 else
                                            self.num_labels, self.input_dim)
