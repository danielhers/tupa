class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, labels=None, model=None):
        """
        :param labels: a list of labels that can be updated later to add a new label
        :param model: if given, copy the weights (from a trained model)
        """
        self.labels = labels or []
        self._num_labels = len(self.labels)
        self.is_frozen = model is not None

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

    def resize(self):
        raise NotImplementedError()

    def finalize(self, *args, **kwargs):
        assert not self.is_frozen, "Cannot freeze a frozen model"
        self._update_num_labels()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
