class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, model_type, filename=None, labels=None, model=None):
        """
        :param labels: a list of labels that can be updated later to add a new label
        :param model: if given, copy the weights (from a trained model)
        """
        self.model_type = model_type
        self.filename = filename
        self.labels = labels or []
        self._num_labels = len(self.labels)
        self.is_frozen = model is not None

    @property
    def num_labels(self):
        return len(self.labels)

    def score(self, features):
        if not self.is_frozen:
            self._update_num_labels()

    def init_features(self, features, train=False):
        pass

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

    def finish(self, train=False):
        """
        Mark the current item as finished.  Fit the model if reached the batch size.
        :param train: fit the model if batch size reached?
        """
        pass

    def advance(self):
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
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
