import os
import shelve
import time

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

    def write(self, filename, sep="\t"):
        raise NotImplementedError()

    @staticmethod
    def save_dict(filename, d):
        """
        Save dictionary to file
        :param filename: file to write to; the actual written file may have an additional suffix
        :param d: dictionary to save
        """
        print("Saving model to '%s'... " % filename, end="", flush=True)
        started = time.time()
        with shelve.open(filename) as db:
            db.update(d)
        print("Done (%.3fs)." % (time.time() - started))

    @staticmethod
    def load_dict(filename):
        """
        Load dictionary from file
        :param filename: file to read from; the actual read file may have an additional suffix
        """
        def try_open(*names):
            exception = None
            for f in names:
                # noinspection PyBroadException
                try:
                    return shelve.open(f, flag="r")
                except Exception as e:
                    exception = e
            if exception is not None:
                raise IOError("Model file not found: " + filename) from exception

        print("Loading model from '%s'... " % filename, end="", flush=True)
        started = time.time()
        with try_open(filename, os.path.splitext(filename)[0]) as db:
            d = dict(db)
        print("Done (%.3fs)." % (time.time() - started))
        return d
