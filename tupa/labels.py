class Labels:
    def __init__(self, size):
        self.size = size  # Maximum number of labels, NOT enforced here but by the user

    @property
    def all(self):
        raise NotImplementedError()

    @all.setter
    def all(self, labels):
        raise NotImplementedError()

    def save(self, skip=False):
        return (None if skip else self.all), self.size

    def load(self, all_size):
        self.all, self.size = all_size
