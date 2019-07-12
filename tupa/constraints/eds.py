from .validation import Constraints


class EdsConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
