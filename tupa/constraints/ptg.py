from .validation import Constraints


class PtgConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(multigraph=True, **kwargs)
