from .validation import Constraints


class SdpConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(allow_orphan_terminals=True, **kwargs)
