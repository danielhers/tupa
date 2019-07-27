from .validation import Constraints, ANCHOR_LAB


class EdsConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, required_outgoing={ANCHOR_LAB})
