from ..constraints import Constraints
from ..conversion.conllu import ConlluConverter


class ConlluConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, unique_outgoing={ConlluConverter.HEAD}, required_outgoing={ConlluConverter.HEAD})
