from ..constraints import Constraints, EdgeTags
from ..conversion.conllu import ConlluConverter


class ConlluConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, unique_outgoing={ConlluConverter.HEAD}, required_outgoing={ConlluConverter.HEAD},
                         childless_incoming_trigger={ConlluConverter.HEAD},
                         childless_outgoing_allowed={EdgeTags.Terminal, EdgeTags.Punctuation})
