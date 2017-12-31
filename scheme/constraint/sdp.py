from ..constraints import Constraints, EdgeTags
from ..conversion.sdp import SdpConverter

TOP_LEVEL = (SdpConverter.ROOT, SdpConverter.TOP)


class SdpConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, top_level_allowed=TOP_LEVEL, top_level_only=TOP_LEVEL,
                         unique_outgoing={SdpConverter.HEAD}, required_outgoing={SdpConverter.HEAD},
                         childless_incoming_trigger={SdpConverter.HEAD, "mwe"},
                         childless_outgoing_allowed={EdgeTags.Terminal, EdgeTags.Punctuation})
