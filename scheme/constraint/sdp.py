from ..constraints import Constraints
from ..conversion.sdp import SdpConverter

TOP_LEVEL = (SdpConverter.ROOT, SdpConverter.TOP)


class SdpConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, top_level_allowed=TOP_LEVEL, top_level_only=TOP_LEVEL)
