from scheme.constraints import Constraints
from scheme.conversion.sdp import SdpConverter

TOP_LEVEL = (SdpConverter.ROOT, SdpConverter.TOP)


class SdpConstraints(Constraints):
    def __init__(self, args):
        super(SdpConstraints, self).__init__(args, top_level_allowed=TOP_LEVEL, top_level_only=TOP_LEVEL)
