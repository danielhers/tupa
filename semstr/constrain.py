from .constraint.amr import AmrConstraints
from .constraint.conllu import ConlluConstraints
from .constraint.sdp import SdpConstraints
from .constraints import UccaConstraints

CONSTRAINTS = {
    None:     UccaConstraints,
    "amr":    AmrConstraints,
    "sdp":    SdpConstraints,
    "conllu": ConlluConstraints,
}
