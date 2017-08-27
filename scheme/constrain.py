from scheme.constraint.amr import AmrConstraints
from scheme.constraint.conllu import ConlluConstraints
from scheme.constraint.sdp import SdpConstraints
from scheme.constraints import UccaConstraints

CONSTRAINTS = {
    None:     UccaConstraints,
    "amr":    AmrConstraints,
    "sdp":    SdpConstraints,
    "conllu": ConlluConstraints,
}
