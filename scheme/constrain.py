from scheme.constraint.amr import AMRConstraints
from scheme.constraint.sdp import SDPConstraints
from scheme.constraints import UCCAConstraints

CONSTRAINTS = {
    None: UCCAConstraints,
    "amr": AMRConstraints,
    "sdp": SDPConstraints,
}
