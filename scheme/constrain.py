from scheme.constraint.amr import AMRConstraints
from scheme.constraints import UCCAConstraints

CONSTRAINTS = {
    None: UCCAConstraints,
    "amr": AMRConstraints,
}
