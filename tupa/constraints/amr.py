from .util import WEEKDAYS, SEASONS, INT_PATTERN, PLACEHOLDER_PATTERN, DAY, MONTH, YEAR, YEAR2, DECADE, \
    WEEKDAY, QUARTER, CENTURY, SEASON, TIMEZONE, DATE_ENTITY, NAME, MINUS, POLARITY, ARG2, VALUE, MODE, MODES
from .validation import Constraints, Valid, ROOT_LAB
from ..recategorization import resolve


def is_int_in_range(value, s=None, e=None):
    m = INT_PATTERN.match(value)
    if not m:
        return Valid(False, "%s is not numeric" % value)
    num = int(m.group(1))
    return Valid(s is None or num >= s, "%s < %s" % (num, s)) and Valid(e is None or num <= e, "%s > %s" % (num, e))


def is_valid_arg(value, *labs, is_parent=True, is_node_label=True):
    if value is None or PLACEHOLDER_PATTERN.search(value):  # Not labeled yet or not yet resolved properly
        return True
    valid = Valid(message="%s incompatible as %s of %s" % (value, "parent" if is_parent else "child", ", ".join(labs)))
    if is_parent:  # node is a parent of the edge
        if {DAY, MONTH, YEAR, YEAR2, DECADE, WEEKDAY, QUARTER, CENTURY, SEASON, TIMEZONE}.intersection(labs):
            return valid(value == DATE_ENTITY)
    elif is_node_label:
        if WEEKDAY in labs:  # :weekday  excl,a=date-entity,b=[monday|tuesday|wednesday|thursday|friday|saturday|sunday]
            return valid(value in WEEKDAYS)
        elif value in WEEKDAYS:
            return valid(WEEKDAY in labs)
        elif SEASON in labs:  # :season excl,a=date-entity,b=[winter|fall|spring|summer]+
            return valid(value in SEASONS)
        elif NAME in labs:
            return valid(value == NAME)
    # property value, i.e., constant
    elif value == MINUS:  # :polarity excl,b_isconst,b_const=-
        return valid({POLARITY, ARG2, VALUE}.issuperset(labs))
    elif POLARITY in labs:
        return valid(value == MINUS)
    elif MODE in labs:  # :mode excl,b_isconst,b_const=[interrogative|expressive|imperative]
        return valid(value in MODES)
    elif value in MODES:
        return valid(MODE in labs)
    elif DAY in labs:  # :day  a=date-entity,b_isconst,b_const=[...]
        return is_int_in_range(value, 1, 31)
    elif MONTH in labs:  # :month  a=date-entity,b_isconst,b_const=[1|2|3|4|5|6|7|8|9|10|11|12]
        return is_int_in_range(value, 1, 12)
    elif QUARTER in labs:  # :quarter  a=date-entity,b_isconst,b_const=[1|2|3|4]
        return is_int_in_range(value, 1, 4)
    elif {YEAR, YEAR2, DECADE, CENTURY}.intersection(labs):  # :year a=date-entity,b_isconst,b_const=[0-9]+
        return is_int_in_range(value)
    return True


class AmrConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(multigraph=True, require_implicit_childless=False, allow_orphan_terminals=True,
                         unique_outgoing={ROOT_LAB},
                         childless_incoming_trigger={POLARITY, CENTURY, DECADE, "polite", "li"}, **kwargs)

    def allow_action(self, action, history):
        return True

    def allow_edge(self, edge):  # Prevent multiple identical edges between the same pair of nodes
        return edge not in edge.parent.outgoing

    def allow_parent(self, node, lab):
        return not lab or is_valid_arg(resolve(node, node.label), lab)

    def allow_child(self, node, lab):
        return not lab or is_valid_arg(resolve(node, node.label), lab, is_parent=False)

    def allow_label(self, node, label):
        return not node.parents or \
               is_valid_arg(label, *node.outgoing_labs) and \
               is_valid_arg(label, *node.incoming_labs, is_parent=False)

    def allow_property_value(self, node, property_value):
        prop, value = property_value
        return is_valid_arg(value, prop, is_parent=False)
