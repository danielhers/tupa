from ..constraints import Constraints
from ..util.amr import *


class AmrConstraints(Constraints):
    def __init__(self, args):
        super().__init__(args, require_implicit_childless=False, allow_root_terminal_children=True,
                         possible_multiple_incoming=TERMINAL_DEP, childless_incoming_trigger=WIKI,
                         childless_outgoing_allowed=TERMINAL_TAGS)

    def allow_action(self, action, history):
        return True

    def _allow_edge(self, edge):  # Prevent multiple identical edges between the same pair of nodes
        return edge.tag in PREFIXED_RELATION_ENUM or edge not in edge.parent.outgoing

    def allow_parent(self, node, tag):
        return (not node.implicit or tag not in TERMINAL_TAGS) and \
               (node.label is None or (is_concept(node.label) or tag in TERMINAL_TAGS)) and \
               is_valid_arg(node, node.label, tag)

    def allow_child(self, node, tag):
        return is_valid_arg(node, node.label, tag, is_parent=False)

    def allow_label(self, node, label):
        return (is_concept(label) or node.outgoing_tags <= TERMINAL_TAGS and not node.is_root) and \
               (not node.parents or
                is_valid_arg(node, label, *node.outgoing_tags) and
                is_valid_arg(node, label, *node.incoming_tags, is_parent=False))
