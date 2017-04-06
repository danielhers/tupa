from util.amr import *


class Constraints(constraints.Constraints):
    def __init__(self, args):
        super(Constraints, self).__init__(args, root_label=VARIABLE_LABEL, require_connected=True,
                                          require_implicit_childless=False, allow_root_terminal_children=True,
                                          possible_multiple_incoming=(), unique_outgoing={INSTANCE_OF},
                                          childless_incoming_trigger=INSTANCE_OF, unique_incoming=(),
                                          mutually_exclusive_outgoing=(), top_level=None)
        self.tag_rules.append(
            constraints.TagRule(trigger={constraints.Direction.incoming: "name"},
                                allowed={constraints.Direction.outgoing: re.compile(
                                    "^(%s|%s|op\d+)$" % (INSTANCE_OF, "|".join(TERMINAL_TAGS)))}))

    def allow_action(self, action, history):
        return True

    def _allow_edge(self, edge):
        return edge not in edge.parent.outgoing  # Prevent multiple identical edges between the same pair of nodes

    def allow_parent(self, node, tag):
        return not (node.implicit and tag in TERMINAL_TAGS or
                    not is_variable(node.label) and tag not in TERMINAL_TAGS) and \
               is_valid_arg(node, node.label, tag)

    def allow_child(self, node, tag):
        return is_concept(node.label) == (tag == INSTANCE_OF) and \
               (node.label == "Const(-)" or tag != "polarity") and \
               is_valid_arg(node, node.label, tag, is_parent=False)

    def allow_label(self, node, label):
        return (is_variable(label) or node.outgoing_tags <= TERMINAL_TAGS) and (
            not is_concept(label) or node.incoming_tags <= {INSTANCE_OF}) and (
            (label == "Const(-)") == (node.incoming_tags == {"polarity"})) and (
            not node.parents or
            is_valid_arg(node, label, *node.parents[0].outgoing_tags) and
            is_valid_arg(node, label, *node.parents[0].incoming_tags, is_parent=False)) and (
            TERMINAL_TAGS & node.outgoing_tags or is_variable(label) or
            not PLACEHOLDER.search(label))  # Prevent text placeholder in implicit node

    def allow_reduce(self, node):
        return node.text is not None or not is_variable(node.label) or INSTANCE_OF in node.outgoing_tags
