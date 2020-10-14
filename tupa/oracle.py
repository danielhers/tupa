from .action import Actions
from .config import Config, COMPOUND, requires_node_labels, requires_node_properties, requires_edge_attributes
from .states.state import InvalidActionError

# Constants for readability, used by Oracle.action
RIGHT = PARENT = NODE = 0
LEFT = CHILD = EDGE = 1
LABEL = 2
PROPERTY = 3
ATTRIBUTE = 4
ACTIONS = (  # index by [NODE/EDGE][PARENT/CHILD or RIGHT/LEFT]
    (  # node actions
        Actions.Node,  # creating a parent
        Actions.Implicit  # creating a child
    ),
    (  # edge actions
        Actions.RightEdge,  # creating a right edge
        Actions.LeftEdge  # creating a left edge
    )
)


class Oracle:
    """
    Oracle to produce gold transition parses given MRP graphs
    To be used for creating training data for a transition-based meaning representation parser
    :param ref_graph: gold RefGraph to get the correct nodes and edges from
    :param framework: string indicating graph framework
    """
    def __init__(self, ref_graph):
        self.args = Config().args
        self.framework = ref_graph.framework
        self.terminal_ids = {int(terminal.id) for terminal in ref_graph.terminals}
        self.nodes_remaining = {int(n.id) for n in ref_graph.nodes
                                if n.id != ref_graph.root.id and int(n.id) not in self.terminal_ids}
        self.edges_remaining = set(ref_graph.edges)
        self.found = False
        self.log = None

    def get_actions(self, state, all_actions, create=True):
        """
        Determine all zero-cost action according to current state
        Asserts that the returned action is valid before returning
        :param state: current State of the parser
        :param all_actions: Actions object used to map actions to IDs
        :param create: whether to create new actions if they do not exist yet
        :return: dict of action ID to Action
        """
        actions = {}
        invalid = []
        for action in self.generate_actions(state):
            all_actions.generate_id(action, create=create)
            if action.id is not None:
                try:
                    if self.args.validate_oracle and not action.is_type(Actions.Swap):
                        # The oracle has a special permission to swap the same pair of nodes more than once,
                        # although we do not allow the parser to do that to avoid infinite loops.
                        # Multiple swaps might be necessary if we created the nodes in a non-projective order.
                        # We prevent this in UCCA with a heuristic disallowing creating new nodes based on remote
                        # parents (see below), but for other frameworks a more sophisticated check would be required.
                        state.check_valid_action(action, message=True)
                    actions[action.id] = action
                except InvalidActionError as e:
                    invalid.append((action, e))
        if self.args.validate_oracle:
            assert actions, self.generate_log(invalid, state)
        return actions

    def generate_log(self, invalid, state):
        self.log = "\n".join(["Oracle found no valid action",
                              state.str("\n"), self.str("\n"),
                              "Actions returned by the oracle:"] +
                             ["  %s: %s" % (action, e) for (action, e) in invalid] or ["None"])
        return self.log

    def generate_actions(self, state):
        """
        Determine all zero-cost action according to current state
        :param state: current State of the parser
        :return: generator of Action items to perform
        """
        self.found = False
        if state.stack:
            s0 = state.stack[-1]
            incoming, outgoing = [[e for e in l if e in self.edges_remaining]
                                  for l in (s0.ref_node.incoming, s0.ref_node.outgoing)]
            if not incoming and not outgoing and not self.need_label(s0) and not self.need_property(s0):
                yield self.action(Actions.Reduce)
            else:
                # Check for node label action: if all terminals have already been connected
                if self.need_label(s0) and not any(self.is_terminal_edge(e) for e in outgoing):
                    yield self.action(s0, LABEL, 1)

                if self.need_property(s0):
                    yield self.action(s0, PROPERTY, 1)

                if self.need_attribute(state.last_edge):
                    yield self.action(state.last_edge, ATTRIBUTE)
                else:
                    # Check for actions to create new nodes
                    for edge in incoming:
                        if edge.src in self.nodes_remaining and not edge.attributes:  # Avoid remote node creation
                            yield self.action(edge, NODE, PARENT)  # Node

                    for edge in outgoing:
                        if edge.tgt in self.nodes_remaining and self.is_implicit_node(edge.child):
                            yield self.action(edge, NODE, CHILD)  # Implicit

                    if len(state.stack) > 1:
                        s1 = state.stack[-2]
                        finished_terminals = not any(self.is_terminal_edge(e) for e in
                                                     self.edges_remaining.intersection(s1.ref_node.outgoing))
                        # Check for node label action: if all terminals have already been connected
                        if self.need_label(s1) and finished_terminals:
                            yield self.action(s1, LABEL, 2)

                        if self.need_property(s1) and finished_terminals:
                            yield self.action(s1, PROPERTY, 2)

                        # Check for actions to create binary edges
                        for edge in incoming:
                            if edge.src == int(s1.id):
                                yield self.action(edge, EDGE, RIGHT)  # RightEdge

                        for edge in outgoing:
                            if edge.tgt == int(s1.id):
                                yield self.action(edge, EDGE, LEFT)  # LeftEdge
                            elif state.buffer and edge.tgt == int(state.buffer[0].id) and \
                                    len(state.buffer[0].ref_node.incoming) == 1:
                                yield self.action(Actions.Shift)  # Special case to allow discarding simple children

                    if not self.found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = dict([(edge.tgt, edge) for edge in outgoing if edge.tgt not in self.nodes_remaining] +
                                       [(edge.src, edge) for edge in incoming if edge.src not in self.nodes_remaining])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1], start=1):  # Skip top two: checked above, not related
                            edge = related.pop(int(s.id), None)
                            if edge is not None:
                                if not self.args.swap:  # We have no chance to reach it, so stop trying
                                    self.remove(edge)
                                    continue
                                if distance is None and self.args.swap == COMPOUND:  # Save the first one
                                    distance = min(i, Config().args.max_swap)  # Do not swap more than allowed
                                if not related:  # All related nodes are in the stack
                                    yield self.action(Actions.Swap(distance))
                                    break

        if not self.found:
            yield self.action(Actions.Shift if state.buffer else Actions.Finish)

    def action(self, node_or_edge, kind=None, direction=None):
        self.found = True
        if kind is None:
            return node_or_edge  # Will be just an Action object in this case
        if kind == LABEL:
            return Actions.Label(direction, ref_node=node_or_edge.ref_node, oracle=self)
        if kind == PROPERTY:
            return Actions.Property(direction, ref_node=node_or_edge.ref_node, oracle=self)
        if kind == ATTRIBUTE:
            return Actions.Attribute(direction, ref_edge=node_or_edge.ref_edge, oracle=self)
        node = (node_or_edge.parent, node_or_edge.child)[direction] if kind == NODE else None
        return ACTIONS[kind][direction](tag=node_or_edge.lab, ref_edge=node_or_edge, ref_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.discard(edge)
        if node is not None:
            self.nodes_remaining.discard(int(node.id))

    def need_label(self, node):
        return requires_node_labels(self.framework) and \
               node is not None and node.text is None and node.label is None and node.ref_node.label

    def need_property(self, node):
        return requires_node_properties(self.framework) and node is not None and node.text is None and \
               set(node.ref_node.properties or ()).difference(node.properties or ())

    def need_attribute(self, edge):
        return requires_edge_attributes(self.framework) and edge is not None and \
               set(edge.ref_edge.attributes or ()).difference(edge.attributes or ())

    def get_node_label(self, state, node):
        true_label = raw_true_label = None
        if node.ref_node is not None:
            raw_true_label = node.ref_node.label
        if raw_true_label is not None:
            true_label, _, _ = raw_true_label.partition("|")
            if self.args.validate_oracle:
                try:
                    state.check_valid_label(true_label, message=True)
                except InvalidActionError as e:
                    raise InvalidActionError("True label is invalid: " + "\n".join(map(str, (true_label, state, e))))
        return true_label, raw_true_label

    def get_node_property_value(self, state, node):
        try:
            true_property_value = next((k, v) for k, v in (node.ref_node.properties.items()
                                                           if node.ref_node.properties else [])
                                       if k not in (node.properties or ()))
        except StopIteration:
            return None
        if self.args.validate_oracle:
            try:
                state.check_valid_property_value(true_property_value, message=True)
            except InvalidActionError as e:
                raise InvalidActionError("True property-value pair is invalid: " +
                                         "\n".join(map(str, (true_property_value, state, e))))
        return true_property_value

    def get_edge_attribute_value(self, state, edge):
        true_attribute_value = next((k, v) for k, v in edge.ref_edge.attributes.items()
                                    if k not in (edge.attributes or ()))
        if self.args.validate_oracle:
            try:
                state.check_valid_attribute_value(true_attribute_value, message=True)
            except InvalidActionError as e:
                raise InvalidActionError("True attribute-value pair is invalid: " +
                                         "\n".join(map(str, (true_attribute_value, state, e))))
        return true_attribute_value

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (
            " ".join(map(str, self.nodes_remaining)), sep,
            " ".join("%s->%s[%s]" % (e.src, e.tgt, e.lab) for e in self.edges_remaining))

    def __str__(self):
        return str(" ")

    @staticmethod
    def is_implicit_node(node):
        return not node.outgoing

    def is_terminal_edge(self, edge):
        return edge.tgt in self.terminal_ids
