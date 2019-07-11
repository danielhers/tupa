from .action import Actions
from .config import Config, COMPOUND
from .states.state import InvalidActionError

# Constants for readability, used by Oracle.action
RIGHT = PARENT = NODE = 0
LEFT = CHILD = EDGE = 1
LABEL = 2
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
    :param graph: gold Graph to get the correct nodes and edges from
    :param conllu: Graph with node per token predicted by a syntactic parser
    :param alignment: Graph with node.id corresponding to graph and node.label corresponding to conllu node.id
    """
    def __init__(self, graph, conllu=None, alignment=None):
        self.args = Config().args
        self.nodes_remaining = {n.id for n in graph.nodes}
        self.edges_remaining = {e for n in graph.nodes for e in n.outgoing_edges}
        self.graph = graph
        self.conllu = conllu
        self.alignment = alignment
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
                    if self.args.validate_oracle:
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
                                  for l in (s0.orig_node.incoming_edges, s0.orig_node.outgoing_edges)]
            if not incoming and not outgoing and not self.need_label(s0):
                yield self.action(Actions.Reduce)
            else:
                # Check for node label action: if all terminals have already been connected
                if self.need_label(s0) and not any(self.is_terminal_edge(e) for e in outgoing):
                    yield self.action(s0, LABEL, 1)

                # Check for actions to create new nodes
                for edge in incoming:
                    if edge.src in self.nodes_remaining and not self.is_implicit_node(edge.src):
                        yield self.action(edge, NODE, PARENT)  # Node

                for edge in outgoing:
                    if edge.tgt in self.nodes_remaining and self.is_implicit_node(edge.tgt):
                        yield self.action(edge, NODE, CHILD)  # Implicit

                if len(state.stack) > 1:
                    s1 = state.stack[-2]
                    # Check for node label action: if all terminals have already been connected
                    if self.need_label(s1) and not any(self.is_terminal_edge(e) for e in
                                                       self.edges_remaining.intersection(s1.orig_node.outgoing_edges)):
                        yield self.action(s1, LABEL, 2)

                    # Check for actions to create binary edges
                    for edge in incoming:
                        if edge.src == s1.id:
                            yield self.action(edge, EDGE, RIGHT)  # RightEdge

                    for edge in outgoing:
                        if edge.tgt == s1.id:
                            yield self.action(edge, EDGE, LEFT)  # LeftEdge
                        elif state.buffer and edge.tgt == state.buffer[0].id and \
                                len(state.buffer[0].orig_node.incoming_edges) == 1:
                            yield self.action(Actions.Shift)  # Special case to allow discarding simple children quickly

                    if not self.found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = dict([(edge.tgt,  edge) for edge in outgoing] +
                                       [(edge.src, edge) for edge in incoming])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1], start=1):  # Skip top two: checked above, not related
                            edge = related.pop(s.id, None)
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

    def action(self, edge, kind=None, direction=None):
        self.found = True
        if kind is None:
            return edge  # Will be just an Action object in this case
        if kind == LABEL:
            return Actions.Label(direction, orig_node=edge.orig_node, oracle=self)
        node = self.graph.find_node((edge.src, edge.tgt)[direction]) if kind == NODE else None
        return ACTIONS[kind][direction](tag=edge.lab, orig_edge=edge, orig_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.discard(edge)
        if node is not None:
            self.nodes_remaining.discard(node.id)

    def need_label(self, node):
        return self.args.node_labels and not node.labeled and node.orig_node.label

    def get_label(self, state, node):
        true_label = raw_true_label = None
        if node.orig_node is not None:
            raw_true_label = node.orig_node.label
        if raw_true_label is not None:
            true_label, _, _ = raw_true_label.partition("|")
            if self.args.validate_oracle:
                try:
                    state.check_valid_label(true_label, message=True)
                except InvalidActionError as e:
                    raise InvalidActionError("True label is invalid: " + "\n".join(map(str, (true_label, state, e))))
        return true_label, raw_true_label

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (
            " ".join(map(str, self.nodes_remaining)), sep,
            " ".join("%s->%s[%s]" % (e.src, e.tgt, e.lab) for e in self.edges_remaining))

    def __str__(self):
        return str(" ")

    def is_implicit_node(self, i):
        return not self.graph.find_node(i).outgoing_edges

    def is_terminal_edge(self, edge):
        return bool(self.graph.find_node(edge.tgt).anchors)
