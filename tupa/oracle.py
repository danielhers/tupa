from operator import attrgetter

from ucca import layer1

from scheme.util.amr import LABEL_ATTRIB, LABEL_SEPARATOR
from .action import Actions
from .config import Config, COMPOUND
from .states.state import InvalidActionError

# Constants for readability, used by Oracle.action
RIGHT = PARENT = NODE = 0
LEFT = CHILD = EDGE = 1
LABEL = 2
ACTIONS = (  # index by [NODE/EDGE][PARENT/CHILD or RIGHT/LEFT][True/False (remote)]
    (  # node actions
        (Actions.Node, Actions.RemoteNode),  # creating a parent
        (Actions.Implicit, None)  # creating a child (remote implicit is not allowed)
    ),
    (  # edge actions
        (Actions.RightEdge, Actions.RightRemote),  # creating a right edge
        (Actions.LeftEdge, Actions.LeftRemote)  # creating a left edge
    )
)


class Oracle(object):
    """
    Oracle to produce gold transition parses given UCCA passages
    To be used for creating training data for a transition-based UCCA parser
    :param passage gold passage to get the correct edges from
    """
    def __init__(self, passage):
        self.args = Config().args
        l1 = passage.layer(layer1.LAYER_ID)
        self.nodes_remaining = {node.ID for node in l1.all
                                if node is not l1.heads[0] and
                                (self.args.linkage or node.tag != layer1.NodeTags.Linkage) and
                                (self.args.implicit or not node.attrib.get("implicit"))}
        self.edges_remaining = {edge for node in passage.nodes.values() for edge in node
                                if (self.args.linkage or edge.tag not in (
                                    layer1.EdgeTags.LinkRelation, layer1.EdgeTags.LinkArgument)) and
                                (self.args.implicit or not edge.child.attrib.get("implicit")) and
                                (self.args.remote or not edge.attrib.get("remote"))}
        self.passage = passage
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
                    state.check_valid_action(action, message=True)
                    actions[action.id] = action
                except InvalidActionError as e:
                    invalid.append((action, e))
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
                                  for l in (s0.orig_node.incoming, s0.orig_node.outgoing)]
            if not incoming and not outgoing and not self.need_label(s0):
                yield self.action(Actions.Reduce)
            else:
                # Check for node label action: if all terminals have already been connected
                if self.need_label(s0) and not any(e.tag == layer1.EdgeTags.Terminal for e in outgoing):
                    yield self.action(s0, LABEL, 0)

                # Check for actions to create new nodes
                for edge in incoming:
                    if edge.parent.ID in self.nodes_remaining and not edge.parent.attrib.get("implicit") and (
                                not edge.attrib.get("remote") or
                                # Allow remote parent if all its children are remote/implicit
                                all(e.attrib.get("remote") or e.child.attrib.get("implicit") for e in edge.parent)):
                        yield self.action(edge, NODE, PARENT)  # Node or RemoteNode

                for edge in outgoing:
                    if edge.child.ID in self.nodes_remaining and edge.child.attrib.get("implicit") and (
                            not edge.attrib.get("remote")):  # Allow implicit child if it is not remote
                        yield self.action(edge, NODE, CHILD)  # Implicit

                if len(state.stack) > 1:
                    s1 = state.stack[-2]
                    # Check for node label action: if all terminals have already been connected
                    if self.need_label(s1) and not any(e.tag == layer1.EdgeTags.Terminal for e in
                                                       self.edges_remaining.intersection(s1.orig_node.outgoing)):
                        yield self.action(s1, LABEL, 1)

                    # Check for actions to create binary edges
                    for edge in incoming:
                        if edge.parent.ID == s1.node_id:
                            yield self.action(edge, EDGE, RIGHT)  # RightEdge or RightRemote

                    for edge in outgoing:
                        if edge.child.ID == s1.node_id:
                            yield self.action(edge, EDGE, LEFT)  # LeftEdge or LeftRemote
                        elif state.buffer and edge.child.ID == state.buffer[0].node_id and \
                                len(state.buffer[0].orig_node.incoming) == 1:
                            yield self.action(Actions.Shift)  # Special case to allow discarding simple children quickly

                    if not self.found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = dict([(edge.child.ID,  edge) for edge in outgoing] +
                                       [(edge.parent.ID, edge) for edge in incoming])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1], start=1):  # Skip top two: checked above, not related
                            edge = related.pop(s.node_id, None)
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
        remote = edge.attrib.get("remote", False)
        node = (edge.parent, edge.child)[direction] if kind == NODE else None
        return ACTIONS[kind][direction][remote](tag=edge.tag, orig_edge=edge, orig_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.discard(edge)
        if node is not None:
            self.nodes_remaining.discard(node.ID)

    def need_label(self, node):
        return self.args.node_labels and not node.labeled and node.orig_node.attrib.get(LABEL_ATTRIB)

    @staticmethod
    def get_label(state, action):
        true_label = raw_true_label = None
        if action.orig_node is not None:
            raw_true_label = action.orig_node.attrib.get(LABEL_ATTRIB)
        if raw_true_label is not None:
            true_label, _, _ = raw_true_label.partition(LABEL_SEPARATOR)
            try:
                state.check_valid_label(true_label, message=True)
            except InvalidActionError as e:
                raise InvalidActionError("True label is invalid: " + "\n".join(map(str, (true_label, state, e))))
        return true_label, raw_true_label

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (" ".join(self.nodes_remaining), sep,
                                                       " ".join(map(str, self.edges_remaining)))

    def __str__(self):
        return str(" ")
