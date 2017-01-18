from parsing.action import Action, Actions
from parsing.config import Config
from ucca import layer1


class Oracle(object):
    """
    Oracle to produce gold transition parses given UCCA passages
    To be used for creating training data for a transition-based UCCA parser
    :param passage gold passage to get the correct edges from
    """
    def __init__(self, passage):
        l1 = passage.layer(layer1.LAYER_ID)
        self.nodes_remaining = {node.ID for node in l1.all
                                if node is not l1.heads[0] and
                                (Config().args.linkage or node.tag != layer1.NodeTags.Linkage) and
                                (Config().args.implicit or not node.attrib.get("implicit"))}
        self.edges_remaining = {edge for node in passage.nodes.values() for edge in node
                                if (Config().args.linkage or edge.tag not in (
                                    layer1.EdgeTags.LinkRelation, layer1.EdgeTags.LinkArgument)) and
                                (Config().args.implicit or not edge.child.attrib.get("implicit")) and
                                (Config().args.remote or not edge.attrib.get("remote"))}
        self.passage = passage
        self.edge_found = False
        self.log = None

    def get_actions(self, state):
        """
        Determine all zero-cost action according to current state
        Asserts that the returned action is valid before returning
        :param state: current State of the parser
        :return: list of Action items to perform
        """
        actions = []
        invalid = []
        for action in self.generate_actions(state):
            action.generate_id()
            try:
                state.assert_valid(action)
                actions.append(action)
            except AssertionError as e:
                invalid.append((action, e))
        assert actions, self.generate_log(invalid, state)
        return actions

    def generate_log(self, invalid, state):
        self.log = "\n".join(["Oracle found no valid action",
                              state.str("\n"), self.str("\n"),
                              "Actions returned by the oracle:"] +
                             ["  %s: %s" % (action, e) for (action, e) in invalid])
        return self.log

    def generate_actions(self, state):
        """
        Determine all zero-cost action according to current state
        :param state: current State of the parser
        :return: generator of Action items to perform
        """
        if not self.edges_remaining or not state.buffer and not state.stack:
            yield Actions.Finish
            if state.stack:
                yield Actions.Reduce
            return

        self.edge_found = False
        if state.stack:
            s0 = state.stack[-1]
            incoming = self.edges_remaining.intersection(s0.orig_node.incoming)
            outgoing = self.edges_remaining.intersection(s0.orig_node.outgoing)
            if not incoming and not outgoing:
                yield Actions.Reduce
                return
            else:
                # Check for actions to create new nodes
                for edge in incoming:
                    if edge.parent.ID in self.nodes_remaining and not edge.attrib.get("remote"):
                        yield self.create_node_action(edge, edge.parent, Actions.Node)

                for edge in outgoing:
                    if edge.child.attrib.get("implicit"):
                        yield self.create_node_action(edge, edge.child, Actions.Implicit)

                if len(state.stack) > 1:
                    s1 = state.stack[-2]
                    # Check for actions to create binary edges
                    for edge in incoming:
                        if edge.parent.ID == s1.node_id:
                            yield self.create_edge_action(edge, Action.RIGHT)

                    for edge in outgoing:
                        if edge.child.ID == s1.node_id:
                            yield self.create_edge_action(edge, Action.LEFT)

                    if not self.edge_found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = dict([(edge.child.ID,  edge) for edge in outgoing] +
                                       [(edge.parent.ID, edge) for edge in incoming])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1]):  # Skip top two, they are not related
                            edge = related.pop(s.node_id, None)
                            if edge is not None:
                                if not Config().args.swap:  # We have no chance to reach it, so stop trying
                                    self.remove(edge)
                                    continue
                                if distance is None and Config().args.compound_swap:  # Save the first one
                                    distance = i + 1
                                if not related:  # All related nodes are in the stack
                                    yield Actions.Swap(distance)
                                    return

        if state.buffer and not self.edge_found:
            yield Actions.Shift

    def create_edge_action(self, edge, direction):
        self.edge_found = True
        return Action.edge_action(direction, edge.attrib.get("remote"), edge.tag,
                                  orig_edge=edge, oracle=self)

    def create_node_action(self, edge, node, action):
        self.edge_found = True
        return action(edge.tag, orig_edge=edge, orig_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.discard(edge)
        if node is not None:
            self.nodes_remaining.discard(node.ID)

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (" ".join(self.nodes_remaining), sep,
                                                       " ".join(map(str, self.edges_remaining)))

    def __str__(self):
        return str(" ")
