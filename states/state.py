import sys
from collections import deque, defaultdict

from parsing.action import Actions
from parsing.config import Config
from parsing.constants import Constraints
from states.edge import Edge
from states.node import Node
from ucca import core, layer0, layer1, textutil
from ucca.layer1 import EdgeTags


class State(object):
    """
    The parser's state, responsible for applying actions and creating the final Passage
    :param passage: a Passage object to get the tokens from, and everything else if training
    """
    def __init__(self, passage):
        self.log = []
        self.finished = False
        l0 = passage.layer(layer0.LAYER_ID)
        l1 = passage.layer(layer1.LAYER_ID)
        assert l0.all, "Empty passage '%s'" % passage.ID
        self.labeled = len(l1.all) > 1
        self.terminals = [Node(i, orig_node=t, text=t.text, paragraph=t.paragraph, tag=t.tag,
                               pos_tag=t.extra.get(textutil.TAG_KEY),
                               dep_rel=t.extra.get(textutil.DEP_KEY),
                               dep_head=t.extra.get(textutil.HEAD_KEY))
                          for i, t in enumerate(l0.all)]
        self.nodes = list(self.terminals)  # Copy the list of terminals; more nodes will be added later
        self.buffer = deque(self.nodes)  # Copy the list of nodes to initialize the buffer
        self.root = self.add_node(l1.heads[0])  # The root is not part of the buffer
        self.stack = [self.root]
        self.passage_id = passage.ID
        self.actions = []  # History of applied actions

    def is_valid(self, action):
        """
        :param action: action to check for validity
        :return: is the action (including tag) valid in the current state?
        """
        try:
            self.assert_valid(action)
        except AssertionError:
            return False
        return True

    def assert_valid(self, action):
        """
        Raise AssertionError if the action is invalid in the current state
        :param action: action to check for validity
        """
        def assert_possible_node():
            if self.labeled:  # We're in training, so we must have an original node to refer to
                assert action.orig_node is not None, "May only create real nodes during training"
            self.assert_node_ratio(extra=1)
            self.assert_height()

        def assert_possible_parent(node):
            assert node.text is None, "Terminals may not have children: %s" % node.text
            assert not node.implicit, "Implicit nodes may not have children: %s" % s0
            if Config().args.constraints:
                assert action.tag not in Constraints.UniqueOutgoing or action.tag not in node.outgoing_tags, \
                    "Outgoing edge tag %s must be unique, but %s already has one" % (
                        action.tag, node)
                assert action.tag not in Constraints.MutuallyExclusiveOutgoing or not \
                    node.outgoing_tags & Constraints.MutuallyExclusiveOutgoing, \
                    "Outgoing edge tags %s are mutually exclusive, but %s already has %s and is being added %s" % (
                        Constraints.MutuallyExclusiveOutgoing, node, node.outgoing_tags, action.tag)
                assert action.tag in Constraints.ChildlessOutgoing or not \
                    node.incoming_tags & Constraints.ChildlessIncoming, \
                    "Units with incoming %s edges may not have children, and %s has incoming %s" % (
                        Constraints.ChildlessIncoming, node, node.incoming_tags)

        def assert_possible_child(node):
            assert node is not self.root, "The root may not have parents"
            assert (node.text is not None) == (action.tag == EdgeTags.Terminal), \
                "Edge tag must be %s iff child is terminal, but node is %s and edge tag is %s" % (
                    EdgeTags.Terminal, node, action.tag)
            if Config().args.constraints:
                assert action.tag not in Constraints.UniqueIncoming or \
                    action.tag not in node.incoming_tags, \
                    "Incoming edge tag %s must be unique, but %s already has one" % (
                        action.tag, node)
                assert action.tag not in Constraints.ChildlessIncoming or \
                    node.outgoing_tags <= Constraints.ChildlessOutgoing, \
                    "Units with incoming %s edges may not have children, but %s has %d" % (
                        Constraints.ChildlessIncoming, node, len(node.children))
                assert action.remote or action.tag in Constraints.possible_multiple_incoming() or \
                    all(e.remote or e.tag in Constraints.possible_multiple_incoming()
                        for e in node.incoming), \
                    "Multiple parents only allowed if they are remote or linkage edges: %s, %s" % (
                        action, node)
                # Commented out due to passage 106, unit 1.300
                # assert not node.incoming_tags or (action.tag in Constraints.LinkerIncoming) == (
                #     node.incoming_tags <= Constraints.LinkerIncoming), \
                #     "Linker units may only have incoming edges with tags from %s, but %s is being added '%s'" % (
                #         Constraints.LinkerIncoming, node, action.tag)

        def assert_possible_edge():
            parent, child = self.get_parent_child(action)
            assert_possible_parent(parent)
            assert_possible_child(child)
            if parent is self.root and Config().args.constraints:
                assert child.text is None, "Root may not have terminal children, but is being added '%s'" % child
                assert action.tag in Constraints.TopLevel, "The root may not have %s edges" % action.tag
            # if Config().args.multiple_edges:  # Removed this option because it is not useful right now
            #     edge = Edge(parent, child, action.tag, remote=action.remote)
            #     assert edge not in parent.outgoing, "Edge must not already exist: %s" % edge
            # else:
            assert child not in parent.children, "Edge must not already exist: %s->%s" % (parent, child)
            assert parent not in child.descendants, "Detected cycle created by edge: %s->%s" % (parent, child)

        if action.is_type(Actions.Finish):
            if Config().args.swap:  # Without swap, the oracle may be incapable even of single action
                assert self.root.outgoing, \
                    "Root must have at least one child at the end of the parse, but has none"
        elif action.is_type(Actions.Shift):
            assert self.buffer, "Buffer must not be empty in order to shift from it"
        else:  # Unary actions
            assert self.actions, "First action must be Shift, but was %s" % action
            assert self.stack, "Action requires non-empty stack: %s" % action
            s0 = self.stack[-1]
            if action.is_type(Actions.Node):
                assert_possible_child(s0)
                assert_possible_node()
            elif action.is_type(Actions.Implicit):
                assert_possible_parent(s0)
                assert_possible_node()
            elif action.is_type(Actions.Reduce):
                assert s0 is not self.root or s0.outgoing, "May not reduce the root without children"
                # Commented out due to passage 126, unit 1.338
                # assert not s0.outgoing_tags & Constraints.SceneSufficientOutgoing and \
                #     not s0.incoming_tags & Constraints.SceneSufficientIncoming or \
                #     s0.outgoing_tags & Constraints.SceneNecessaryOutgoing, \
                #     "May not reduce a scene before it has any outgoing edge of %s (it has only %s)" % (
                #         Constraints.SceneNecessaryOutgoing, s0.outgoing_tags)
                # Commented out due to passage 126, unit 1.60
                # assert s0.incoming_tags == Constraints.LinkerIncoming or not \
                #     s0.incoming_tags & Constraints.LinkerIncoming, \
                #     "May not reduce a linker before it has all incoming edges of %s (it has only %s)" % (
                #         Constraints.LinkerIncoming, s0.incoming_tags)
            else:  # Binary actions
                assert len(self.stack) > 1, "Action requires at least two stack elements: %s" % action
                if action.is_type(Actions.LeftEdge, Actions.RightEdge, Actions.LeftRemote, Actions.RightRemote):
                    assert_possible_edge()
                elif action.is_type(Actions.Swap):
                    # A regular swap is possible since the stack has at least two elements;
                    # A compound swap is possible if the stack is longer than the distance
                    distance = action.tag or 1
                    assert 1 <= distance < len(self.stack), "Invalid swap distance: %d" % distance
                    swapped = self.stack[-distance - 1]
                    # To prevent swap loops: only swap if the nodes are currently in their original order
                    assert self.swappable(s0, swapped),\
                        "Swapping already-swapped nodes: %s (swap index %d) <--> %s (swap index %d)" % (
                            swapped, swapped.swap_index, s0, s0.swap_index)
                else:
                    raise Exception("Invalid action: %s" % action)

    @staticmethod
    def swappable(right, left):
        return left.swap_index < right.swap_index

    # noinspection PyTypeChecker
    def transition(self, action):
        """
        Main part of the parser: apply action given by oracle or classifier
        :param action: Action object to apply
        """
        action.apply()
        self.log = []
        if action.is_type(Actions.Shift):  # Push buffer head to stack; shift buffer
            self.stack.append(self.buffer.popleft())
        elif action.is_type(Actions.Node):  # Create new parent node and add to the buffer
            parent = self.add_node(action.orig_node)
            self.update_swap_index(parent)
            self.add_edge(Edge(parent, self.stack[-1], action.tag))
            self.buffer.appendleft(parent)
        elif action.is_type(Actions.Implicit):  # Create new child node and add to the buffer
            child = self.add_node(action.orig_node, implicit=True)
            self.update_swap_index(child)
            self.add_edge(Edge(self.stack[-1], child, action.tag))
            self.buffer.appendleft(child)
        elif action.is_type(Actions.Reduce):  # Pop stack (no more edges to create with this node)
            self.stack.pop()
        elif action.is_type(Actions.LeftEdge, Actions.LeftRemote, Actions.RightEdge, Actions.RightRemote):
            parent, child = self.get_parent_child(action)
            self.add_edge(Edge(parent, child, action.tag, remote=action.remote))
        elif action.is_type(Actions.Swap):  # Place second (or more) stack item back on the buffer
            distance = action.tag or 1
            s = slice(-distance - 1, -1)
            self.log.append("%s <--> %s" % (", ".join(map(str, self.stack[s])), self.stack[-1]))
            self.buffer.extendleft(reversed(self.stack[s]))  # extendleft reverses the order
            del self.stack[s]
        elif action.is_type(Actions.Finish):  # Nothing left to do
            self.finished = True
        else:
            raise Exception("Invalid action: " + action)
        if Config().args.verify:
            intersection = set(self.stack).intersection(self.buffer)
            assert not intersection, "Stack and buffer overlap: %s" % intersection
        self.assert_node_ratio()
        action.index = len(self.actions)
        self.actions.append(action)

    def add_node(self, *args, **kwargs):
        """
        Called during parsing to add a new Node (not core.Node) to the temporary representation
        :param args: ordinal arguments for Node()
        :param kwargs: keyword arguments for Node()
        """
        node = Node(len(self.nodes), *args, **kwargs)
        if Config().args.verify:
            assert node not in self.nodes, "Node already exists"
        self.nodes.append(node)
        self.log.append("node: %s" % node)
        return node

    def add_edge(self, edge):
        edge.add()
        self.log.append("edge: %s" % edge)

    def get_parent_child(self, action):
        if action.is_type(Actions.LeftEdge, Actions.LeftRemote):
            return self.stack[-1], self.stack[-2]
        elif action.is_type(Actions.RightEdge, Actions.RightRemote):
            return self.stack[-2], self.stack[-1]
        else:
            return None, None

    def create_passage(self, assert_proper=True):
        """
        Create final passage from temporary representation
        :param assert_proper: fail if this results in an improper passage
        :return: core.Passage created from self.nodes
        """
        passage = core.Passage(self.passage_id)
        l0 = layer0.Layer0(passage)
        terminals = [l0.add_terminal(text=terminal.text, punct=terminal.tag == layer0.NodeTags.Punct,
                                     paragraph=terminal.paragraph) for terminal in self.terminals]
        l1 = layer1.Layer1(passage)
        if self.labeled:  # We are in training and we have a gold passage
            l1.heads[0].extra["remarks"] = self.root.node_id  # For reference
            self.fix_terminal_tags(terminals)
        remotes = []  # To be handled after all nodes are created
        linkages = []  # To be handled after all non-linkage nodes are created
        self.topological_sort()  # Sort self.nodes
        for node in self.nodes:
            if self.labeled and assert_proper:
                assert node.text or node.outgoing or node.implicit, "Non-terminal leaf node: %s" % node
                assert node.node or node is self.root or node.is_linkage, "Non-root without incoming: %s" % node
            if node.is_linkage:
                linkages.append(node)
            else:
                for edge in node.outgoing:
                    if edge.remote:
                        remotes.append((node, edge))
                    else:
                        edge.child.add_to_l1(l1, node, edge.tag, terminals, self.labeled)

        for node, edge in remotes:  # Add remote edges
            try:
                assert node.node is not None, "Remote edge from nonexistent node"
                assert edge.child.node is not None, "Remote edge to nonexistent node"
                node.node.add(edge.tag, edge.child.node, edge_attrib={"remote": True})
            except AssertionError:
                if assert_proper:
                    raise

        for node in linkages:  # Add linkage nodes and edges
            try:
                link_relation = None
                link_args = []
                for edge in node.outgoing:
                    assert edge.child.node is not None, "Linkage edge to nonexistent node"
                    if edge.tag == EdgeTags.LinkRelation:
                        assert link_relation is None, \
                            "Multiple link relations: %s, %s" % (link_relation, edge.child.node)
                        link_relation = edge.child.node
                    elif edge.tag == EdgeTags.LinkArgument:
                        link_args.append(edge.child.node)
                assert link_relation is not None, "No link relations: %s" % node
                # if len(link_args) < 2:
                #     print("Less than two link arguments for linkage %s" % node, file=sys.stderr)
                node.node = l1.add_linkage(link_relation, *link_args)
                if node.node_id:  # We are in training and we have a gold passage
                    node.node.extra["remarks"] = node.node_id  # For reference
            except AssertionError:
                if assert_proper:
                    raise

        return passage

    def fix_terminal_tags(self, terminals):
        for terminal, orig_terminal in zip(terminals, self.terminals):
            if terminal.tag != orig_terminal.tag:
                if Config().args.verbose:
                    print("%s is the wrong tag for terminal: %s" % (terminal.tag, terminal.text),
                          file=sys.stderr)
                terminal.tag = orig_terminal.tag

    def topological_sort(self):
        """
        Sort self.nodes topologically, each node appearing as early as possible
        Also sort each node's outgoing and incoming edge according to the node order
        """
        levels = defaultdict(list)
        level_by_index = {}
        stack = [node for node in self.nodes if not node.outgoing]
        while stack:
            node = stack.pop()
            if node.index not in level_by_index:
                parents = [edge.parent for edge in node.incoming]
                if parents:
                    unexplored_parents = [parent for parent in parents
                                          if parent.index not in level_by_index]
                    if unexplored_parents:
                        for parent in unexplored_parents:
                            stack.append(node)
                            stack.append(parent)
                    else:
                        level = 1 + max(level_by_index[parent.index] for parent in parents)
                        levels[level].append(node)
                        level_by_index[node.index] = level
                else:
                    levels[0].append(node)
                    level_by_index[node.index] = 0
        self.nodes = [node for level, level_nodes in sorted(levels.items())
                      for node in sorted(level_nodes, key=lambda x: x.node_index or x.index)]
        for node in self.nodes:
            node.outgoing.sort(key=lambda x: x.child.node_index or self.nodes.index(x.child))
            node.incoming.sort(key=lambda x: x.parent.node_index or self.nodes.index(x.parent))

    def node_ratio(self, extra=0):
        return (len(self.nodes) + extra) / len(self.terminals) - 1

    def assert_node_ratio(self, extra=0):
        max_ratio = Config().args.max_nodes
        assert self.node_ratio(extra=extra) <= max_ratio, \
            "Reached maximum ratio (%.3f) of non-terminals to terminals" % max_ratio

    def assert_height(self):
        max_height = Config().args.max_height
        assert self.root.height <= max_height, \
            "Reached maximum graph height (%d)" % max_height

    def update_swap_index(self, node):
        """
        Update the node's swap index according to the nodes before and after it.
        Usually the swap index is just the index, and that is what it is initialized to.
        If the buffer is not empty and the next node on it is not a terminal, it means that it is
        a non-terminal that was created at some point, probably before this node (because this method
        should be run just when this node is created).
        In that case, the buffer head's index will be lower than this node's index, and we will
        update this node's swap index to be the arithmetic average between the previous node
        (stack top) and the next node (buffer head).
        This will make sure that when we perform the validity check on the SWAP action,
        we will correctly identify this node as always having appearing before b (what is the
        current buffer head). Otherwise, we would prevent swapping this node with b even though
        it should be a legal action (because they have never been swapped before).
        :param node: the new node that was added
        """
        if self.buffer:
            b = self.buffer[0]
            if self.stack and (b.text is not None or b.swap_index <= node.swap_index):
                s = self.stack[-1]
                node.swap_index = (s.swap_index + b.swap_index) / 2

    def str(self, sep):
        return "stack: [%-20s]%sbuffer: [%s]" % (" ".join(map(str, self.stack)), sep,
                                                 " ".join(map(str, self.buffer)))

    def __str__(self):
        return self.str(" ")

    def __eq__(self, other):
        return self.stack == other.stack and self.buffer == other.buffer and \
               self.nodes == other.nodes

    def __hash__(self):
        return hash((tuple(self.stack), tuple(self.buffer), tuple(self.nodes)))
