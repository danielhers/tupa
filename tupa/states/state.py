import sys
from collections import deque, defaultdict

from states.edge import Edge
from states.node import Node
from tupa.action import Actions
from tupa.config import Config
from tupa.constraints import Direction
from ucca import core, layer0, layer1, textutil
from ucca.layer1 import EdgeTags


class State(object):
    """
    The parser's state, responsible for applying actions and creating the final Passage
    :param passage: a Passage object to get the tokens from, and everything else if training
    """
    def __init__(self, passage):
        self.args = Config().args
        self.constraints = Config().constraints
        self.log = []
        self.finished = False
        l0 = passage.layer(layer0.LAYER_ID)
        l1 = passage.layer(layer1.LAYER_ID)
        assert l0.all, "Empty passage '%s'" % passage.ID
        self.labeled = len(l1.all) > 1
        self.terminals = [Node(i + 1, orig_node=t, text=t.text, paragraph=t.paragraph, tag=t.tag,
                               pos_tag=t.extra.get(textutil.TAG_KEY),
                               dep_rel=t.extra.get(textutil.DEP_KEY),
                               dep_head=t.extra.get(textutil.HEAD_KEY),
                               lemma=t.extra.get(textutil.LEMMA_KEY))
                          for i, t in enumerate(l0.all)]
        self.stack = []
        self.buffer = deque()
        self.nodes = []
        self.heads = set()
        self.root = self.add_node(orig_node=l1.heads[0])  # The root is not part of the buffer
        self.stack.append(self.root)
        self.buffer += self.terminals
        self.nodes += self.terminals
        self.passage_id = passage.ID
        self.actions = []  # History of applied actions
        self.labels = set()  # Set of resolved labels for all nodes

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
            self.assert_node_ratio(extra=1)
            self.assert_height()
            assert self.constraints.allow_node(Node(None, label=action.label), self.labels), \
                "May not create node with label %s (existing: %s)" % (action.label, ",".join(filter(None, self.labels)))

        def assert_possible_parent(node):
            assert node.text is None, "Terminals may not have children: %s" % node.text
            if self.args.constraints:
                assert not self.constraints.require_implicit_childless or not node.implicit, \
                    "Implicit nodes may not have children: %s" % s0
                for rule in self.constraints.tag_rules:
                    rule.check(node, action.tag, Direction.outgoing)

        def assert_possible_child(node):
            assert node is not self.root, "The root may not have parents"
            assert (node.text is not None) == (action.tag == EdgeTags.Terminal), \
                "Edge tag must be %s iff child is terminal, but node is %s and edge tag is %s" % (
                    EdgeTags.Terminal, node, action.tag)
            if self.args.constraints:
                for rule in self.constraints.tag_rules:
                    rule.check(node, action.tag, Direction.incoming)
                assert self.constraints.possible_multiple_incoming is None or \
                    action.remote or action.tag in self.constraints.possible_multiple_incoming or \
                    all(e.remote or e.tag in self.constraints.possible_multiple_incoming for e in node.incoming), \
                    "Multiple parents only allowed if they are remote or linkage edges: %s, %s" % (action, node)

        def assert_possible_edge():
            parent, child = self.get_parent_child(action)
            assert_possible_parent(parent)
            assert_possible_child(child)
            if parent is self.root and self.args.constraints:
                assert self.constraints.allow_root_terminal_children or child.text is None, \
                    "Root may not have terminal children, but is being added '%s'" % child
                assert self.constraints.top_level is None or action.tag in self.constraints.top_level, \
                    "The root may not have %s edges" % action.tag
            if self.args.constraints and self.constraints.allow_multiple_edges:
                edge = Edge(parent, child, action.tag, remote=action.remote)
                assert edge not in parent.outgoing, "Edge must not already exist: %s" % edge
            else:
                assert child not in parent.children, "Edge must not already exist: %s->%s" % (parent, child)
            assert parent not in child.descendants, "Detected cycle created by edge: %s->%s" % (parent, child)

        if action.is_type(Actions.Finish):
            if self.args.swap:  # Without swap, the oracle may be incapable even of single action
                assert self.root.outgoing, \
                    "Root must have at least one child at the end of the parse, but has none"
                if self.args.constraints and self.constraints.require_connected:
                    for n in self.nodes:
                        assert n is self.root or n.is_linkage or n.text or n.incoming, \
                            "Non-terminal %s must have at least one parent at the end of the parse, but has none" % n
        elif action.is_type(Actions.Shift):
            assert self.buffer, "Buffer must not be empty in order to shift from it"
        else:  # Unary actions
            if self.args.constraints and self.constraints.require_first_shift:
                assert self.actions, "First action must be %s, but was %s" % (Actions.Shift, action)
            assert self.stack, "Action requires non-empty stack: %s" % action
            s0 = self.stack[-1]
            if action.is_type(Actions.Node, Actions.RemoteNode):
                assert_possible_child(s0)
                assert_possible_node()
            elif action.is_type(Actions.Implicit):
                assert_possible_parent(s0)
                assert_possible_node()
            elif action.is_type(Actions.Reduce):
                assert s0 is not self.root or s0.outgoing, "May not reduce the root without children"
                if self.args.constraints and self.constraints.require_connected:
                    assert s0 is self.root or s0.is_linkage or s0.text or s0.incoming, \
                        "May not reduce a non-terminal node without incoming edges"
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
                        "Swapping already-swapped nodes: %s (swap index %g) <--> %s (swap index %g)" % (
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
        elif action.is_type(Actions.Node, Actions.RemoteNode):  # Create new parent node and add to the buffer
            parent = self.add_node(orig_node=action.orig_node, label=action.label)
            self.add_edge(Edge(parent, self.stack[-1], action.tag, remote=action.remote))
            self.buffer.appendleft(parent)
        elif action.is_type(Actions.Implicit):  # Create new child node and add to the buffer
            child = self.add_node(orig_node=action.orig_node, label=action.label, implicit=True)
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
            raise Exception("Invalid action: %s" % action)
        if self.args.verify:
            intersection = set(self.stack).intersection(self.buffer)
            assert not intersection, "Stack and buffer overlap: %s" % intersection
        self.assert_node_ratio()
        action.index = len(self.actions)
        self.actions.append(action)

    def add_node(self, **kwargs):
        """
        Called during parsing to add a new Node (not core.Node) to the temporary representation
        :param kwargs: keyword arguments for Node()
        """
        node = Node(len(self.nodes), swap_index=self.calculate_swap_index(), **kwargs)
        if self.args.verify:
            assert node not in self.nodes, "Node already exists"
        self.nodes.append(node)
        self.heads.add(node)
        self.log.append("node: %s (swap_index: %g)" % (node, node.swap_index))
        return node

    def calculate_swap_index(self):
        """
        Update a new node's swap index according to the nodes before and after it.
        Usually the swap index is just the index, i.e., len(self.nodes).
        If the buffer is not empty and its head is not a terminal, it means that it is a non-terminal created before.
        In that case, the buffer head's index will be lower than the new node's index, so the new node's swap index will
        be the arithmetic mean between the previous node (stack top) and the next node (buffer head).
        Then, in the validity check on the SWAP action, we will correctly identify this node as always having appearing
        before the current buffer head. Otherwise, we would prevent swapping them even though it should be valid
        (because they have never been swapped before).
        """
        if self.buffer:
            b = self.buffer[0]
            if self.stack and (b.text is not None or b.swap_index <= len(self.nodes)):
                s = self.stack[-1]
                return (s.swap_index + b.swap_index) / 2
        return None

    def add_edge(self, edge):
        edge.add()
        self.heads.discard(edge.child)
        if self.args.constraints:
            self.labels.add(self.constraints.resolve_label(edge.parent))
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
        self.root.node = l1.heads[0]
        if self.labeled:  # We have a reference passage
            l1.heads[0].extra["remarks"] = self.root.node_id  # For reference
            self.fix_terminal_tags(terminals)
        remotes = []  # To be handled after all nodes are created
        linkages = []  # To be handled after all non-linkage nodes are created
        self.topological_sort()  # Sort self.nodes
        for node in self.nodes:
            if self.labeled and assert_proper:
                assert node.text or node.outgoing or node.implicit, "Non-terminal leaf node: %s" % node
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
                l1.add_remote(node.node, edge.tag, edge.child.node)
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
                if self.args.verbose > 1:
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
        max_ratio = self.args.max_nodes
        assert self.node_ratio(extra=extra) <= max_ratio, \
            "Reached maximum ratio (%.3f) of non-terminals to terminals" % max_ratio

    def assert_height(self):
        max_height = self.args.max_height
        for head in self.heads:
            assert head.height <= max_height, "Reached maximum graph height (%d)" % max_height

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
