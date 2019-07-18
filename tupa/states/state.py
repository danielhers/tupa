from collections import deque

from graph import Graph

from .edge import StateEdge
from .node import StateNode, compress_anchors, expand_name
from .ref_graph import RefGraph
from ..action import Actions
from ..config import Config, requires_node_labels, requires_node_properties, requires_edge_attributes, \
    requires_anchors, requires_tops
from ..constraints.util import NAME
from ..constraints.validation import CONSTRAINTS, Constraints, Direction, ROOT_LAB, ANCHOR_LAB, DEFAULT_LABEL
from ..model import NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY
from ..recategorization import resolve


class InvalidActionError(AssertionError):
    def __init__(self, *args, is_type=False):
        super().__init__(*args)
        self.is_type = is_type


class State:
    """
    The parser's state, responsible for applying actions and creating the final Graph
    """
    def __init__(self, graph, conllu, target=None):
        """
        :param graph: Graph for training, not needed for testing except for getting the graph id
        :param conllu: Graph with node per token predicted by a syntactic parser
        :param target: target framework string
        """
        if conllu is None:
            raise ValueError("conllu is required for tokens and features")
        self.args = Config().args
        self.input_graph = graph
        self.framework = target or self.input_graph.framework
        self.constraints = CONSTRAINTS.get(self.framework, Constraints)()
        self.has_ref = bool(graph and graph.nodes)
        self.ref_graph = RefGraph(self.input_graph, conllu, self.framework)
        self.root = StateNode.copy(self.ref_graph.root)
        self.terminals = list(map(StateNode.copy, self.ref_graph.terminals))
        self.stack = [self.root]
        self.buffer = deque(self.terminals)
        self.heads = set()
        self.nodes = [self.root] + self.terminals
        self.non_virtual_nodes = []
        self.actions = []  # History of applied actions
        self.log = []
        self.finished = False
        self.need_label = self.need_property = self.need_attribute = self.last_edge = None  # Which edge/node is next
        self.type_validity_cache = {}

    def create_graph(self):
        """
        Create final graph from temporary representation
        :return: Graph created from self.nodes
        """
        Config().print("Creating %s graph %s from state..." % (self.framework, self.input_graph.id), level=2)
        graph = Graph(self.input_graph.id, self.framework)
        graph.input = self.input_graph.input
        for node in self.non_virtual_nodes:
            if node.label is None and requires_node_labels(self.framework):
                node.label = DEFAULT_LABEL
            node.label = resolve(node, node.label)  # Must be before properties in case label is to be resolved to NAME
            if node.properties:
                node.properties = {prop: resolve(node, value) for prop, value in node.properties.items()}
                if self.framework == "amr" and node.label == NAME:
                    node.properties = expand_name(node.properties)
                properties, values = zip(*node.properties.items())
            else:
                properties, values = (None, None)
            node.graph_node = graph.add_node(int(node.id), label=node.label, properties=properties, values=values)
        for edge in self.root.outgoing:
            edge.child.graph_node.is_top = True
        for node in self.non_virtual_nodes:
            anchors = []
            for edge in node.outgoing:
                if edge.child.text is not None:
                    if requires_anchors(self.framework):
                        anchors += edge.child.ref_node.anchors
                else:
                    attributes, values = zip(*edge.attributes.items()) if edge.attributes else (None, None)
                    graph.add_edge(int(edge.parent.id), int(edge.child.id), edge.lab,
                                   attributes=attributes, values=values)
            node.graph_node.anchors = compress_anchors(anchors) if anchors else None
        return graph

    def is_valid_action(self, action):
        """
        :param action: action to check for validity
        :return: is the action (including tag) valid in the current state?
        """
        valid = self.type_validity_cache.get(action.type_id)
        if valid is None:
            try:
                self.check_valid_action(action)
                valid = True
            except InvalidActionError as e:
                valid = False
                if e.is_type:
                    self.type_validity_cache[action.type_id] = valid
        return valid

    def check_valid_action(self, action, message=False):
        """
        Raise InvalidActionError if the action is invalid in the current state
        :param action: action to check for validity
        :param message: whether to add an informative message to the thrown exception
        """
        def _check_possible_node():
            self.check(self.node_ratio() < self.args.max_node_ratio,
                       message and "Non-terminals/terminals ratio: %.3f" % self.args.max_node_ratio, is_type=True)
            for head in self.heads:
                self.check(head.height <= self.args.max_height,
                           message and "Graph height: %d" % self.args.max_height, is_type=True)

        def _check_possible_parent(parent_node, edge_lab):
            self.check(parent_node.text is None, message and "Terminals may not have children: %s" % parent_node.text,
                       is_type=True)
            self.check((edge_lab == ROOT_LAB) == parent_node.is_root,
                       message and "All and only root edges must be '%s'" % ROOT_LAB)
            if self.args.constraints and edge_lab is not None:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(parent_node, edge_lab, Direction.outgoing, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_parent(parent_node, edge_lab),
                           message and "%s may not be a '%s' parent (currently %s)" % (
                               parent_node, edge_lab, ", ".join(map(str, parent_node.outgoing)) or "childless"))

        def _check_possible_child(child_node, edge_lab):
            self.check(not child_node.is_root, message and "Root may not have parents", is_type=True)
            self.check((edge_lab == ANCHOR_LAB) == (child_node.text is not None),
                       message and "All and only terminal edges must be '%s'" % ANCHOR_LAB)
            if self.args.constraints and edge_lab is not None:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(child_node, edge_lab, Direction.incoming, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_child(child_node, edge_lab),
                           message and "%s may not be a '%s' child (currently %s, %s)" % (
                               child_node, edge_lab, ", ".join(map(str, child_node.incoming)) or "parentless",
                               ", ".join(map(str, child_node.outgoing)) or "childless"))

        def _check_possible_edge(parent_node, child_node, edge_lab):
            _check_possible_parent(parent_node, edge_lab)
            _check_possible_child(child_node, edge_lab)
            if self.args.constraints and edge_lab is not None:
                if ROOT_LAB in parent_node.incoming_labs:
                    self.check(self.constraints.top_level_allowed is None or not edge_lab or
                               edge_lab in self.constraints.top_level_allowed,
                               message and "Root may not have %s edges" % edge_lab)
                else:
                    self.check(self.constraints.top_level_only is None or
                               edge_lab not in self.constraints.top_level_only,
                               message and "Only root may have %s edges" % edge_lab)
            self.check(not parent_node.is_root or child_node.text is None,
                       message and "Virtual terminal child %s of virtual root" % child_node, is_type=True)
            if self.constraints.multigraph:  # Nodes may be connected by more than one edge
                edge = StateEdge(parent_node, child_node, edge_lab)
                self.check(self.constraints.allow_edge(edge), message and "Edge not allowed: %s (currently: %s)" % (
                    edge, ", ".join(map(str, parent_node.outgoing)) or "childless"))
            else:  # Simple graph, i.e., no more than one edge between the same pair of nodes
                self.check(child_node not in parent_node.children, message and "%s is already %s's child: %s" % (
                    child_node, parent_node, ", ".join(map(str, parent_node.outgoing))), is_type=True)
            self.check(parent_node not in child_node.descendants, message and "Detected cycle by edge: %s->%s" % (
                parent_node, child_node), is_type=True)

        def _check_possible_label():
            self.check(requires_node_labels(self.framework), message and "Node labels disabled", is_type=True)
            try:
                node_to_label = self.stack[-action.tag]
            except IndexError:
                node_to_label = None
            self.check(node_to_label is not None, message and "Labeling invalid node %s when stack size is %d" % (
                action.tag, len(self.stack)))
            self.check(node_to_label.label is None, message and "Labeling already-labeled node: %s" % node_to_label)
            self.check(node_to_label.text is None, message and "Setting label of virtual terminal: %s" % node_to_label)
            self.check(node_to_label is not self.root, "Setting label of virtual root")

        def _check_possible_property():
            self.check(requires_node_properties(self.framework), message and "Node properties disabled",
                       is_type=True)
            try:
                node_for_prop = self.stack[-action.tag]
            except IndexError:
                node_for_prop = None
            self.check(node_for_prop is not None, message and "Setting property on invalid node %s when stack size "
                                                              "is %d" % (action.tag, len(self.stack)))
            self.check(node_for_prop.text is None, message and "Setting property of virtual terminal: %s" %
                       node_for_prop)
            self.check(node_for_prop is not self.root, "Setting property of virtual root")
            if self.framework == "amr":  # In AMR, properties and values are be evoked by tokens
                max_props = self.args.max_node_ratio * len(self.terminals)
            else:
                max_props = self.args.max_properties_per_node
            self.check(len(node_for_prop.properties or ()) < max_props,
                       message and "Exceeded maximum number of properties per node: %s" % node_for_prop)

        def _check_possible_attribute():
            self.check(requires_edge_attributes(self.framework), message and "Edge attributes disabled")
            self.check(self.last_edge is not None, message and "Setting attribute on edge when no edge exists")
            self.check(self.last_edge.lab not in (ROOT_LAB, ANCHOR_LAB),
                       message and "Setting attribute on %s edge" % self.last_edge.lab)
            self.check(len(self.last_edge.attributes or ()) < self.args.max_attributes_per_edge,
                       message and "Exceeded maximum number of attributes per edge: %s" % self.last_edge)

        if self.args.constraints:
            self.check(self.constraints.allow_action(action, self.actions),
                       message and "Action not allowed: %s " % action + (
                           ("after " + ", ".join("%s" % a for a in self.actions[-3:])) if self.actions else "as first"))
        if action.is_type(Actions.Finish):
            self.check(not self.buffer, "May only finish at the end of the input buffer", is_type=True)
            if self.args.swap and requires_tops(self.framework):  # Without swap, the oracle parse may be incomplete
                self.check(self.root.outgoing or not self.non_virtual_nodes,
                           message and "Root has no child at parse end", is_type=True)
            for node in self.non_virtual_nodes:
                self.check(not self.args.require_connected or ROOT_LAB in node.incoming_labs or
                           node.incoming, message and "Non-terminal %s has no parent at parse end" % node)
                self.check(not requires_node_labels(self.framework) or node.label is not None,
                           message and "Non-terminal %s has no label at parse end (orig node label: '%s')" % (
                               node, node.ref_node.label if node.ref_node else None))
        else:
            self.check(self.action_ratio() < self.args.max_action_ratio,
                       message and "Actions/terminals ratio: %.3f" % self.args.max_action_ratio, is_type=True)
            if action.is_type(Actions.Shift):
                self.check(self.buffer, message and "Shifting from empty buffer", is_type=True)
            elif action.is_type(Actions.Label):
                _check_possible_label()
            elif action.is_type(Actions.Property):
                _check_possible_property()
            elif action.is_type(Actions.Attribute):
                _check_possible_attribute()
            else:   # Unary actions
                self.check(self.stack, message and "%s with empty stack" % action, is_type=True)
                s0 = self.stack[-1]
                if action.is_type(Actions.Reduce):
                    if s0.text is None:
                        self.check(not self.args.require_connected or s0.incoming,
                                   message and "Reducing parentless non-terminal %s" % s0, is_type=True)
                        self.check(not self.constraints.required_outgoing or
                                   s0.outgoing_labs.intersection(self.constraints.required_outgoing),
                                   message and "Reducing non-terminal %s without %s edge" % (
                                       s0, self.constraints.required_outgoing), is_type=True)
                    self.check(not requires_node_labels(self.framework) or s0.text or s0.is_root or s0.label,
                               message and "Reducing non-terminal %s without label" % s0, is_type=True)
                elif action.is_type(Actions.Swap):
                    # A regular swap is possible since the stack has at least two elements;
                    # A compound swap is possible if the stack is longer than the distance
                    distance = action.tag or 1
                    self.check(1 <= distance < len(self.stack), message and "Invalid swap distance: %d" % distance)
                    swapped = self.stack[-distance - 1]
                    # To prevent swap loops: only swap if the nodes are currently in their original order
                    self.check(self.swappable(s0, swapped),
                               message and "Already swapped nodes: %s (swap index %g) <--> %s (swap index %g)"
                               % (swapped, swapped.swap_index, s0, s0.swap_index))
                else:
                    pct = self.get_parent_child_tag(action)
                    self.check(pct, message and "%s with len(stack) = %d" % (action, len(self.stack)), is_type=True)
                    parent, child, tag = pct
                    if parent is None:
                        _check_possible_child(child, tag)
                        _check_possible_node()
                    elif child is None:
                        _check_possible_parent(parent, tag)
                        _check_possible_node()
                    else:  # Binary actions
                        _check_possible_edge(parent, child, tag)

    @staticmethod
    def swappable(right, left):
        return left.swap_index < right.swap_index

    def is_valid_label(self, label):
        """
        :param label: label to check for validity
        :return: is the label valid in the current state?
        """
        try:
            self.check_valid_label(label)
        except InvalidActionError:
            return False
        return True

    def check_valid_label(self, label, message=False):
        self.check(label is not None, message and "None node label")
        if self.args.constraints:
            resolved_label = resolve(self.need_label, label, conservative=True, is_node_label=True)
            valid = self.constraints.allow_label(self.need_label, resolved_label)
            self.check(valid, message and "May not label %s as %s (%s): %s" % (
                self.need_label, label, resolved_label, valid))

    def is_valid_property_value(self, property_value):
        """
        :param property_value: property_value to check for validity
        :return: is the property_value valid in the current state?
        """
        try:
            self.check_valid_property_value(property_value)
        except InvalidActionError:
            return False
        return True

    def check_valid_property_value(self, property_value, message=False):
        self.check(property_value is not None, message and "None property-value pair")
        if self.args.constraints:
            try:
                prop, value = property_value
            except ValueError as e:
                raise ValueError("Invalid property-value pair: " + str(property_value)) from e
            resolved_value = resolve(self.need_property, value, conservative=True, is_node_label=False)
            valid = self.constraints.allow_property_value(self.need_property, (prop, resolved_value))
            self.check(valid, message and "May not set property value for %s to %s=%s (%s): %s" % (
                self.need_property, prop, value, resolved_value, valid))

    def is_valid_attribute_value(self, attribute_value):
        """
        :param attribute_value: attribute_value to check for validity
        :return: is the attribute_value valid in the current state?
        """
        try:
            self.check_valid_attribute_value(attribute_value)
        except InvalidActionError:
            return False
        return True

    def check_valid_attribute_value(self, attribute_value, message=False):
        self.check(attribute_value is not None, message and "None attribute-value pair")
        if self.args.constraints:
            valid = self.constraints.allow_attribute_value(self.need_attribute, attribute_value)
            self.check(valid, message and "May not set attribute value for %s to %s: %s" % (
                self.need_attribute, attribute_value, valid))

    def is_valid_annotation(self, key=None):
        """
        :param key: one of NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY or None
        :return function to check validity of values
        """
        if key is None:
            return self.is_valid_action
        if key == NODE_LABEL_KEY:
            return self.is_valid_label
        if key == NODE_PROPERTY_KEY:
            return self.is_valid_property_value
        if key == EDGE_ATTRIBUTE_KEY:
            return self.is_valid_attribute_value
        raise ValueError("Invalid key: " + str(key))

    @staticmethod
    def check(condition, *args, **kwargs):
        if not condition:
            raise InvalidActionError(*args, **kwargs)

    # noinspection PyTypeChecker
    def transition(self, action):
        """
        Main part of the parser: apply action given by oracle or classifier
        :param action: Action object to apply
        """
        action.apply()
        self.log = []
        pct = self.get_parent_child_tag(action)
        if pct:
            parent, child, tag = pct
            if parent is None:
                parent = action.node = self.add_node(ref_node=action.ref_node)
            if child is None:
                child = action.node = self.add_node(ref_node=action.ref_node)
            action.edge = self.add_edge(StateEdge(parent, child, tag, ref_edge=action.ref_edge))
            if action.node:
                self.buffer.appendleft(action.node)
        elif action.is_type(Actions.Shift):  # Push buffer head to stack; shift buffer
            self.stack.append(self.buffer.popleft())
        elif action.is_type(Actions.Label):
            self.need_label = self.stack[-action.tag]  # The parser is responsible to choose a label and set it
        elif action.is_type(Actions.Property):
            self.need_property = self.stack[-action.tag]  # The parser is responsible to choose a property and set it
        elif action.is_type(Actions.Attribute):
            self.need_attribute = self.last_edge  # The parser is responsible to choose an attribute and set it
        elif action.is_type(Actions.Reduce):  # Pop stack (no more edges to create with this node)
            self.stack.pop()
        elif action.is_type(Actions.Swap):  # Place second (or more) stack item back on the buffer
            distance = action.tag or 1
            s = slice(-distance - 1, -1)
            self.log.append("%s <--> %s" % (", ".join(map(str, self.stack[s])), self.stack[-1]))
            self.buffer.extendleft(reversed(self.stack[s]))  # extendleft reverses the order
            del self.stack[s]
        elif action.is_type(Actions.Finish):  # Nothing left to do
            self.finished = True
        else:
            raise ValueError("Invalid action: %s" % action)
        action.index = len(self.actions)
        self.actions.append(action)
        self.type_validity_cache = {}

    def add_node(self, ref_node=None):
        """
        Called during parsing to add a new StateNode (not graph.Node) to the temporary representation
        :param ref_node: original StateNode() when training
        """
        index = len(self.nodes)
        node = StateNode(index, index if ref_node is None else ref_node.id, swap_index=self.calculate_swap_index(),
                         ref_node=ref_node)
        self.nodes.append(node)
        self.non_virtual_nodes.append(node)
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
            b0 = self.buffer[0]
            if self.stack and (b0.text is not None or b0.swap_index <= len(self.nodes)):
                s0 = self.stack[-1]
                return (s0.swap_index + b0.swap_index) / 2
        return None

    def add_edge(self, edge):
        edge.add()
        self.heads.discard(edge.child)
        self.log.append("edge: %s" % edge)
        self.last_edge = edge
        return edge
    
    PARENT_CHILD = (
        (Actions.LeftEdge, (-1, -2)),
        (Actions.RightEdge, (-2, -1)),
        (Actions.Node, (None, -1)),
        (Actions.Implicit, (-1, None)),
    )

    def get_parent_child_tag(self, action):
        try:
            for action_type, indices in self.PARENT_CHILD:
                if action.is_type(action_type):
                    parent, child = [None if i is None else self.stack[i] for i in indices]
                    break
            else:
                return None
            return parent, child, action.tag
        except IndexError:
            return None

    def assign_node_label(self, label):
        assert self.need_label is not None, "Called assign_node_label() when need_label is None"
        assert label is not None, "Labeling node %s with None label" % self.need_label
        self.need_label.label = label
        self.log.append("label: %s" % self.need_label)
        self.type_validity_cache = {}
        self.need_label = None

    def assign_node_property_value(self, property_value):
        assert self.need_property is not None, "Called assign_node_property() when need_property is None"
        assert property_value is not None, "Assigning node %s with None property-value pair" % self.need_property
        prop, value = property_value
        if self.need_property.properties is None:
            self.need_property.properties = {}
        self.need_property.properties[prop] = value
        self.log.append("property: %s" % self.need_property)
        self.type_validity_cache = {}
        self.need_property = None

    def assign_edge_attribute_value(self, attribute_value):
        assert self.need_attribute is not None, "Called assign_edge_attribute() when need_attribute is None"
        assert attribute_value is not None, "Assigning edge %s with None attribute-value pair" % self.need_attribute
        attrib, value = attribute_value
        if self.need_attribute.attributes is None:
            self.need_attribute.attributes = {}
        self.need_attribute.attributes[attrib] = value
        self.log.append("attribute: %s" % self.need_attribute)
        self.type_validity_cache = {}
        self.need_attribute = None

    def node_ratio(self):
        return (len(self.nodes) / len(self.terminals) - 1) if self.terminals else 0

    def action_ratio(self):
        return (len(self.actions) / len(self.terminals)) if self.terminals else 0

    def str(self, sep):
        return "stack: [%-20s]%sbuffer: [%s]" % (" ".join(map(str, self.stack)), sep,
                                                 " ".join(map(str, self.buffer)))

    def __str__(self):
        return self.str(" ")

    def __eq__(self, other):
        return self.stack == other.stack and self.buffer == other.buffer and self.nodes == other.nodes

    def __hash__(self):
        return hash((tuple(self.stack), tuple(self.buffer), tuple(self.nodes)))
