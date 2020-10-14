from enum import Enum
from itertools import groupby
from operator import attrgetter


ROOT_ID = -1
ROOT_LAB = "TOP"
ANCHOR_LAB = "ANCHOR"
DEFAULT_LABEL = "name"


class Direction(Enum):
    incoming = 0
    outgoing = 1


def contains(s, lab):
    return s is not None and (s.match(lab) if hasattr(s, "match") else lab == s if isinstance(s, str) else lab in s)


def incoming_labs(node, except_edge):
    return getattr(node, "incoming_labs", set(e.lab for e in node.incoming if e != except_edge))


def outgoing_labs(node, except_edge):
    return getattr(node, "outgoing_labs", set(e.lab for e in node if e != except_edge))


def labs(node, except_edge, direction):
    return incoming_labs(node, except_edge) if direction == Direction.incoming else outgoing_labs(node, except_edge)


# Generic class to define rules on allowed incoming/outgoing edge tags based on triggers
class LabRule:
    def __init__(self, trigger, allowed=None, disallowed=None):
        self.trigger = trigger
        self.allowed = allowed
        self.disallowed = disallowed

    def violation(self, node, lab, direction, message=False):
        edge, lab = (lab, lab.lab) if hasattr(lab, "lab") else (None, lab)
        for d in Direction:
            trigger = self.trigger.get(d)
            if any(contains(trigger, t) for t in labs(node, edge, d)):  # Trigger on edges that node already has
                allowed = None if self.allowed is None else self.allowed.get(direction)
                if allowed is not None and not contains(allowed, lab):
                    return message and "Units with %s '%s' edges must get only %s '%s' edges, but got '%s' for '%s'" % (
                        d.name, trigger, direction.name, allowed, lab, node)
                disallowed = None if self.disallowed is None else self.disallowed.get(direction)
                if disallowed is not None and contains(disallowed, lab):
                    return message and ("Multiple %s '%s' edges for '%s'" % (d.name, lab, node)
                                        if (d.name, trigger) == (direction.name, disallowed) else
                                        "Units with %s '%s' edges must not get %s '%s' edges, but got '%s' for '%s'" % (
                                            d.name, trigger, direction.name, disallowed, lab, node))
        trigger = self.trigger.get(direction)
        if edge is None and contains(trigger, lab):  # Trigger on edges that node is getting now (it does not exist yet)
            for d in Direction:
                allowed = None if self.allowed is None else self.allowed.get(d)
                if allowed is not None and not all(contains(allowed, t) for t in labs(node, edge, d)):
                    return message and "Units getting %s '%s' edges must have only %s '%s' edges, but '%s' has '%s'" % (
                        direction.name, lab, d.name, allowed, node, ",".join(labs(node, edge, d)))
                disallowed = None if self.disallowed is None else self.disallowed.get(d)
                if disallowed is not None and any(contains(disallowed, t) for t in labs(node, edge, d)):
                    return message and "Units getting %s '%s' edges must not have %s '%s' edges, but '%s' has '%s'" % (
                        direction.name, lab, d.name, disallowed, node, ",".join(labs(node, edge, d)))
        return None


def set_prod(set1, set2=None):
    for x in set1:
        for y in set1 if set2 is None else set2:
            yield x, y


class Valid:
    def __init__(self, valid=True, message=""):
        self.valid = valid
        self.message = message

    def __bool__(self):
        return self.valid

    def __str__(self):
        return self.message

    def __call__(self, valid, message=None):
        return Valid(valid, "; ".join(filter(None, (self.message, message))))


# Generic class to define constraints on parser actions
class Constraints:
    def __init__(self, multigraph=False, require_implicit_childless=True, allow_orphan_terminals=False,
                 top_level_allowed=None, top_level_only=None,
                 possible_multiple_incoming=(), childless_incoming_trigger=(), childless_outgoing_allowed=(ANCHOR_LAB,),
                 unique_incoming=(), unique_outgoing=(), mutually_exclusive_incoming=(), mutually_exclusive_outgoing=(),
                 exclusive_outgoing=(), required_outgoing=(), **kwargs):
        self.multigraph = multigraph
        self.require_implicit_childless = require_implicit_childless
        self.allow_orphan_terminals = allow_orphan_terminals
        self.top_level_allowed = top_level_allowed
        self.top_level_only = top_level_only
        self.possible_multiple_incoming = possible_multiple_incoming
        self.required_outgoing = required_outgoing
        self.tag_rules = \
            [LabRule(trigger={Direction.incoming: childless_incoming_trigger},
                     allowed={Direction.outgoing: childless_outgoing_allowed}),
             LabRule(trigger={Direction.outgoing: exclusive_outgoing},
                     allowed={Direction.outgoing: exclusive_outgoing})] + \
            [LabRule(trigger={Direction.incoming: t}, disallowed={Direction.incoming: t}) for t in unique_incoming] + \
            [LabRule(trigger={Direction.outgoing: t}, disallowed={Direction.outgoing: t}) for t in unique_outgoing] + \
            [LabRule(trigger={Direction.incoming: t1}, disallowed={Direction.incoming: t2})
             for t1, t2 in set_prod(mutually_exclusive_incoming)] + \
            [LabRule(trigger={Direction.outgoing: t1}, disallowed={Direction.outgoing: t2})
             for t1, t2 in set_prod(mutually_exclusive_outgoing)]

    def allow_action(self, action, history):
        return history or action.tag is None  # First action must not create nodes/edges

    def allow_edge(self, edge):
        return True

    def allow_parent(self, node, lab):
        return True

    def allow_child(self, node, lab):
        return True

    def allow_label(self, node, label):
        return True

    def allow_property_value(self, node, property_value):
        return True

    def allow_attribute_value(self, edge, attribute_value):
        return True


def ucca_constraints(**kwargs):
    from .ucca import UccaConstraints
    return UccaConstraints(**kwargs)


def sdp_constraints(**kwargs):
    from .sdp import SdpConstraints
    return SdpConstraints(**kwargs)


def amr_constraints(**kwargs):
    from .amr import AmrConstraints
    return AmrConstraints(**kwargs)


def eds_constraints(**kwargs):
    from .eds import EdsConstraints
    return EdsConstraints(**kwargs)


def ptg_constraints(**kwargs):
    from .ptg import PtgConstraints
    return PtgConstraints(**kwargs)


CONSTRAINTS = {
    "ucca":   ucca_constraints,
    "amr":    amr_constraints,
    "dm":     sdp_constraints,
    "psd":    sdp_constraints,
    "eds":    eds_constraints,
    "ptg":    ptg_constraints,
}


def detect_cycles(graph):
    stack = [list(graph.tops)]
    visited = set()
    path = []
    path_set = set(path)
    while stack:
        for node in stack[-1]:
            if node in path_set:
                yield "Detected cycle (%s)" % "->".join(n.id for n in path)
            elif node not in visited:
                visited.add(node)
                path.append(node)
                path_set.add(node)
                stack.append(node.children)
                break
        else:
            if path:
                path_set.remove(path.pop())
            stack.pop()


def join(edges):
    return ", ".join("%s-[%s]->%s" % (e.parent.id, e.lab, e.child.id) for e in edges)


def check_orphan_terminals(constraints, terminal):
    if not constraints.allow_orphan_terminals:
        if not terminal.incoming:
            yield "Orphan %s terminal (%s) '%s'" % (terminal.lab, terminal.id, terminal)


def check_multigraph(constraints, node):
    if not constraints.multigraph:
        for parent_id, edges in groupby(node.incoming, key=attrgetter("parent.id")):
            edges = list(edges)
            if len(edges) > 1:
                yield "Multiple edges from %s to %s (%s)" % (parent_id, node.id, join(edges))


def check_top_level_only(constraints, node):
    if constraints.top_level_only and not node.is_top:
        for edge in node:
            if edge.lab in constraints.top_level_only:
                yield "Non-top level %s edge (%s)" % (edge.lab, edge)


def check_required_outgoing(constraints, node):
    if constraints.required_outgoing and not any(e.lab in constraints.required_outgoing for e in node):
        yield "Non-terminal without outgoing %s (%s)" % (constraints.required_outgoing, node.id)


def check_tag_rules(constraints, node):
    for edge in node:
        for rule in constraints.tag_rules:
            for violation in (rule.violation(node, edge, Direction.outgoing, message=True),
                              rule.violation(edge.child, edge, Direction.incoming, message=True)):
                if violation:
                    yield "%s (%s)" % (violation, join([edge]))
        valid = constraints.allow_parent(node, edge.lab)
        if not valid:
            yield "%s may not be a '%s' parent (%s, %s): %s" % (
                node.id, edge.lab, join(node.incoming), join(node), valid)
        valid = constraints.allow_child(edge.child, edge.lab)
        if not valid:
            yield "%s may not be a '%s' child (%s, %s): %s" % (
                edge.child.id, edge.lab, join(edge.child.incoming), join(edge.child), valid)
        valid = constraints.allow_edge(edge)
        if not valid:
            "Illegal edge: %s (%s)" % (join([edge]), valid)
