from enum import Enum

from ucca.layer1 import EdgeTags


class Direction(Enum):
    incoming = 0
    outgoing = 1


def contains(s, tag):
    return s is not None and (s.match(tag) if hasattr(s, "match") else tag == s if isinstance(s, str) else tag in s)


def tags(node, direction):
    return node.incoming_tags if direction == Direction.incoming else node.outgoing_tags


class TagRule:
    def __init__(self, trigger, allowed=None, disallowed=None):
        self.trigger = trigger
        self.allowed = allowed
        self.disallowed = disallowed

    def check(self, node, tag, direction):
        for d in Direction:
            trigger = self.trigger.get(d)
            if any(contains(trigger, t) for t in tags(node, d)):  # Trigger on edges that node already has
                allowed = None if self.allowed is None else self.allowed.get(direction)
                if allowed is not None:
                    assert contains(allowed, tag), \
                        "Units with %s '%s' edges must get only %s '%s' edges, but got '%s' for '%s'" % (
                            d.name, trigger, direction.name, allowed, tag, node)
                disallowed = None if self.disallowed is None else self.disallowed.get(direction)
                if disallowed is not None:
                    assert not contains(disallowed, tag), \
                        "Units with %s '%s' edges must not get %s '%s' edges, but got '%s' for '%s'" % (
                            d.name, trigger, direction.name, disallowed, tag, node)
        trigger = self.trigger.get(direction)
        if contains(trigger, tag):  # Trigger on edges that node is getting now
            for d in Direction:
                allowed = None if self.allowed is None else self.allowed.get(d)
                if allowed is not None:
                    assert all(contains(allowed, t) for t in tags(node, d)), \
                        "Units getting %s '%s' edges must have only %s '%s' edges, but '%s' has '%s'" % (
                            direction.name, tag, d.name, allowed, node, tags(node, d))
                disallowed = None if self.disallowed is None else self.disallowed.get(d)
                if disallowed is not None:
                    assert not any(contains(disallowed, t) for t in tags(node, d)), \
                        "Units getting %s '%s' edges must not have %s '%s' edges, but got '%s' has '%s'" % (
                            direction.name, tag, d.name, disallowed, node, tags(node, d))


def set_prod(set1, set2=None):
    for x in set1:
        for y in set1 if set2 is None else set2:
            yield x, y


class Constraints(object):
    def __init__(self, args, require_first_shift=True, require_connected=True, require_implicit_childless=True,
                 allow_root_terminal_children=False, allow_multiple_edges=False,
                 top_level={EdgeTags.ParallelScene, EdgeTags.Linker, EdgeTags.Function, EdgeTags.Ground,
                            EdgeTags.Punctuation},
                 possible_multiple_incoming={EdgeTags.LinkArgument, EdgeTags.LinkRelation},
                 childless_incoming_trigger=EdgeTags.Function,
                 childless_outgoing_allowed={EdgeTags.Terminal, EdgeTags.Punctuation},
                 unique_incoming={EdgeTags.Function, EdgeTags.Ground, EdgeTags.ParallelScene, EdgeTags.Linker,
                                  EdgeTags.LinkRelation, EdgeTags.Connector, EdgeTags.Punctuation, EdgeTags.Terminal},
                 unique_outgoing={EdgeTags.LinkRelation, EdgeTags.Process, EdgeTags.State},
                 mutually_exclusive_incoming=(),
                 mutually_exclusive_outgoing={EdgeTags.Process, EdgeTags.State}):
        self.args = args
        self.require_first_shift = require_first_shift and not args.implicit
        self.require_connected = require_connected
        self.require_implicit_childless = require_implicit_childless
        self.allow_root_terminal_children = allow_root_terminal_children
        self.allow_multiple_edges = allow_multiple_edges
        self.top_level = top_level
        self.possible_multiple_incoming = possible_multiple_incoming if self.args.linkage else ()
        self.tag_rules = \
            [TagRule(trigger={Direction.incoming: childless_incoming_trigger},
                     allowed={Direction.outgoing: childless_outgoing_allowed})] + \
            [TagRule(trigger={Direction.incoming: t}, disallowed={Direction.incoming: t}) for t in unique_incoming] + \
            [TagRule(trigger={Direction.outgoing: t}, disallowed={Direction.outgoing: t}) for t in unique_outgoing] + \
            [TagRule(trigger={Direction.incoming: t1}, disallowed={Direction.incoming: t2})
             for t1, t2 in set_prod(mutually_exclusive_incoming)] + \
            [TagRule(trigger={Direction.outgoing: t1}, disallowed={Direction.outgoing: t2})
             for t1, t2 in set_prod(mutually_exclusive_outgoing)]
    # LinkerIncoming = {EdgeTags.Linker, EdgeTags.LinkRelation}
    # TagRule(trigger=(LinkerIncoming, None), allowed=(LinkerIncoming, None)),  # disabled due to passage 106 unit 1.300
