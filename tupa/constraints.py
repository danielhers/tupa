from ucca.layer1 import EdgeTags


class TagRule:
    def __init__(self, trigger, allowed=(None, None), disallowed=(None, None)):
        self.trigger = trigger
        self.allowed = allowed
        self.disallowed = disallowed

    @staticmethod
    def contains(s, tag):
        return s is not None and (s.match(tag) if hasattr(s, "match") else tag == s if isinstance(s, str) else tag in s)

    @staticmethod
    def tags(node, direction):
        return (node.incoming_tags, node.outgoing_tags)[direction]

    def check(self, node, tag, direction):
        for d in Constraints.DIRECTIONS:
            if any(self.contains(self.trigger[d], t) for t in self.tags(node, d)):  # Trigger on what node already has
                if self.allowed[direction] is not None:
                    assert self.contains(self.allowed[direction], tag), \
                        "Units with %s '%s' edges must get only %s '%s' edges, but got '%s' for '%s'" % (
                            Constraints.TITLE[d], self.trigger[d], Constraints.TITLE[direction],
                            self.allowed[direction], tag, node)
                if self.disallowed[direction] is not None:
                    assert not self.contains(self.disallowed[direction], tag), \
                        "Units with %s '%s' edges must not get %s '%s' edges, but got '%s' for '%s'" % (
                            Constraints.TITLE[d], self.trigger[d], Constraints.TITLE[direction],
                            self.disallowed[direction], tag, node)
        if self.contains(self.trigger[direction], tag):  # Trigger on what node is getting now
            for d in Constraints.DIRECTIONS:
                if self.allowed[d] is not None:
                    assert all(self.contains(self.allowed[d], t) for t in self.tags(node, d)), \
                        "Units getting %s '%s' edges must have only %s '%s' edges, but got '%s' for '%s'" % (
                            Constraints.TITLE[d], tag, Constraints.TITLE[direction],
                            self.allowed[d], self.trigger[direction], node)
                if self.disallowed[d] is not None:
                    assert not any(self.contains(self.disallowed[d], t) for t in self.tags(node, d)), \
                        "Units getting %s '%s' edges must not have %s '%s' edges, but got '%s' for '%s'" % (
                            Constraints.TITLE[d], tag, Constraints.TITLE[direction],
                            self.disallowed[d], self.trigger[direction], node)


class Constraints(object):
    INCOMING = 0
    OUTGOING = 1
    DIRECTIONS = (INCOMING, OUTGOING)
    TITLE = "incoming", "outgoing"

    # Require all non-roots to have incoming edges
    require_connected = True

    # Implicit nodes may not have children
    require_implicit_childless = True

    # Disallow terminal children to the root
    allow_root_terminal_children = False

    # Allow multiple edges (with different tags) between the same pair of nodes
    allow_multiple_edges = False

    # A unit may not have more than one incoming edge with the same tag, if it is one of these:
    UniqueIncoming = {
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
        EdgeTags.Connector,
        EdgeTags.Punctuation,
        # EdgeTags.Terminal,
    }

    # A unit may not have more than one outgoing edge with the same tag, if it is one of these:
    UniqueOutgoing = {
        EdgeTags.LinkRelation,
        EdgeTags.Process,
        EdgeTags.State,
    }

    # A unit may not have any children if it has any of these incoming edge tags:
    ChildlessIncoming = {
        EdgeTags.Function,
    }

    # A childless unit may still have these outgoing edge tags:
    ChildlessOutgoing = {
        EdgeTags.Terminal,
        EdgeTags.Punctuation,
    }

    # A linker may only have incoming edges with these tags, and must have both:
    LinkerIncoming = {
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
    }

    # Outgoing edges from the root may only have these tags:
    TopLevel = {
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.Punctuation,
    }

    # Only a unit with one of these incoming tags may also have another non-remote incoming edge:
    PossibleMultipleIncoming = {
        EdgeTags.LinkArgument,
        EdgeTags.LinkRelation,
    }

    tag_rules = [
        TagRule(trigger=(None, EdgeTags.Process), disallowed=(None, EdgeTags.State)),
        TagRule(trigger=(None, EdgeTags.State), disallowed=(None, EdgeTags.Process)),
        TagRule(trigger=(ChildlessIncoming, None), allowed=(None, ChildlessOutgoing)),
        # TagRule(trigger=(LinkerIncoming, None), allowed=(LinkerIncoming, None)),  # passage 106, unit 1.300
    ] + \
        [TagRule(trigger=(t, None), disallowed=(t, None)) for t in UniqueIncoming] + \
        [TagRule(trigger=(None, t), disallowed=(None, t)) for t in UniqueOutgoing]

    def __init__(self, args):
        self.args = args

    def is_top_level(self, tag):
        return tag in self.TopLevel

    def is_possible_multiple_incoming(self, tag):
        return self.args.linkage and tag in self.PossibleMultipleIncoming

    # Require the first action to be shift, i.e., do not allow implicit children to the root
    @property
    def require_first_shift(self):
        return not self.args.implicit
