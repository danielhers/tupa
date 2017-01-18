import re

from parsing.config import Config, Singleton


class Action(object):
    type_to_id = {}

    RIGHT = 0
    LEFT = 1
    MAX_SWAP = 15  # default maximum size for compound swap

    def __init__(self, action_type, tag=None, orig_edge=None, orig_node=None, oracle=None):
        self.type = action_type  # String
        self.tag = tag  # Usually the tag of the created edge; but if COMPOUND_SWAP, the distance
        self.orig_node = orig_node  # Node created by this action, if any (during training)
        self.orig_edge = orig_edge
        self.edge = None  # Will be set by State when the edge created by this action is known
        self.oracle = oracle
        self.index = None

        self.type_id = Action.type_to_id.get(self.type)  # Allocate ID for fast comparison
        if self.type_id is None:
            self.type_id = len(Action.type_to_id)
            Action.type_to_id[self.type] = self.type_id
        self._id = None

    def is_type(self, *others):
        return self.type_id in (o.type_id for o in others)

    def apply(self):
        if self.oracle is not None:
            self.oracle.remove(self.orig_edge, self.orig_node)

    @staticmethod
    def from_string(s):
        m = re.match("(.*)-(.*)", s)
        if m:  # String contains tag
            action_type, tag = m.groups()
            return Action(action_type, tag)
        return Action(s)

    @classmethod
    def edge_action(cls, direction, remote, tag, *args, **kwargs):
        remote_action, edge_action = (Actions.RightRemote, Actions.RightEdge) \
            if direction == cls.RIGHT else (Actions.LeftRemote, Actions.LeftEdge)
        action = remote_action if remote else edge_action
        return action(tag, *args, **kwargs)

    def __repr__(self):
        return Action.__name__ + "(" + self.type + (", " + self.tag if self.tag else "") + ")"

    def __str__(self):
        return self.type + ("-" + str(self.tag) if self.tag else "")

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __call__(self, *args, **kwargs):
        return Action(self.type, *args, **kwargs)

    @property
    def remote(self):
        return self.is_type(Actions.LeftRemote, Actions.RightRemote)

    @property
    def is_swap(self):
        return self.is_type(Actions.Swap)

    @property
    def id(self):
        self.generate_id()
        return self._id

    def generate_id(self):
        if self._id is None:
            key = (self.type_id, self.tag)
            actions = Actions()
            self._id = actions.ids.get(key)
            if self._id is None:  # New action, add to list
                self._id = len(actions.all)
                actions.all.append(self)
                actions.ids[key] = self._id


class Actions(object, metaclass=Singleton):

    Shift = Action("SHIFT")
    Node = Action("NODE")
    Implicit = Action("IMPLICIT")
    Reduce = Action("REDUCE")
    LeftEdge = Action("LEFT-EDGE")
    RightEdge = Action("RIGHT-EDGE")
    LeftRemote = Action("LEFT-REMOTE")
    RightRemote = Action("RIGHT-REMOTE")
    Swap = Action("SWAP")
    Finish = Action("FINISH")

    def __init__(self):
        self._all = None
        self._ids = None

    def init(self):
        # edge and node action will be created as they are returned by the oracle
        self.all = [Actions.Reduce, Actions.Shift, Actions.Finish] + \
                   (map(Actions.Swap, range(1, Action.MAX_SWAP)) if Config().args.compound_swap
                    else [Actions.Swap] if Config().args.swap else [])

    @property
    def all(self):
        if self._all is None:
            self.init()
        return self._all

    @all.setter
    def all(self, actions):
        self._all = list(actions)
        self._ids = {(action.type_id, action.tag): i for i, action in enumerate(actions)}

    @property
    def ids(self):
        if self._all is None:
            self.init()
        return self._ids


