from .config import Config, COMPOUND


class Action(object):
    type_to_id = {}

    def __init__(self, action_type, tag=None, has_label=False, orig_edge=None, orig_node=None, oracle=None, id_=None):
        self.type = action_type  # String
        self.tag = tag  # Usually the tag of the created edge; but if COMPOUND_SWAP, the distance
        self.has_label = has_label  # Whether this action type requires a label or not
        self.orig_node = orig_node  # Node created by this action, if any (during training)
        self.orig_edge = orig_edge  # Edge created by this action, if any (during training)
        self.node = None  # Will be set by State when the node created by this action is known
        self.edge = None  # Will be set by State when the edge created by this action is known
        self.oracle = oracle  # Reference to oracle, to inform it of actually created nodes/edges
        self.index = None  # Index of this action in history

        self.type_id = Action.type_to_id.get(self.type)  # Allocate ID for fast comparison
        if self.type_id is None:
            self.type_id = len(Action.type_to_id)
            Action.type_to_id[self.type] = self.type_id
        self.id = id_

    def is_type(self, *others):
        return self.type_id in (o.type_id for o in others)

    def apply(self):
        if self.oracle is not None:
            self.oracle.remove(self.orig_edge, self.orig_node)

    def __repr__(self):
        return Action.__name__ + "(" + ", ".join(filter(None, (self.type, self.tag))) + ")"

    def __str__(self):
        s = self.type
        if self.tag:
            s += "-%s" % self.tag
        return s

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __call__(self, *args, **kwargs):
        return Action(self.type, *args, **kwargs, has_label=self.has_label)

    @property
    def remote(self):
        return self.is_type(Actions.RemoteNode, Actions.LeftRemote, Actions.RightRemote)

    @property
    def is_swap(self):
        return self.is_type(Actions.Swap)


class Actions(object):
    Shift = Action("SHIFT")
    Node = Action("NODE", has_label=True)
    RemoteNode = Action("REMOTE-NODE", has_label=True)
    Implicit = Action("IMPLICIT", has_label=True)
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
                   (list(map(Actions.Swap, range(1, Config().args.max_swap))) if Config().args.swap == COMPOUND
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
        for action in self._all:
            self.generate_id(action)

    @property
    def ids(self):
        if self._all is None:
            self.init()
        return self._ids

    def generate_id(self, action, create=True):
        if action.id is None:
            key = (action.type_id, action.tag)
            action.id = self.ids.get(key)
            if create and action.id is None:  # New action, add to list
                # noinspection PyTypeChecker
                action.id = len(self.all)
                self.all.append(action(tag=action.tag, id_=action.id))
                self.ids[key] = action.id
