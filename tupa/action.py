from tupa.config import Config, Singleton


class Action(object):
    type_to_id = {}
    MAX_SWAP = 15  # default maximum size for compound swap

    def __init__(self, action_type, tag=None, label=None, orig_edge=None, orig_node=None, oracle=None):
        self.type = action_type  # String
        self.tag = tag  # Usually the tag of the created edge; but if COMPOUND_SWAP, the distance
        self.label = label  # Label of the created node, if any
        self.orig_node = orig_node  # Node created by this action, if any (during training)
        self.orig_edge = orig_edge  # Edge created by this action, if any (during training)
        self.edge = None  # Will be set by State when the edge created by this action is known
        self.oracle = oracle  # Reference to oracle, to inform it of actually created nodes/edges
        self.index = None  # Index of this action in history

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

    def __repr__(self):
        return Action.__name__ + "(" + ", ".join(filter(None, (self.type, self.tag, self.label))) + ")"

    def __str__(self):
        s = self.type
        if self.tag:
            s += "-%s" % self.tag
            if self.label:
                s += "/%s" % self.label
        return s

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __call__(self, *args, **kwargs):
        return Action(self.type, *args, **kwargs)

    @property
    def remote(self):
        return self.is_type(Actions.RemoteNode, Actions.LeftRemote, Actions.RightRemote)

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
                # noinspection PyTypeChecker
                self._id = len(actions.all)
                actions.all.append(self)
                actions.ids[key] = self._id


class Actions(object, metaclass=Singleton):
    Shift = Action("SHIFT")
    Node = Action("NODE")
    RemoteNode = Action("REMOTE-NODE")
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
