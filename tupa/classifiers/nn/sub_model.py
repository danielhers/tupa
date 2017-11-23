from collections import OrderedDict


class SubModel(object):
    def __init__(self, params=None, save_path=()):
        self.params = OrderedDict() if params is None else params  # string (param identifier) -> parameter
        self.save_path = save_path

    def save_sub_model(self, d, *args):
        self.get_sub_dict(d).update(args + (("param_keys", list(self.params.keys())),))
        return list(self.params.values())

    def load_sub_model(self, d, *args):
        d = self.get_sub_dict(d)
        param_keys = d.get("param_keys", ())
        assert len(param_keys) <= len(args), "%s loaded values: expected %d, got %d" % ("/".join(self.save_path),
                                                                                        len(param_keys), len(args))
        self.params.clear()
        self.params.update(zip(param_keys, args))
        return d

    def get_sub_dict(self, d):
        for element in self.save_path:
            d = d.setdefault(element, OrderedDict())
        return d

    def __str__(self):
        return "/".join(self.save_path) + (": " if self.save_path else "") + ", ".join(self.params.keys())
