from collections import OrderedDict


class SubModel:
    def __init__(self, params=None, save_path=(), shared=False, copy_shared=False):
        self.params = OrderedDict() if params is None else params  # string (param identifier) -> parameter
        self.save_path = save_path
        self.shared = shared
        self.copy_shared = copy_shared

    def save_sub_model(self, d, *args):
        self.get_sub_dict(d).update(args + (("param_keys", list(self.params.keys())),))
        return list(self.params.values())

    def load_sub_model(self, d, *args, load_path=None):
        d = self.get_sub_dict(d, load_path=load_path)
        param_keys = d.get("param_keys", ())
        assert len(param_keys) <= len(args), "%s loaded values: expected %d, got %d" % ("/".join(self.save_path),
                                                                                        len(param_keys), len(args))
        self.params.clear()
        self.params.update(zip(param_keys, args))
        return d

    def get_sub_dict(self, d, load_path=None):
        for element in load_path or self.save_path:
            d = d.setdefault(element, OrderedDict())
        return d

    def params_str(self):
        return "/".join(self.save_path) + (": " if self.save_path else "") + ", ".join(self.params.keys())

    def invalidate_caches(self):
        for model in self.sub_models():
            model.invalidate_caches()

    def sub_models(self):
        return ()
