import argparse


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance

    def reload(cls):
        cls.instance = None


class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values is None:
            values = "1"
        try:
            values = int(values)
        except ValueError:
            values = values.count("v") + 1
        setattr(args, self.dest, values)


def add_verbose_argument(argparser, **kwargs):
    return argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, **kwargs)


def get_group_arg_names(group):
    return [a.dest for a in group._group_actions]


def add_boolean_option(argparser, name, description, default=False, short=None, short_no=None):
    group = argparser.add_mutually_exclusive_group()
    options = [] if short is None else ["-" + short]
    options.append("--" + name)
    group.add_argument(*options, action="store_true", default=default, help="include " + description)
    no_options = [] if short_no is None else ["-" + short_no]
    no_options.append("--no-" + name)
    group.add_argument(*no_options, action="store_false", dest=name.replace("-", "_"), default=default,
                       help="exclude " + description)
    return group
