import faulthandler
import sys

import os
import signal


def tracefunc(frame, event, arg):
    if event.endswith("call") and arg:
        if (getattr(arg, "__module__", None) or getattr(arg.__self__.__class__, "__module__")) == "_dynet":
            print(">", os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, arg.__qualname__, "(",
                  ", ".join("%s=%r" % (v, frame.f_locals[v])
                            for v in frame.f_code.co_varnames[:frame.f_code.co_argcount] if v != "self"), ")")
            # print(arg.__qualname__)
    return tracefunc


def set_tracefunc():
    sys.setprofile(tracefunc)


def print_traceback(*args, **kwargs):
    del args, kwargs
    faulthandler.dump_traceback()


def set_traceback_listener(sig=None):
    try:
        signal.signal(signal.SIGUSR1 if sig is None else sig, print_traceback)
    except AttributeError:
        pass
