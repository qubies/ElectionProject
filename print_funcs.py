import sys
from functools import wraps


def print_banner(s, width=80, banner_token="-"):
    """
    pretty banner for cli tasks
    """
    if len(s) > width:
        return s
    rem = width - len(s)
    rhs = rem // 2
    lhs = rem - rhs
    if rhs > 0:
        rhs_pad = " " + (rhs - 1) * banner_token
    else:
        rhs_pad = ""
    lhs_pad = (lhs - 1) * banner_token + " "
    print(lhs_pad + s + rhs_pad, file=sys.stderr)


class PrintContext:
    def __init__(self, s, width=80, banner_token="-"):
        self.width = width
        self.banner_token = banner_token
        self.s = s

    def __enter__(self):
        print_banner(self.s, self.width, self.banner_token)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print_banner("Done " + self.s, self.width, self.banner_token)


def print_banner_completion_wrapper(s, width=80, banner_token="-"):
    """
    prints a banner and the beginning and end of some func
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print_banner(s, width, banner_token)
            result = func(*args, **kwargs)
            print_banner("Done " + s, width, banner_token)
            print(file=sys.stderr)
            return result

        return wrapper

    return wrap
