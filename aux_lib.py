"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

# Factorize an iterable into pairs of consequent items
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# Convert a non-negative decimal integer into a given base (the result is an array with numbers)
def int2base(x, base):
    assert (isinstance(x, int) and x >= 0), "This function works only for non-negative integers"

    if x == 0:
        return [0]

    digits = []

    while int(x):
        digits.append(int(x % base))
        x /= base

    digits.reverse()

    return digits

# A time measuring class
class Timing:
    m_start = None

    def __init__(self):
        # Start by default, another call to start will REstart
        self.start()

    def start(self):
        self.m_start = self.get_time()

    # Gets a string of the elapsed time since start() was called
    def get_elapsed_time(self, separator = ":"):
        if self.m_start is None:
            raise Exception("Timing: Can't get elapsed time because the start method was never called")

        elapsed = self.get_time() - self.m_start

        return self.secondsToString(elapsed, separator)

    def get_elapsed_secs(self):
        if self.m_start is None:
            raise Exception("Timing: Can't get elapsed time because the start method was never called")

        elapsed = self.get_time() - self.m_start

        return elapsed

    @staticmethod
    def secondsToString(secs, separator=":"):
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)

        elapsed_components = []
        if secs < 1:
            return "{}ms".format(round(secs * 1000))

        if d > 0:
            elapsed_components.append("{0:.0f}d".format(d))

        if h > 0:
            elapsed_components.append("{0:.0f}h".format(h))

        if m > 0:
            elapsed_components.append("{0:.0f}m".format(m))

        if s > 0:
            elapsed_components.append("{0:.0f}s".format(round(s)))

        return separator.join(elapsed_components)

    @staticmethod
    def get_time():
        from timeit import default_timer as timer
        return timer()
