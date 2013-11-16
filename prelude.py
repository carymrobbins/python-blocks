"""
For convenience, all names are exported from itertools.  This is so you don't
have to remember whether or not a particular function needs to be imported
from itertools since this library is very similar.
"""

from functools import partial
from itertools import (
    count, cycle, repeat, chain, compress, dropwhile, groupby, ifilter,
    ifilterfalse, islice, imap, starmap, tee, takewhile, izip, izip_longest,
    product, permutations, combinations, combinations_with_replacement)
import operator


def curry(n):
    """ Pass an integer for number of args to curry. """
    def do(f):
        def curried_function(*args):
            if len(args) >= n:
                return f(*args)
            else:
                return partial(curried_function, *args)
        return curried_function
    return do


curry1 = curry(1)
curry2 = curry(2)
curry3 = curry(3)


def uncurry(f, args=None, kwargs=None):
    """
    Takes arguments as iterables and executes f with them.  Virtually identical
    to the deprecated function apply().
    * NOTE - uncurry is not the inverse of curry! *
    """
    return f(*args, **kwargs)


@curry2
def uncurry2(*args):
    """ Executes uncurry once it has received at least two arguments. """
    return uncurry(*args)


@curry3
def uncurry3(*args):
    """ Executes uncurry once it has received at least three arguments."""
    return uncurry(*args)


def identity(x):
    return x


def const(x):
    def do(*args, **kwargs):
        return x
    return do


def compose2(f, g):
    def do(*args, **kwargs):
        return f(g(*args, **kwargs))
    return do


def compose(*functions):
    if functions:
        return reduce(compose2, functions)
    else:
        return identity


def foldl(sequence, initial):
    def do(f):
        return reduce(f, sequence, initial)
    return do


def foldr(sequence, initial):
    return foldl(reversed(sequence), initial)


def foldl1(sequence):
    def do(f):
        return reduce(f, sequence)
    return do


def foldr1(sequence):
    return foldl1(reversed(sequence))


def scanl(sequence, initial):
    def do(f):
        acc = initial
        yield acc
        for item in sequence:
            acc = f(acc, item)
            yield acc
    return do


def scanr(sequence, initial):
    return scanl(reversed(sequence), initial)


def scanl1(sequence):
    def do(f):
        it = iter(sequence)
        acc = next(it)
        yield acc
        for item in it:
            acc = f(acc, item)
            yield acc
    return do


def scanr1(sequence):
    return scanl1(reversed(sequence))


def iterate(*args, **kwargs):
    def do(f):
        while True:
            yield f(*args, **kwargs)
    return do


@curry3
def until(value, predicate, function):
    while not predicate(value):
        value = function(value)
    return value


@curry2
def take(n, sequence):
    return islice(sequence, n)


@curry2
def drop(n, sequence):
    return islice(sequence, n, None)


@curry2
def replicate(n, value):
    i = 0
    while i < n:
        i += 1
        yield value


@curry2
def split_at(n, sequence):
    it1, it2 = tee(sequence)
    return islice(it1, n), islice(it2, n, None)


def maps(*sequences):
    def do(f):
        return imap(f, *sequences)
    return do


def each(*sequences):
    def do(f):
        for args in izip(*sequences):
            f(*args)
    return do


def filters(sequence):
    def do(f):
        return ifilter(f, sequence)
    return do


concat = chain.from_iterable


def concat_maps(*sequences):
    def do(f):
        return concat(imap(f, *sequences))
    return do


def functioncaller(*args, **kwargs):
    def do(f):
        return f(*args, **kwargs)
    return do


@curry2
def index(n, sequence):
    return next(islice(sequence, n, n + 1))


def fst((x, _)):
    return x


def snd((_, y)):
    return y


def head(sequence):
    return next(iter(sequence))


def tail(sequence):
    it = iter(sequence)
    next(it)
    return it


def last(sequence):
    for x in sequence:
        pass
    try:
        return x
    except UnboundLocalError:
        raise ValueError('Cannot get last of empty sequence.')


def empty(sequence):
    try:
        next(iter(sequence))
    except StopIteration:
        return True
    return False


def length(sequence):
    i = -1
    for i, _ in enumerate(sequence):
        pass
    return i + 1


def init(sequence):
    it = iter(sequence)
    try:
        x = next(it)
    except StopIteration:
        raise ValueError('Cannot get init of empty sequence.')
    while True:
        try:
            y = next(it)
        except StopIteration:
            break
        yield x
        try:
            x = next(it)
        except StopIteration:
            break
        yield y


def flip(f):
    @curry2
    def do(y, x):
        return f(x, y)
    return do


def even(x):
    return x % 2 == 0


def odd(x):
    return x % 2 == 1


class Function(object):
    def __init__(self, function=None):
        self.function = function or identity

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __mul__(self, other):
        """Simulates composition via star operator.
        (f * g)(x) == f(g(x))
        """
        return Function(compose2(self.function, other))


# Defining helper objects for the identity and const function, respectively.
# These make it simple to use the Function class features, e.g.
# >>> f = I * op('+', 2) * op('*', 3)
# >>> f(4) == (2 + (3 * (4))) == 14
I = Function()
K = Function(compose2(Function, const))


def op(symbol, *args, **kwargs):
    f = _curried_operators.get(symbol)
    if f is not None:
        return f(*args, **kwargs)
    else:
        f = _basic_operators.get(symbol)
        if f is not None:
            return f
        else:
            raise NotImplementedError(
                'Operator "{}" not implemented'.format(symbol))


def flip_op(symbol, *args, **kwargs):
    f = flip(_curried_operators.get(symbol))
    if f is not None:
        return f(*args, **kwargs)
    else:
        raise NotImplementedError(
            'Operator "{}" not implemented'.format(symbol))


# Binary functions as operators that require currying.
_curried_operators = {k: curry2(v) for k, v in {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.div,
    '//': operator.floordiv,
    '**': operator.pow,
    '%': operator.mod,
    '>>': operator.rshift,
    '<<': operator.lshift,
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '&': operator.and_,
    '|': operator.or_,
    '^': operator.xor,
    'and': lambda x, y: x and y,
    'or': lambda x, y: x or y,
    'is': operator.is_,
    'is not': operator.is_not,
    'is_not': operator.is_not,
    'in': operator.contains,
    'isinstance': isinstance,
    'issubclass': issubclass,
}.iteritems()}

# Binary functions as operators that are naturally curried.
_curried_operators.update({
    '.': operator.attrgetter,
    '.()': operator.methodcaller,
    '()': functioncaller,
    '[]': curry1(operator.itemgetter),
})

# Unary functions as operators.
_basic_operators = {
    'not': lambda x: not x,
    '~': operator.not_,
    'neg': operator.neg,
}
