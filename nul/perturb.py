from foc import *
from ouch import *

from .utils import from_jamos, to_jamos


def rotate(x):
    i = randint(len(x))
    return x[i:] + x[:i]


def insert(x):
    o, i = choice(range(0, len(x)), 2)
    return x[:i] + [x[o]] + x[i:]


def remove(x, limit=3):
    if len(x) < limit:
        return x
    i = randint(len(x))
    return x[:i] + x[i + 1 :]


def swap(x):
    i, j = choice(range(0, len(x)), 2)
    o = x[:]
    o[j], o[i] = o[i], o[j]
    return o


def rotate_word(x):
    return cf_(unwords, rotate, words)(x)


def insert_word(x):
    return cf_(unwords, insert, words)(x)


def remove_word(x):
    return cf_(unwords, remove, words)(x)


def swap_word(x):
    return cf_(unwords, swap, words)(x)


def rotate_jamo(x):
    return cf_(from_jamos, rotate, to_jamos)(x)


def insert_jamo(x):
    return cf_(from_jamos, insert, chars, to_jamos)(x)


def remove_jamo(x):
    return cf_(from_jamos, remove, chars, to_jamos)(x)


def swap_jamo(x):
    return cf_(from_jamos, swap, chars, to_jamos)(x)


def insert_wspace(x):
    x = to_jamos(x)
    i = randint(len(x))
    s = choice([" ", "  ", "\t", "\n"])
    return from_jamos(x[:i] + s + x[i:])


def remove_wspace(x):
    x = words(x)
    i = randint(len(x))
    return unwords(x[:i] + [unchars(x[i : i + 2])] + x[i + 2 :])


def no_wspace(x):
    return unchars(words(x))


def invertcase(x):
    def invert(o):
        return o.lower() if o.isupper() else o.upper()

    return cf_(unchars, map(invert), chars)(x)


def perturb(x):
    return choice(
        [
            rotate_word,
            insert_word,
            remove_word,
            swap_word,
            rotate_jamo,
            insert_jamo,
            remove_jamo,
            swap_word,
            insert_wspace,
            remove_wspace,
            no_wspace,
            invertcase,
        ]
    )(x)


def perturbs(x, rep=3):
    """recursively apply 'perturb' to 'x' the number of 'rep'-times"""
    return take(rep, iterate(perturb, x))[-1]
