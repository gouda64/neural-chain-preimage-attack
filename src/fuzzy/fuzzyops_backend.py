# This ancillary file to
# "Using fuzzy bits and neural networks to partially invert few rounds
# of some cryptographic hash functions" paper by S.V. Goncharov
# is placed in public domain.

import numpy as np
from tensorflow.python.keras import backend as K



def lebe4(w):
    return K.concatenate([w[0x18:0x20], w[0x10:0x18], w[8:0x10], w[0:8]])


def fznot(w):
    return 1.0 - w


def fzor(w1, w2):
    # return K.maximum(w1, w2)
    return w1 + w2 - w1 * w2 # it's associative...


def fzand(w1, w2):
    # return K.minimum(w1, w2)
    return w1 * w2


def fzxor(w1, w2):
    # return K.maximum(K.minimum(w1, 1.0 - w2), K.minimum(1.0 - w1, w2))
    # return (w1 - w2) * (w1 - w2)

    # w1n, w2n = 1.0 - w1, 1.0 - w2
    # m1, m2 = w1 * w2n, w1n * w2
    # return m1 + m2 - m1 * m2 # ((fznot A) fzand B) fzor (A fzand (fznot B))

    return w1 * (1.0 - w2) + (1.0 - w1) * w2 # associative...


def fzadd(w1, w2, l=0x20):
    # r = K.variable(np.array([]), dtype='float64')
    # r = K.variable(np.array([0.0] * l), dtype='float64')
    c = K.variable(np.array([0.0]), dtype='float64')

    x = w1 * (1.0 - w2) + (1.0 - w1) * w2
    s1 = w1 + w2 # vectorization is faster...

    ss = []

    for i in range(l):
        # b1, b2 = w1[i], w2[i]

        # s = fzxor(fzxor(b1, b2), c)
        # c = fzor(fzor(fzand(fzand(1.0 - b1, b2), c), fzand(fzand(b1, 1.0 - b2), c)), fzor(fzand(fzand(b1, b2), 1.0 - c), fzand(fzand(b1, b2), c)))

        # s1 = fzxor(b1, b2)
        # c1 = fzand(b1, b2)
        # s = fzxor(s1, c)
        # c2 = fzand(s1, c)
        # c = fzxor(c1, c2) # full adder

        # s = b1 * (1.0 - b2) + (1.0 - b1) * b2
        # s = s * (1.0 - c) + (1.0 - s) * c
        # c = 0.5 * (b1 + b2 + c - s) # "interpolation" inside (b1;b2;c)-cube, based on values in 8 vertices
        xi = x[i]
        s = xi * (1.0 - c) + (1.0 - xi) * c
        ss.append(s)
        c = 0.5 * (s1[i] + c - s) # same, but faster a little

    # t = b1 + b2 + c
    # s = 0.5 * (1.0 - K.cos(np.pi * t))
    # c = 0.5 * (t - s)

    # t = b1 + b2 + c
    # s = K.minimum(t, 4.0 - t)
    # s = K.minimum(s, 2.0 - s)
    # c = 0.5 * (t - s)

    # t = b1 + b2 + c
    # c = K.minimum(K.maximum(t - 1.0, 0.0), 1.0)
    # s = t - 2.0 * c # same as previous...

    # r = K.concatenate([r, s])
    # r = r[i].assign(s[0]) # doesn't work...

    return K.concatenate(ss)


def fzrotl(w, n=1, l=0x20):
    return K.concatenate([w[(l-n):], w[:(l-n)]])


def fzrotr(w, n=1, l=0x20):
    return K.concatenate([w[n:], w[:n]])


def fzshl(w, n=1, l=0x20):
    return K.concatenate([K.zeros(n), w[:(l-n)]])


def fzshr(w, n=1, l=0x20):
    return K.concatenate([w[n:], K.zeros(n)])