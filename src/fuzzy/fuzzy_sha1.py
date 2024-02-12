# This ancillary file to
# "Using fuzzy bits and neural networks to partially invert few rounds
# of some cryptographic hash functions" paper by S.V. Goncharov
# is placed in public domain.

import numpy as np
from tensorflow.python.keras import backend as K
import random

import bytehash
from fuzzybitsconv import *

from fuzzyops_backend import *
# from fuzzyops_circ_backend import * # must be imported after previous one to replace NOT, AND, OR, XOR, ADD

SHA1_K1 = K.constant(np.array(int2fzb(0x5A827999, 0x20)), dtype='float64')
SHA1_K2 = K.constant(np.array(int2fzb(0x6ED9EBA1, 0x20)), dtype='float64')
SHA1_K3 = K.constant(np.array(int2fzb(0x8F1BBCDC, 0x20)), dtype='float64')
SHA1_K4 = K.constant(np.array(int2fzb(0xCA62C1D6, 0x20)), dtype='float64')

SHA1_H0 = K.constant(np.array(int2fzb(0x67452301, 0x20)), dtype='float64')
SHA1_H1 = K.constant(np.array(int2fzb(0xEFCDAB89, 0x20)), dtype='float64')
SHA1_H2 = K.constant(np.array(int2fzb(0x98BADCFE, 0x20)), dtype='float64')
SHA1_H3 = K.constant(np.array(int2fzb(0x10325476, 0x20)), dtype='float64')
SHA1_H4 = K.constant(np.array(int2fzb(0xC3D2E1F0, 0x20)), dtype='float64')

def add32fuzzy(msg, length=0x40, rounds=1): # test; not used further
    return fzadd(msg[:0x20], msg[0x20:])

def sha1fuzzy(msg, length, rounds=0x50):
    msg = K.concatenate([msg, np.array(([0.0] * 7) + [1.0] +  ([0.0] * ((0x1C0 - length - 8) & 0x1FF)) + lebe(int2fzb(length, 0x40))).astype('float64')])

    h0, h1, h2, h3, h4 = K.identity(SHA1_H0), K.identity(SHA1_H1), K.identity(SHA1_H2), K.identity(SHA1_H3), K.identity(SHA1_H4)

    n = (length + ((0x1C0 - length) & 0x1FF) + 0x40) >> 9

    w = [None for k in range(0x50)]

    for i in range(n):
        chunk = msg[(i<<9):((i+1)<<9)]

        a, b, c, d, e = K.identity(h0), K.identity(h1), K.identity(h2), K.identity(h3), K.identity(h4)

        for j in range(0, min(0x10, rounds)):
            w[j] = lebe4(chunk[(j<<5):((j+1)<<5)])
            f = fzxor(d, fzand(b, fzxor(c, d)))
            a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K1), w[j]), d, c, fzrotl(b, 30), a
        for j in range(0x10, min(0x14, rounds)):
            w[j] = fzrotl(fzxor(fzxor(fzxor(w[j-3], w[j-8]), w[j-0xE]), w[j-0x10]), 1)
            f = fzxor(d, fzand(b, fzxor(c, d)))
            a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K1), w[j]), d, c, fzrotl(b, 30), a
        for j in range(0x14, min(0x28, rounds)):
            w[j] = fzrotl(fzxor(fzxor(fzxor(w[j-3], w[j-8]), w[j-0xE]), w[j-0x10]), 1)
            f = fzxor(fzxor(b, c), d)
            a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K2), w[j]), d, c, fzrotl(b, 30), a
        for j in range(0x28, min(0x3C, rounds)):
            w[j] = fzrotl(fzxor(fzxor(fzxor(w[j-3], w[j-8]), w[j-0xE]), w[j-0x10]), 1)
            f = fzor(fzor(fzand(b, c), fzand(b, d)), fzand(c, d))
            a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K3), w[j]), d, c, fzrotl(b, 30), a
        for j in range(0x3C, min(0x50, rounds)):
            w[j] = fzrotl(fzxor(fzxor(fzxor(w[j-3], w[j-8]), w[j-0xE]), w[j-0x10]), 1)
            f = fzxor(fzxor(b, c), d)
            a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K4), w[j]), d, c, fzrotl(b, 30), a

        h0, h1, h2, h3, h4 = fzadd(h0, a), fzadd(h1, b), fzadd(h2, c), fzadd(h3, d), fzadd(h4, e)

    return K.concatenate([lebe4(h0), lebe4(h1), lebe4(h2), lebe4(h3), lebe4(h4)])


def sha1roundfuzzy(stateAndWord, length=192, rounds=1):
    a, b, c, d, e, word = stateAndWord[:0x20], stateAndWord[0x20:0x40], stateAndWord[0x40:0x60], stateAndWord[0x60:0x80], stateAndWord[0x80:0xA0], stateAndWord[0xA0:]
    f = fzxor(d, fzand(b, fzxor(c, d)))
    a, e, d, c, b = fzadd(fzadd(fzadd(fzadd(fzrotl(a, 5), f), e), SHA1_K1), word), d, c, fzrotl(b, 30), a

    return K.concatenate([a, b, c, d, e])

def test():
    msgstr = "The quick brown fox jumps over the lazy dog"

    msgbits = str2fzb(msgstr)
    print(msgbits)

    # msgbits[0] = 0.1

    msg = np.array(msgbits).astype('float64')

    fuzzhash1 = K.eval(sha1fuzzy(K.variable(msg, dtype='float64'), length=len(msg), rounds=2))

    print("FuzzHashSHA1-160-Hex:", fzb2hex(fuzzhash1))
    print("ByteHashSHA1-160-Hex:", bytehash.bytes2hex(bytehash.sha1bytes(bytes(msgstr, encoding='ASCII'), rounds=2)))

# test()

def msg_to_hash(msgbits, rounds):
    msg = np.array(msgbits).astype('float64')

    fuzzhash1 = K.eval(sha1fuzzy(K.variable(msg, dtype='float64'), length=len(msg), rounds=rounds))
    return fzb2hex(fuzzhash1)

def gen_data(input_length, size, file):
    print("updated")
    with open(file, 'w') as pw:
        for i in range(input_length):
            bits = np.random.uniform(size=size)
            pw.write(msg_to_hash(bits, 2) + "," + msg_to_hash(bits, 3) + "\n")
    return

if __name__ == '__main__':
    gen_data(pow(10, 3), 432, "./fuzzy-3.csv")