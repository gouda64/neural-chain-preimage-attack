# This ancillary file to
# "Using fuzzy bits and neural networks to partially invert few rounds
# of some cryptographic hash functions" paper by S.V. Goncharov
# is placed in public domain.

def int2fzb(x, n):
    b = [0.0] * n
    for i in range(n):
        b[i] = x & 1
        x >>= 1
    return b

def str2fzb(s):
    l = len(s)
    b = []
    for i in range(l):
        b += int2fzb(ord(s[i]), 8)
    return b

def fzb2hex(bits, uc=False):
    bits = list(bits) # copy
    for i in range(len(bits)):
        if (bits[i] < 0.5) and (bits[i] > -0.5):
            bits[i] = 0
        else:
            bits[i] = 1
    digsym = "0123456789abcdef"
    if uc == True:
        digsym = "0123456789ABCDEF"
    n = len(bits) >> 3
    s = ""
    for i in range(n):
        hex = bits[((i<<3)+4):((i+1)<<3)]
        s += digsym[hex[0] + (hex[1]<<1) + (hex[2]<<2) + (hex[3]<<3)]
        hex = bits[(i<<3):((i<<3)+4)]
        s += digsym[hex[0] + (hex[1]<<1) + (hex[2]<<2) + (hex[3]<<3)]
    return s

def lebe(bits):
    bits = list(bits) # copy
    n = len(bits) >> 3
    for i in range(n >> 1):
        bits[(i<<3):((i+1)<<3)], bits[((n-1-i)<<3):((n-i)<<3)] = bits[((n-1-i)<<3):((n-i)<<3)], bits[(i<<3):((i+1)<<3)]
    return bits