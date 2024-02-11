# This ancillary file to
# "Using fuzzy bits and neural networks to partially invert few rounds
# of some cryptographic hash functions" paper by S.V. Goncharov
# is placed in public domain.

import hashlib



def int2bytes(x, n=4):
    b = [0] * n
    for i in range(n):
        b[i] = x & 0xFF
        x >>= 8
    return b

def bytes2int(bytes):
    s = 0
    l = len(bytes)
    for i in range(l):
        s <<= 8
        s += bytes[l-1-i]
    return s

def bytes2hex(bytes, uc=False):
    digsym = "0123456789abcdef"
    if uc == True:
        digsym = "0123456789ABCDEF"
    l = len(bytes)
    s = ""
    for i in range(l):
        s += digsym[(bytes[i] >> 4) & 0xF]
        s += digsym[bytes[i] & 0xF]
    return s


def lebe(bytes):
    bytes = list(bytes) # copy
    l = len(bytes)
    for i in range(l >> 1):
        bytes[i], bytes[l-1-i] = bytes[l-1-i], bytes[i]
    return bytes

def rotl(d, n):
    return ((d << n) & 0xFFFFFFFF) | (d >> (0x20 - n))

def rotr(d, n):
    return (d >> n) | ((d << (0x20 - n)) & 0xFFFFFFFF)


def md5bytes(bytes, rounds=0x40):
    s = ((7, 12, 17, 22) * 4) + ((5, 9, 14, 20) * 4) + ((4, 11, 16, 23) * 4) + ((6, 10, 15, 21) * 4)
    ks = (0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
          0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
          0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x2441453,  0xd8a1e681, 0xe7d3fbc8,
          0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
          0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
          0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x4881d05,  0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
          0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
          0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391)

    a0, b0, c0, d0 = 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476

    msglen = len(bytes) << 3 # length in bits
    bytes = list(bytes) # copy

    bytes += [0x80]
    bytes += [0] * ((0x38 - len(bytes)) & 0x3F)
    bytes += int2bytes(msglen, 8) # now len(bytes) mod 64 = 0

    n = len(bytes) >> 6

    w = [0] * 0x10

    for i in range(n):
        a, b, c, d = a0, b0, c0, d0

        chunk = bytes[(i<<6):((i+1)<<6)]

        for j in range(0, min(0x10, rounds)):
            w[j] = bytes2int(chunk[(j<<2):((j+1)<<2)])
            f = d ^ (b & (c ^ d))
            g = j
            a, d, c, b = d, c, b, (b + rotl((f + a + ks[j] + w[g]) & 0xFFFFFFFF, s[j])) & 0xFFFFFFFF
        for j in range(0x10, min(0x20, rounds)):
            f = c ^ (d & (b ^ c))
            g = (5*j + 1) & 0xF
            a, d, c, b = d, c, b, (b + rotl((f + a + ks[j] + w[g]) & 0xFFFFFFFF, s[j])) & 0xFFFFFFFF
        for j in range(0x20, min(0x30, rounds)):
            f = b ^ c ^ d
            g = (3*j + 5) & 0xF
            a, d, c, b = d, c, b, (b + rotl((f + a + ks[j] + w[g]) & 0xFFFFFFFF, s[j])) & 0xFFFFFFFF
        for j in range(0x30, min(0x40, rounds)):
            f = c ^ (b | (~d))
            g = (7*j) & 0xF
            a, d, c, b = d, c, b, (b + rotl((f + a + ks[j] + w[g]) & 0xFFFFFFFF, s[j])) & 0xFFFFFFFF

        a0, b0, c0, d0 = (a0 + a) & 0xFFFFFFFF, (b0 + b) & 0xFFFFFFFF, (c0 + c) & 0xFFFFFFFF, (d0 + d) & 0xFFFFFFFF

    return int2bytes(a0) + int2bytes(b0) + int2bytes(c0) + int2bytes(d0)


def sha1bytes(bytes, rounds=0x50):
    k1 = 0x5A827999
    k2 = 0x6ED9EBA1
    k3 = 0x8F1BBCDC
    k4 = 0xCA62C1D6

    h0 = 0x67452301
    h1 = 0xEFCDAB89
    h2 = 0x98BADCFE
    h3 = 0x10325476
    h4 = 0xC3D2E1F0

    msglen = len(bytes) << 3 # length in bits
    bytes = list(bytes) # copy

    bytes += [0x80]
    bytes += [0] * ((0x38 - len(bytes)) & 0x3F)
    bytes += lebe(int2bytes(msglen, 8)) # now len(bytes) mod 64 = 0

    n = len(bytes) >> 6

    w = [0] * 0x50

    for i in range(n):
        a, b, c, d, e = h0, h1, h2, h3, h4

        chunk = bytes[(i<<6):((i+1)<<6)]

        for j in range(0, min(0x10, rounds)):
            w[j] = bytes2int(lebe(chunk[(j<<2):((j+1)<<2)]))
            f = d ^ (b & (c ^ d))
            a, e, d, c, b  = (rotl(a, 5) + f + e + k1 + w[j]) & 0xFFFFFFFF, d, c, rotl(b, 0x1E), a
        for j in range(0x10, min(0x14, rounds)):
            w[j] = rotl(w[j-3] ^ w[j-8] ^ w[j-0xE] ^ w[j-0x10], 1)
            f = d ^ (b & (c ^ d))
            a, e, d, c, b  = (rotl(a, 5) + f + e + k1 + w[j]) & 0xFFFFFFFF, d, c, rotl(b, 0x1E), a
        for j in range(0x14, min(0x28, rounds)):
            w[j] = rotl(w[j-3] ^ w[j-8] ^ w[j-0xE] ^ w[j-0x10], 1)
            f = b ^ c ^ d
            a, e, d, c, b  = (rotl(a, 5) + f + e + k2 + w[j]) & 0xFFFFFFFF, d, c, rotl(b, 0x1E), a
        for j in range(0x28, min(0x3C, rounds)):
            w[j] = rotl(w[j-3] ^ w[j-8] ^ w[j-0xE] ^ w[j-0x10], 1)
            f = (b & c) | (b & d) | (c & d)
            a, e, d, c, b  = (rotl(a, 5) + f + e + k3 + w[j]) & 0xFFFFFFFF, d, c, rotl(b, 0x1E), a
        for j in range(0x3C, min(0x50, rounds)):
            w[j] = rotl(w[j-3] ^ w[j-8] ^ w[j-0xE] ^ w[j-0x10], 1)
            f = b ^ c ^ d
            a, e, d, c, b  = (rotl(a, 5) + f + e + k4 + w[j]) & 0xFFFFFFFF, d, c, rotl(b, 0x1E), a

        h0, h1, h2, h3, h4 = (h0 + a) & 0xFFFFFFFF, (h1 + b) & 0xFFFFFFFF, (h2 + c) & 0xFFFFFFFF, (h3 + d) & 0xFFFFFFFF, (h4 + e) & 0xFFFFFFFF

    return lebe(int2bytes(h0)) + lebe(int2bytes(h1)) + lebe(int2bytes(h2)) + lebe(int2bytes(h3)) + lebe(int2bytes(h4)) # concatenation


def sha256bytes(bytes, rounds=0x40):
    ks = (0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
          0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
          0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
          0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
          0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
          0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
          0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
          0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2)

    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19

    msglen = len(bytes) << 3 # length in bits
    bytes = list(bytes) # copy

    bytes += [0x80]
    bytes += [0] * ((0x38 - len(bytes)) & 0x3F)
    bytes += lebe(int2bytes(msglen, 8)) # now len(bytes) mod 64 = 0

    n = len(bytes) >> 6

    w = [0] * 0x40

    for i in range(n):
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7

        chunk = bytes[(i<<6):((i+1)<<6)]

        for j in range(min(0x40, rounds)):
            if j >= 0x10:
                s = w[j-15]
                s0 = rotr(s, 7) ^ rotr(s, 18) ^ (s >> 3)
                s = w[j-2]
                s1 = rotr(s, 17) ^ rotr(s, 19) ^ (s >> 10)
                w[j] = (w[j-16] + s0 + w[j-7] + s1) & 0xFFFFFFFF
            else:
                w[j] = bytes2int(lebe(chunk[(j<<2):((j+1)<<2)]))

            s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h + s1 + ch + ks[j] + w[j]) & 0xFFFFFFFF
            s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (s0 + maj) & 0xFFFFFFFF

            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF

        h0, h1, h2, h3, h4, h5, h6, h7 = (h0 + a) & 0xFFFFFFFF, (h1 + b) & 0xFFFFFFFF, (h2 + c) & 0xFFFFFFFF, (h3 + d) & 0xFFFFFFFF, (h4 + e) & 0xFFFFFFFF, (h5 + f) & 0xFFFFFFFF, (h6 + g) & 0xFFFFFFFF, (h7 + h) & 0xFFFFFFFF

    return lebe(int2bytes(h0)) + lebe(int2bytes(h1)) + lebe(int2bytes(h2)) + lebe(int2bytes(h3)) + lebe(int2bytes(h4)) + lebe(int2bytes(h5)) + lebe(int2bytes(h6)) + lebe(int2bytes(h7)) # concatenation


# ----------------------------- KECCAK -----------------------------

# Basically it's Gilles Van Assche's implementation
# from https://github.com/gvanas/KeccakCodePackage/blob/master/Standalone/CompactFIPS202/Python/CompactFIPS202.py

def rotl64(a, n, l=0x40):
    return ((a >> (l - n)) + (a << n)) & ((1<<l) - 1)

def KeccakPermOnLanes(lanes, laneByteWidth, rounds):
    laneWidth = laneByteWidth << 3
    laneWidthMask = laneWidth - 1

    R = 1
    for round in range(rounds):
        # Î¸
        C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
        D = [C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1, laneWidth) for x in range(5)]
        lanes = [[lanes[x][y]^D[x] for y in range(5)] for x in range(5)]

        # Ï and Ï€
        (x, y) = (1, 0)
        current = lanes[x][y]
        tsum = 0
        for t in range(1, 25):
            tsum += t
            (x, y) = (y, (2*x+3*y)%5)
            (current, lanes[x][y]) = (lanes[x][y], rotl64(current, tsum & laneWidthMask, laneWidth))

        # Ï‡
        for y in range(5):
            T = [lanes[x][y] for x in range(5)]
            for x in range(5):
                lanes[x][y] = T[x] ^((~T[(x+1)%5]) & T[(x+2)%5])

        # Î¹
        for j in range(7):
            R = ((R << 1) ^ ((R >> 7)*0x71)) & 0xFF
            if (R & 2):
                lanes[0][0] = lanes[0][0] ^ (1 << ((1<<j)-1))

    return lanes


def KeccakPerm(state, laneByteWidth, rounds):
    lanes = [[0 for y in range(5)] for x in range(5)]
    offset = 0
    for y in range(5):
        for x in range(5):
            b = state[offset:(offset+laneByteWidth)]
            lanes[x][y] = sum((b[i] << (i<<3)) for i in range(laneByteWidth))
            offset += laneByteWidth

    lanes = KeccakPermOnLanes(lanes, laneByteWidth, rounds)

    state = bytearray(25 * laneByteWidth)
    offset = 0
    for y in range(5):
        for x in range(5):
            a = lanes[x][y]
            state[offset:(offset+laneByteWidth)] = list((a >> (i<<3)) & 0xFF for i in range(laneByteWidth))
            offset += laneByteWidth

    return state


def Keccak(rate, capacity, inputBytes, outputByteLen, laneByteWidth, rounds):
    capacity = (outputByteLen << 3) << 1
    rate = 25 * (laneByteWidth << 3) - capacity

    delimitedSuffix = 0x06

    outputBytes = bytearray()
    state = bytearray([0 for i in range(25 * laneByteWidth)])
    rateInBytes = rate >> 3
    blockSize = 0
    if (((rate + capacity) != (25 * (laneByteWidth << 3))) or ((rate & 7) != 0)):
        return
    inputOffset = 0

    # === Absorb all the input blocks ===

    while(inputOffset < len(inputBytes)):
        blockSize = min(len(inputBytes) - inputOffset, rateInBytes)
        for i in range(blockSize):
            state[i] = state[i] ^ inputBytes[i+inputOffset]
        inputOffset += blockSize
        if (blockSize == rateInBytes):
            state = KeccakPerm(state, laneByteWidth, rounds)
            blockSize = 0

    # === Do the padding and switch to the squeezing phase ===

    state[blockSize] = state[blockSize] ^ delimitedSuffix
    if (((delimitedSuffix & 0x80) != 0) and (blockSize == (rateInBytes-1))):
        state = KeccakPerm(state, laneByteWidth, rounds)
    state[rateInBytes-1] = state[rateInBytes-1] ^ 0x80
    state = KeccakPerm(state, laneByteWidth, rounds)

    # === Squeeze out all the output blocks ===

    while(outputByteLen > 0):
        blockSize = min(outputByteLen, rateInBytes)
        outputBytes = outputBytes + state[0:blockSize]
        outputByteLen -= blockSize
        if (outputByteLen > 0):
            state = KeccakPerm(state, laneByteWidth, rounds)

    return outputBytes


def keccak1600_256bytes(inputBytes, rounds=24):
    return Keccak(1088, 512, inputBytes, 256 >> 3, 8, rounds)


def keccak1600_512bytes(inputBytes, rounds=24):
    return Keccak(576, 1024, inputBytes, 512 >> 3, 8, rounds)


def keccak1600_80bytes(inputBytes, rounds=24):
    return Keccak(1440, 160, inputBytes, 80 >> 3, 8, rounds)


def keccak200_80bytes(inputBytes, rounds=18):
    return Keccak(40, 160, inputBytes, 80 >> 3, 1, rounds)

# ----------------------------------------------------------------


def test():
    testr1 = "The quick brown fox jumps over the lazy dog"
    print("Message: '" + testr1 + "'")


    hash = md5bytes(bytes(testr1, encoding='ASCII'), rounds=0x40)
    print("ByteHashMD5-Hex: " + bytes2hex(hash))

    truehash = hashlib.md5()
    truehash.update(bytes(testr1, encoding='ASCII'))
    print("TrueHashMD5-Hex: " + truehash.hexdigest())


    hash = sha1bytes(bytes(testr1, encoding='ASCII'), rounds=0x50)
    print("ByteHashSHA1-Hex: " + bytes2hex(hash))

    truehash = hashlib.sha1()
    truehash.update(bytes(testr1, encoding='ASCII'))
    print("TrueHashSHA1-Hex: " + truehash.hexdigest())


    hash = sha256bytes(bytes(testr1, encoding='ASCII'), rounds=0x40)
    print("ByteHashSHA2-256-Hex: " + bytes2hex(hash))

    truehash = hashlib.sha256()
    truehash.update(bytes(testr1, encoding='ASCII'))
    print("TrueHashSHA2-256-Hex: " + truehash.hexdigest())


    hash = keccak1600_256bytes(bytes(testr1, encoding='ASCII'), rounds=24)
    print("ByteHashSHA3-256-Hex: " + bytes2hex(hash))

    print("TrueHashSHA3-256-Hex: 69070dda01975c8c120c3aada1b282394e7f032fa9cf32f4cb2259a0897dfc04")


    hash = keccak1600_512bytes(bytes(testr1, encoding='ASCII'), rounds=24)
    print("ByteHashSHA3-512-Hex: " + bytes2hex(hash))

    print("TrueHashSHA3-512-Hex: 01dedd5de4ef14642445ba5f5b97c15e47b9ad931326e4b0727cd94cefc44fff23f07bf543139939b49128caf436dc1bdee54fcb24023a08d9403f9b4bf0d450")


# test()
