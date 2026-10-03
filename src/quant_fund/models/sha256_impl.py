"""SHA-256 compression function implemented from spec (FIPS 180-4).

Full padding, message schedule, and 64-round compression in pure Python;
bench cross-checks the hex digest of a fixture message against hashlib
plus the avalanche property (one input bit flip changes ~50% of digest
bits).
"""

import hashlib

import numpy as np

from quant_fund.models._crypto_synth import SHA_MSG

_K = np.array(
    [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ],
    dtype=np.uint32,
)

_H0 = np.array(
    [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ],
    dtype=np.uint32,
)


def _rotr(x: np.uint32, n: int) -> np.uint32:
    return np.uint32((np.uint32(x) >> n) | (np.uint32(x) << np.uint32(32 - n)))


def sha256(msg: bytes) -> str:
    data = bytearray(msg)
    ml = len(data) * 8
    data.append(0x80)
    while len(data) % 64 != 56:
        data.append(0)
    data += ml.to_bytes(8, "big")
    h = _H0.copy()
    for off in range(0, len(data), 64):
        w = np.zeros(64, dtype=np.uint32)
        for i in range(16):
            w[i] = int.from_bytes(data[off + 4 * i : off + 4 * i + 4], "big")
        for i in range(16, 64):
            s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = np.uint32(
                np.uint64(w[i - 16]) + np.uint64(s0) + np.uint64(w[i - 7]) + np.uint64(s1)
            )
        a, b, c, d, e, f, g, hh = h
        for i in range(64):
            s1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            t1 = np.uint32(
                (np.uint64(hh) + np.uint64(s1) + np.uint64(ch) + np.uint64(_K[i]) + np.uint64(w[i]))
                & 0xFFFFFFFF
            )
            s0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = np.uint32((np.uint64(s0) + np.uint64(maj)) & 0xFFFFFFFF)
            hh, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                np.uint32((np.uint64(d) + np.uint64(t1)) & 0xFFFFFFFF),
                c,
                b,
                a,
                np.uint32((np.uint64(t1) + np.uint64(t2)) & 0xFFFFFFFF),
            )
        h = (
            (h.astype(np.uint64) + np.array([a, b, c, d, e, f, g, hh], dtype=np.uint64))
            & 0xFFFFFFFF
        ).astype(np.uint32)
    return "".join(f"{int(x):08x}" for x in h)


def bench_sha256_impl(seed: int = 4701) -> dict[str, float]:
    del seed
    mine = sha256(SHA_MSG)
    ref = hashlib.sha256(SHA_MSG).hexdigest()
    # avalanche: flip one bit of the message
    flipped = bytearray(SHA_MSG)
    flipped[0] ^= 0x01
    h2 = sha256(bytes(flipped))
    bits = sum(bin(int(a, 16) ^ int(b, 16)).count("1") for a, b in zip(mine, h2, strict=True))
    return {
        "synthetic_sha_kat_match": float(mine == ref),
        "synthetic_sha_avalanche_bits": float(bits),
        "synthetic_sha_avalanche_ratio": float(bits) / 256.0,
        "synthetic_sha_empty_match": float(sha256(b"") == hashlib.sha256(b"").hexdigest()),
        "synthetic_sha_long_match": float(
            sha256(b"x" * 200) == hashlib.sha256(b"x" * 200).hexdigest()
        ),
    }
