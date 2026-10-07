"""AES S-box derived from GF(2^8) inversion + affine transform (SYNTHETIC).

S(x) = A * x^{-1} + 0x63 computed bitwise over the Rijndael field with
modulus x^8 + x^4 + x^3 + x + 1. Bench: all 256 entries vs spot-checked
KATs, bijectivity, and the AES differential-uniformity bound (<= 4).
"""

import numpy as np

from quant_fund.models._crypto_synth import AES_SBOX_KNOWN

_POLY = 0x11B


def _gmul(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= _POLY & 0xFF
        b >>= 1
    return p


def _ginv(a: int) -> int:
    if a == 0:
        return 0
    # brute inverse is fine for 256 entries
    for b in range(1, 256):
        if _gmul(a, b) == 1:
            return b
    return 0


def sbox(x: int) -> int:
    inv = _ginv(x)
    out = inv
    for i in range(4):
        out ^= ((inv << (i + 1)) | (inv >> (7 - i))) & 0xFF
    return out ^ 0x63


def _full_table() -> np.ndarray:
    return np.array([sbox(x) for x in range(256)], dtype=np.uint8)


def bench_aes_sbox(seed: int = 4703) -> dict[str, float]:
    del seed
    tbl = _full_table()
    kat_ok = all(int(tbl[k]) == v for k, v in AES_SBOX_KNOWN.items())
    is_perm = len(set(tbl.tolist())) == 256
    # S(S(x)) should have no fixed point
    t2 = tbl[tbl]
    fixed = int(np.sum(t2 == np.arange(256)))
    # differential uniformity: max_x,y |{x: S(x)^S(x^dx)=dy}| <= 4 for AES
    x = np.arange(256)
    worst = 0
    for dx in range(1, 256):
        d = tbl[x] ^ tbl[x ^ dx]
        worst = max(worst, int(np.bincount(d, minlength=256).max()))
    return {
        "synthetic_sbox_kat": float(kat_ok),
        "synthetic_sbox_permutation": float(is_perm),
        "synthetic_sbox_double_fixed": float(fixed),
        "synthetic_sbox_diff_uniformity": float(worst),
        "synthetic_sbox_biject_err": float(256 - len(set(tbl.tolist()))),
    }
