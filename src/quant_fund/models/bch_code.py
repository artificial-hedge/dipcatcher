"""BCH(15,7,2) systematic encode + Berlekamp-Massey/Chien decode (SYNTHETIC).

GF(16) with primitive polynomial x^4+x+1. Encode: c = m*x^8 + (m*x^8 mod g)
with generator g = x^8+x^7+x^6+x^4+1. Decode: syndromes S_j = r(alpha^j)
for j=1..4; Berlekamp-Massey yields the error locator; a Chien search
finds the roots and flips the located bits. Bench sweeps every planted
1- and 2-bit error pattern plus a triple-error honest-negative case.
"""

import itertools

from quant_fund.models._code_synth import BCH_G, BCH_K, BCH_N, BCH_POLY, msg_bits

_EXP = [0] * 30
_LOG = [0] * 16
_x = 1
for _i in range(15):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x10:
        _x ^= BCH_POLY
for _i in range(15, 30):
    _EXP[_i] = _EXP[_i - 15]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _gf_div(a: int, b: int) -> int:
    return _EXP[_LOG[a] - _LOG[b]]


def _poly_div_mod(numer: int) -> int:
    """numer mod generator; polys as int bitmasks."""
    shift = numer.bit_length() - BCH_G.bit_length()
    while shift >= 0:
        numer ^= BCH_G << shift
        shift = numer.bit_length() - BCH_G.bit_length()
    return numer


def _encode(m: int) -> int:
    shifted = m << (BCH_N - BCH_K)
    return shifted | _poly_div_mod(shifted)


def _poly_eval_gf(c: int, alpha_pow: int) -> int:
    """Evaluate codeword poly at alpha^alpha_pow (Horner, MSB first)."""
    acc = 0
    base = _EXP[alpha_pow % 15]
    for i in range(BCH_N - 1, -1, -1):
        acc = _gf_mul(acc, base) ^ ((c >> i) & 1)
    return acc


def _berlekamp_massey(s: list[int]) -> list[int]:
    c = [0] * (len(s) + 1)
    b = [0] * (len(s) + 1)
    c[0] = b[0] = 1
    l_deg, m_shift, bb = 0, 1, 1
    for n_i in range(len(s)):
        d = s[n_i]
        for i in range(1, l_deg + 1):
            d ^= _gf_mul(c[i], s[n_i - i])
        if d == 0:
            m_shift += 1
            continue
        coef = _gf_div(d, bb)
        t = c[:]
        for i in range(len(b)):
            if b[i] and i + m_shift < len(c):
                c[i + m_shift] ^= _gf_mul(coef, b[i])
        if 2 * l_deg <= n_i:
            l_deg = n_i + 1 - l_deg
            b, bb = t, d
            m_shift = 1
        else:
            m_shift += 1
    return c[: l_deg + 1]


def _decode(r: int) -> int:
    syn = [_poly_eval_gf(r, j) for j in range(1, 5)]
    if all(s == 0 for s in syn):
        return r
    locator = _berlekamp_massey(syn)
    out = r
    for pos in range(BCH_N):
        val = 0
        for j, coef in enumerate(locator):
            val ^= _gf_mul(coef, _EXP[(j * (15 - pos)) % 15])
        if val == 0:
            out ^= 1 << pos
    return out


def bench_bch_code(seed: int = 5007) -> dict[str, float]:
    m = sum(int(b) << i for i, b in enumerate(msg_bits(seed, BCH_K)))
    c = _encode(m)
    ok1 = ok2 = 0
    for i in range(BCH_N):
        ok1 += int(_decode(c ^ (1 << i)) == c)
    for i, j in itertools.combinations(range(BCH_N), 2):
        ok2 += int(_decode(c ^ (1 << i) ^ (1 << j)) == c)
    dec3 = _decode(c ^ 0b0111)  # 3 errors: beyond t=2 guarantee
    return {
        "synthetic_bch_single_ok": float(ok1),
        "synthetic_bch_single_frac": ok1 / BCH_N,
        "synthetic_bch_double_ok": float(ok2),
        "synthetic_bch_double_frac": ok2 / (BCH_N * (BCH_N - 1) // 2),
        "synthetic_bch_triple_wrong": float(dec3 != c),
        "synthetic_bch_clean_ok": float(_decode(c) == c),
    }
