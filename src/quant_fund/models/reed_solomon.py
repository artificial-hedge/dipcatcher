"""Reed–Solomon canon: RS(n,k) systematic encode + Berlekamp–
Massey error-locator + Forney magnitude + Chien search decode
over GF(256). Bench: corrects exactly ⌊(n−k)/2⌋ errors, fails
loudly beyond capacity. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.gf256 import (
    _EXP,
    gf_div,
    gf_mul,
    gf_poly_eval,
    gf_poly_mul,
    gf_pow,
)

FloatArray = NDArray[np.float64]


def rs_generator(nsym: int) -> list[int]:
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, _EXP[i]])
    return g


def rs_encode(msg: list[int], nsym: int) -> list[int]:
    """Systematic encode: msg | parity (length len(msg)+nsym)."""
    g = rs_generator(nsym)
    out = list(msg) + [0] * nsym
    for i in range(len(msg)):
        coef = out[i]
        if coef:
            for j in range(len(g)):
                out[i + j] ^= gf_mul(g[j], coef)
    return list(msg) + out[-nsym:]


def _syndromes(r: list[int], nsym: int) -> list[int]:
    return [gf_poly_eval(r, _EXP[i]) for i in range(nsym)]


def rs_decode(r: list[int], nsym: int) -> tuple[list[int] | None, int]:
    """Berlekamp–Massey + Chien + Forney. Returns
    (corrected msg or None, n_errors_found)."""
    synd = _syndromes(r, nsym)
    if max(synd) == 0:
        return list(r[:-nsym]), 0
    # BM: locator Λ, correction poly B, L = current degree,
    # m = shift since last update, b = last nonzero discrepancy
    lam = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for i in range(nsym):
        d = synd[i]
        for j in range(1, L + 1):
            d ^= gf_mul(lam[j], synd[i - j])
        if d == 0:
            m += 1
        elif i >= 2 * L:
            T = list(lam)
            coef = gf_div(d, b)
            need = m + len(B)
            if len(lam) < need:
                lam += [0] * (need - len(lam))
            for j in range(len(B)):
                lam[j + m] ^= gf_mul(coef, B[j])
            B = T
            L = i + 1 - L
            m = 1
            b = d
        else:
            coef = gf_div(d, b)
            need = m + len(B)
            if len(lam) < need:
                lam += [0] * (need - len(lam))
            for j in range(len(B)):
                lam[j + m] ^= gf_mul(coef, B[j])
            m += 1
    return _finish(r, nsym, lam, synd)


def _finish(
    r: list[int], nsym: int, lam: list[int], synd: list[int]
) -> tuple[list[int] | None, int]:
    n = len(r)
    # Chien search: codeword index idx (0 = leftmost/high-order
    # coefficient) is an error iff Λ(α^{n-1-idx}) = 0.
    errs = [idx for idx in range(n) if gf_poly_eval(lam, _EXP[(n - 1 - idx) % 255]) == 0]
    deg = len(lam) - 1
    if len(errs) != deg:
        return None, deg
    # Magnitudes by a GF Vandermonde solve:
    # Σ_k e_k · X_k^i = S_i for i = 0..deg-1, X_k = α^{n-1-idx_k}.
    v = deg
    X = [_EXP[(n - 1 - idx) % 255] for idx in errs]
    A = [[gf_pow(X[k], i) for k in range(v)] for i in range(v)]
    rhs = list(synd[:v])
    for col in range(v):
        piv = next((rr for rr in range(col, v) if A[rr][col] != 0), None)
        if piv is None:
            return None, deg
        A[col], A[piv] = A[piv], A[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        inv = gf_div(1, A[col][col])
        for rr in range(v):
            if rr != col and A[rr][col] != 0:
                f = gf_mul(A[rr][col], inv)
                for cc in range(v):
                    A[rr][cc] ^= gf_mul(f, A[col][cc])
                rhs[rr] ^= gf_mul(f, rhs[col])
    out = list(r)
    for k, idx in enumerate(errs):
        out[idx] ^= gf_div(rhs[k], A[k][k])
    if max(_syndromes(out, nsym)) != 0:
        return None, deg
    return out[:-nsym], deg


def bench_reed_solomon(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    nsym = 10
    k = 100 - nsym
    msg = [int(x) for x in rng.integers(0, 256, k)]
    code = rs_encode(msg, nsym)
    dec0, _ = rs_decode(code, nsym)
    out["synthetic_rs_clean_ber"] = float(np.mean(np.array(dec0) != np.array(msg)))
    bad = list(code)
    for pos in rng.choice(len(bad), 5, replace=False):
        bad[int(pos)] ^= int(rng.integers(1, 256))
    dec, ne = rs_decode(bad, nsym)
    out["synthetic_rs_t5_fixed"] = float(dec == msg)
    out["synthetic_rs_t5_nerr"] = float(ne)
    bad2 = list(code)
    for pos in rng.choice(len(bad2), 7, replace=False):
        bad2[int(pos)] ^= int(rng.integers(1, 256))
    dec2, _ = rs_decode(bad2, nsym)
    out["synthetic_rs_over_capacity_bad"] = float(dec2 != msg)
    bad3 = list(code)
    for pos in rng.choice(len(bad3), 4, replace=False):
        bad3[int(pos)] ^= int(rng.integers(1, 256))
    dec3, _ = rs_decode(bad3, nsym)
    out["synthetic_rs_t4_fixed"] = float(dec3 == msg)
    return out
