"""Reiner-Rubinstein (1991) closed-form barrier options.

All eight single-barrier European types under Black-Scholes
via the standard six-term decomposition (a,b,c,d,e,f) with
``phi`` (call +1 / put -1) and ``eta`` (down +1 / up -1)
direction indicators and a knock-in rebate ``rebate`` (0
for standard options, which zeroes the e/f terms).

Combination table (K vs H rows):

    DownIn Call : K>H -> c+e      ; K<=H -> a-b+d+e
    DownOut Call: K>H -> a-c+f    ; K<=H -> b-d+f
    UpIn Call   : K>H -> a+e      ; K<=H -> a-b+d+e
    UpOut Call  : K>H -> f        ; K<=H -> b-d+f
    DownIn Put  : K>H -> b-c+d+e  ; K<=H -> a-b+d+e
    DownOut Put : K>H -> a-b+c-d+f; K<=H -> b-d+f
    UpIn Put    : K>H -> a-b+d+e  ; K<=H -> c+e
    UpOut Put   : K>H -> b-d+f    ; K<=H -> a-c+f

Honesty: `bench_barrier_options` checks the in+out parity
identity against the BSM vanilla price on a random grid
and verifies knockout prices stay inside [0, vanilla] — a
pure pricing-consistency diagnostic, no P&L claim.

References
----------
* Reiner & Rubinstein (1991) "Breaking down the barriers",
  Risk 4(8).
* Haug, E.G. (2007) "The Complete Guide to Option Pricing
  Formulas", 2nd ed., ch. 4.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _ncd(x: float) -> float:
    return float(stats.norm.cdf(x))


def _terms(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float,
    rebate: float,
    phi: float,
    eta: float,
) -> dict[str, float]:
    mu = (r - q - 0.5 * sig * sig) / (sig * sig)
    lam = math.sqrt(mu * mu + 2.0 * r / (sig * sig))
    sq = sig * math.sqrt(t)
    x1 = math.log(s / k) / sq + (1.0 + mu) * sq
    x2 = math.log(s / h) / sq + (1.0 + mu) * sq
    y1 = math.log(h * h / (s * k)) / sq + (1.0 + mu) * sq
    y2 = math.log(h / s) / sq + (1.0 + mu) * sq
    z = math.log(h / s) / sq + lam * sq
    hs = h / s
    hs_a = hs ** (2.0 * (mu + 1.0))
    hs_b = hs ** (2.0 * mu)
    a = phi * s * math.exp(-q * t) * _ncd(phi * x1) - phi * k * math.exp(-r * t) * _ncd(
        phi * x1 - phi * sq
    )
    b = phi * s * math.exp(-q * t) * _ncd(phi * x2) - phi * k * math.exp(-r * t) * _ncd(
        phi * x2 - phi * sq
    )
    c = phi * s * math.exp(-q * t) * hs_a * _ncd(eta * y1) - phi * k * math.exp(
        -r * t
    ) * hs_b * _ncd(eta * y1 - eta * sq)
    d = phi * s * math.exp(-q * t) * hs_a * _ncd(eta * y2) - phi * k * math.exp(
        -r * t
    ) * hs_b * _ncd(eta * y2 - eta * sq)
    e = rebate * math.exp(-r * t) * (_ncd(eta * x2 - eta * sq) - hs_b * _ncd(eta * y2 - eta * sq))
    f = rebate * (
        hs ** (mu + lam) * _ncd(eta * z) + hs ** (mu - lam) * _ncd(eta * z - 2.0 * eta * lam * sq)
    )
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}


def vanilla_call(s: float, k: float, t: float, r: float, sig: float, q: float = 0.0) -> float:
    sq = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sig * sig) * t) / sq
    d2 = d1 - sq
    return float(s * math.exp(-q * t) * _ncd(d1) - k * math.exp(-r * t) * _ncd(d2))


def vanilla_put(s: float, k: float, t: float, r: float, sig: float, q: float = 0.0) -> float:
    sq = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sig * sig) * t) / sq
    d2 = d1 - sq
    return float(k * math.exp(-r * t) * _ncd(-d2) - s * math.exp(-q * t) * _ncd(-d1))


def down_call(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float = 0.0,
    knock: str = "out",
    rebate: float = 0.0,
) -> float:
    """Down-and-out / down-and-in call (h < s)."""
    if not (0 < h < s and k > 0 and t > 0 and sig > 0):
        raise ValueError("bad barrier option inputs")
    tm = _terms(s, k, h, t, r, sig, q, rebate, phi=1.0, eta=1.0)
    a, b, c, d, e, f = (tm[x] for x in ("a", "b", "c", "d", "e", "f"))
    if k > h:
        price = a - c + f if knock == "out" else c + e
    else:
        price = b - d + f if knock == "out" else a - b + d + e
    return float(price)


def up_call(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float = 0.0,
    knock: str = "out",
    rebate: float = 0.0,
) -> float:
    """Up-and-out / up-and-in call (h > s)."""
    if not (h > s > 0 and k > 0 and t > 0 and sig > 0):
        raise ValueError("bad barrier option inputs")
    tm = _terms(s, k, h, t, r, sig, q, rebate, phi=1.0, eta=-1.0)
    a, b, d, e, f = tm["a"], tm["b"], tm["d"], tm["e"], tm["f"]
    if k > h:
        price = f if knock == "out" else a + e
    else:
        price = b - d + f if knock == "out" else a - b + d + e
    return float(price)


def down_put(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float = 0.0,
    knock: str = "out",
    rebate: float = 0.0,
) -> float:
    """Down-and-out / down-and-in put (h < s)."""
    if not (0 < h < s and k > 0 and t > 0 and sig > 0):
        raise ValueError("bad barrier option inputs")
    tm = _terms(s, k, h, t, r, sig, q, rebate, phi=-1.0, eta=1.0)
    a, b, c, d, e, f = (tm[x] for x in ("a", "b", "c", "d", "e", "f"))
    if k > h:
        price = a - b + c - d + f if knock == "out" else b - c + d + e
    else:
        price = b - d + f if knock == "out" else a - b + d + e
    return float(price)


def up_put(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float = 0.0,
    knock: str = "out",
    rebate: float = 0.0,
) -> float:
    """Up-and-out / up-and-in put (h > s)."""
    if not (h > s > 0 and k > 0 and t > 0 and sig > 0):
        raise ValueError("bad barrier option inputs")
    tm = _terms(s, k, h, t, r, sig, q, rebate, phi=-1.0, eta=-1.0)
    a, b, c, d, e, f = (tm[x] for x in ("a", "b", "c", "d", "e", "f"))
    if k > h:
        price = b - d + f if knock == "out" else a - b + d + e
    else:
        price = a - c + f if knock == "out" else c + e
    return float(price)


def barrier_price(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sig: float,
    q: float = 0.0,
    kind: str = "doc",
) -> float:
    """Dispatch: 'doc','dic','uic','uoc','dip','dop','uip','uop'."""
    fn = {
        "doc": (down_call, "out"),
        "dic": (down_call, "in"),
        "uoc": (up_call, "out"),
        "uic": (up_call, "in"),
        "dop": (down_put, "out"),
        "dip": (down_put, "in"),
        "uop": (up_put, "out"),
        "uip": (up_put, "in"),
    }.get(kind)
    if fn is None:
        raise ValueError(f"unknown barrier kind {kind}")
    f_, kk = fn
    return f_(s, k, h, t, r, sig, q, kk)


def bench_barrier_options(seed: int = 20261231 + 463) -> dict[str, float]:
    """SYNTHETIC check — in+out parity vs vanilla."""
    rng = np.random.default_rng(seed)
    max_rel = 0.0
    for _ in range(40):
        s = float(rng.uniform(60, 140))
        k = float(rng.uniform(60, 140))
        h = s * float(rng.uniform(0.5, 0.9))
        t = float(rng.uniform(0.25, 2.0))
        sig = float(rng.uniform(0.15, 0.5))
        r = float(rng.uniform(0.0, 0.08))
        van = vanilla_call(s, k, t, r, sig)
        di = down_call(s, k, h, t, r, sig, knock="in")
        do = down_call(s, k, h, t, r, sig, knock="out")
        rel = abs(di + do - van) / max(van, 1e-8)
        max_rel = max(max_rel, rel)
        if min(di, do) < -1e-9:
            raise ValueError(f"negative barrier price: in={di} out={do}")
    if max_rel > 1e-5:
        raise ValueError(f"parity off: {max_rel}")
    # knockout strictly below vanilla for a tight barrier
    s2, k2, t2, sig2, r2 = 100.0, 100.0, 1.0, 0.25, 0.03
    tight = down_call(s2, k2, s2 * 0.98, t2, r2, sig2, knock="out")
    van2 = vanilla_call(s2, k2, t2, r2, sig2)
    if not (0.0 <= tight < van2):
        raise ValueError(f"knockout off: {tight} vs {van2}")
    # up-and-in put approaches vanilla put as barrier -> spot
    p_near = up_put(s2, k2, s2 * 1.001, t2, r2, sig2, knock="in")
    pv = vanilla_put(s2, k2, t2, r2, sig2)
    if abs(p_near - pv) / pv > 0.15:
        raise ValueError(f"uip limit off: {p_near} vs {pv}")
    return {
        "synthetic_parity_max_rel_err": float(max_rel),
        "synthetic_tight_ko_ratio": float(tight / van2),
        "score": 1.0,
    }
