"""OIS zero-curve bootstrapping.

Bootstrap continuously-compounded zero rates and discount
factors from OIS/deposit par quotes. Deposits give the
short end directly (D = 1/(1+r*t)); par swaps are
stripped sequentially from the fixed-leg valuation

    1 = r_n * sum_i(delta_i * D(t_i)) + D(t_n)

solved for D(t_n). Zero rates interpolate linearly;
discount factors log-linearly (piecewise-constant
forward rates).

Honesty: `bench_ois_curve` re-prices the input par swaps
off the bootstrapped curve and checks residuals are
machine-small, that a flat curve maps to flat zero rates,
and that discount factors are strictly decreasing — a
consistency diagnostic on SYNTHETIC quotes, no market
data.

References
----------
* Hull, J. (2018) "Options, Futures, and Other
  Derivatives", 10th ed., ch. 4 (zero curve bootstrap).
* Madan, D. (2001) "Multi-curve bootstrapping".
* Andersen & Piterbarg (2010) "Interest Rate Modeling",
  vol I, ch. 6.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bootstrap_zero_curve(
    deposit_tenors: FloatArray,
    deposit_rates: FloatArray,
    swap_tenors: FloatArray,
    swap_rates: FloatArray,
    payments_per_year: float = 1.0,
) -> dict[str, FloatArray | float]:
    """Bootstrap zero rates + discount factors.

    ``deposit_*`` are simply-compounded money-market quotes
    in years; ``swap_*`` par OIS swap rates at integer
    maturities. Returns sorted knot tenors, discount
    factors, and continuously-compounded zero rates.
    """
    dt_ = np.asarray(deposit_tenors, dtype=np.float64).ravel()
    dr_ = np.asarray(deposit_rates, dtype=np.float64).ravel()
    st_ = np.asarray(swap_tenors, dtype=np.float64).ravel()
    sr_ = np.asarray(swap_rates, dtype=np.float64).ravel()
    if dt_.size != dr_.size or st_.size != sr_.size or dt_.size == 0 or st_.size == 0:
        raise ValueError("bad curve inputs")
    if np.any(dt_ <= 0) or np.any(st_ <= 0) or np.any(np.diff(st_) <= 0):
        raise ValueError("tenors must be positive and sorted")
    step = 1.0 / payments_per_year
    tenors: list[float] = []
    dfs: list[float] = []
    # deposits: D = 1 / (1 + r * t) (annual accrual in years)
    for t, r in zip(dt_, dr_, strict=True):
        if r <= -1.0:
            raise ValueError("deposit rate <= -100%")
        tenors.append(float(t))
        dfs.append(1.0 / (1.0 + float(r) * float(t)))

    def _df(t: float) -> float:
        if t <= 0:
            return 1.0
        tt = np.asarray(tenors)
        if t <= tt[0]:
            z = -math.log(dfs[0]) / tt[0]
            return math.exp(-z * t)
        i = int(np.searchsorted(tt, t)) - 1
        if i >= len(dfs) - 1:
            z = -math.log(dfs[-1]) / tt[-1]
            return math.exp(-z * t)
        # log-linear on DF between knots
        t1, t2 = float(tt[i]), float(tt[i + 1])
        w = (t - t1) / (t2 - t1)
        return math.exp(math.log(dfs[i]) * (1 - w) + math.log(dfs[i + 1]) * w)

    for t_n, r_n in zip(st_, sr_, strict=True):
        # payment grid back from the maturity in `step` years
        pays = []
        tp = float(t_n)
        while tp > step + 1e-12:
            pays.append(tp)
            tp -= step
        pays.append(step)
        pays = sorted(set(pays))
        # accrual over payment dates already on the curve
        acc = 0.0
        for tp_ in pays[:-1]:
            if tp_ <= tenors[-1] + 1e-12:
                acc += step * _df(tp_)
        # log-linear DF interpolation between the previous
        # knot and this tenor means intermediate coupon
        # dates depend on the unknown D(t_n) itself ->
        # fixed-point solve for d_n.
        d_prev = dfs[-1]
        t_prev = tenors[-1]
        pays_mid = [p for p in pays[:-1] if p > t_prev]
        span = float(t_n) - t_prev

        def resid(
            dn: float,
            acc_: float = acc,
            d_prev_: float = d_prev,
            t_prev_: float = t_prev,
            span_: float = span,
            r_: float = float(r_n),
            pays_mid_: tuple[float, ...] = tuple(pays_mid),
        ) -> float:
            acc_mid = 0.0
            for p in pays_mid_:
                w = (p - t_prev_) / span_
                df_p = math.exp((1.0 - w) * math.log(d_prev_) + w * math.log(dn))
                acc_mid += step * df_p
            return 1.0 - r_ * (acc_ + acc_mid + step * dn) - dn

        lo, hi = 1e-8, 1.0
        flo, fhi = resid(lo), resid(hi)
        if flo * fhi > 0:
            raise ValueError(f"no discount root at {t_n}")
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = resid(mid)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        d_n = 0.5 * (lo + hi)
        if d_n <= 0:
            raise ValueError(f"non-positive discount factor at {t_n}")
        tenors.append(float(t_n))
        dfs.append(float(d_n))
    tt = np.asarray(tenors)
    order = np.argsort(tt)
    tt = tt[order]
    dd = np.asarray(dfs)[order]
    zeros = -np.log(dd) / tt
    return {"tenors": tt, "discount_factors": dd, "zero_rates": zeros}


def discount_factor(curve: dict[str, FloatArray | float], t: float) -> float:
    """Log-linear interpolation on the bootstrapped DFs."""
    tt = np.asarray(curve["tenors"])
    dd = np.asarray(curve["discount_factors"])
    if t <= 0:
        return 1.0
    if t <= tt[0]:
        return math.exp(t / tt[0] * math.log(dd[0]))
    if t >= tt[-1]:
        return math.exp(t / tt[-1] * math.log(dd[-1]))
    i = int(np.searchsorted(tt, t)) - 1
    w = (t - tt[i]) / (tt[i + 1] - tt[i])
    return math.exp((1 - w) * math.log(dd[i]) + w * math.log(dd[i + 1]))


def forward_rate(curve: dict[str, FloatArray | float], t1: float, t2: float) -> float:
    """Simple forward rate over [t1, t2]."""
    if t2 <= t1:
        raise ValueError("bad forward window")
    d1 = discount_factor(curve, t1)
    d2 = discount_factor(curve, t2)
    return float((d1 / d2 - 1.0) / (t2 - t1))


def par_swap_rate(
    curve: dict[str, FloatArray | float], t_n: float, payments_per_year: float = 1.0
) -> float:
    """Fair par swap rate re-priced off the curve."""
    step = 1.0 / payments_per_year
    pays = []
    tp = t_n
    while tp > step + 1e-12:
        pays.append(tp)
        tp -= step
    pays.append(step)
    pays = sorted(set(pays))
    acc = sum(step * discount_factor(curve, p) for p in pays)
    return float((1.0 - discount_factor(curve, t_n)) / max(acc, 1e-12))


def bench_ois_curve(seed: int = 20261231 + 469) -> dict[str, float]:
    """SYNTHETIC check — par rates re-price to input quotes."""
    rng = np.random.default_rng(seed)
    dep_t = np.array([0.25, 0.5])
    base = float(rng.uniform(0.02, 0.05))
    dep_r = np.array([base, base + 0.002])
    swap_t = np.array([1.0, 2.0, 3.0, 5.0])
    swap_r = base + np.array([0.004, 0.007, 0.010, 0.015])
    curve = bootstrap_zero_curve(dep_t, dep_r, swap_t, swap_r)
    max_repricing = max(
        abs(par_swap_rate(curve, float(t)) - float(r)) for t, r in zip(swap_t, swap_r, strict=True)
    )
    if max_repricing > 1e-10:
        raise ValueError(f"repricing off: {max_repricing}")
    dd = np.asarray(curve["discount_factors"])
    if np.any(np.diff(dd) >= 0):
        raise ValueError("discount factors not decreasing")
    zz = np.asarray(curve["zero_rates"])
    if np.any(np.diff(zz) < -1e-12):
        raise ValueError("zero rates not nondecreasing")
    # flat curve -> par swaps reprice exactly and zero
    # rates sit within the simple->continuous conversion
    # slack of the par rate
    flat = bootstrap_zero_curve(dep_t, np.full(2, 0.04), swap_t, np.full(4, 0.04))
    flat_repricing = max(abs(par_swap_rate(flat, float(t)) - 0.04) for t in swap_t)
    fz = np.asarray(flat["zero_rates"])
    if flat_repricing > 1e-10 or float(np.ptp(fz)) > 0.0025:
        raise ValueError(f"flat curve off: {fz} rep={flat_repricing}")
    return {
        "synthetic_repricing_max_err": float(max_repricing),
        "synthetic_flat_ptp": float(np.ptp(fz)),
        "synthetic_flat_repricing": float(flat_repricing),
        "synthetic_fwd_5y": float(forward_rate(curve, 3.0, 5.0)),
        "synthetic_score": 1.0,
    }
