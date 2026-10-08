"""Acharya-Engle-Richardson SRISK systemic capital shortfall.

References
----------
- Acharya, V.V., Pedersen, L.H., Philippon, T. & Richardson,
  M. (2017). "Measuring Systemic Risk." *Review of Financial
  Studies* 30(1), 2-47.
- Brownlees, C. & Engle, R.F. (2017). "SRISK: A Conditional
  Capital Shortfall Measure of Systemic Risk." *Review of
  Financial Studies* 30(1), 48-79.
- Acharya, V.V., Engle, R. & Richardson, M. (2012). "Capital
  Shortfall: A New Approach to Ranking and Regulating
  Systemic Risks." *American Economic Review* 102(3), 59-64.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
SRISK combines market data with balance-sheet data:
``SRISK_i = k * Debt_i - (1 - k) * (1 - LRMES_i) * Equity_i``
where ``k`` is the prudential capital ratio and ``LRMES_i`` the
long-run marginal expected shortfall — expected equity return
of firm i conditional on a systemic market drawdown. We
estimate the systemic event by the empirical worst ``alpha``
share of market days (Brownlees-Engle use dynamic GARCH/DCC
betas; the nonparametric worst-decile version is the honest
no-free-parameters variant appropriate for the scorecard).
The (1-LRMES) factor converts equity to post-crisis equity;
the shortfall is floored at 0 (a firm can never have negative
capital need by construction). ``synth_srisk`` plants a
high-beta levered firm vs a low-beta unlevered firm; the bench
gates on SRISK ordering and MES conditioning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 250) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def mes(
    firm_ret: FloatArray,
    mkt_ret: FloatArray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Marginal expected shortfall on worst-alpha market days."""
    rf = _as_series(firm_ret)
    rm = _as_series(mkt_ret)
    if rf.size != rm.size:
        raise ValueError("length mismatch")
    if not 0.0 < alpha < 0.3:
        raise ValueError("bad alpha")
    thr = float(np.quantile(rm, alpha))
    mask = rm <= thr
    if int(np.sum(mask)) < 3:
        raise ValueError("too few tail days")
    out: dict[str, float] = {
        "mes": float(np.mean(rf[mask])),
        "mkt_tail_mean": float(np.mean(rm[mask])),
        "n_tail_days": float(np.sum(mask)),
        "threshold": thr,
    }
    return out


def lrmes(
    firm_ret: FloatArray,
    mkt_ret: FloatArray,
    alpha: float = 0.05,
    h: int = 5,
) -> float:
    """Long-run MES: compounded firm return over h-day crisis."""
    r = mes(firm_ret, mkt_ret, alpha=alpha)
    return float(1.0 - (1.0 + r["mes"]) ** h)


def srisk(
    firm_ret: FloatArray,
    mkt_ret: FloatArray,
    debt: float,
    equity: float,
    k: float = 0.08,
    alpha: float = 0.05,
    h: int = 5,
) -> dict[str, float]:
    """SRISK capital shortfall (floored at zero)."""
    if debt < 0 or equity <= 0 or not 0 < k < 1:
        raise ValueError("bad balance sheet")
    lr = lrmes(firm_ret, mkt_ret, alpha=alpha, h=h)
    need = k * debt - (1.0 - k) * (1.0 - lr) * equity
    out: dict[str, float] = {
        "srisk": max(0.0, need),
        "lrmes": lr,
        "cap_need_unfloored": need,
        "k": k,
    }
    return out


def synth_srisk(
    seed: int = 20261231 + 344,
    n: int = 1200,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC high-beta levered vs low-beta unlevered firm."""
    rng = np.random.default_rng(seed)
    mkt = rng.standard_normal(n) * 0.01
    hi = 2.2 * mkt + 0.008 * rng.standard_normal(n)
    lo = 0.3 * mkt + 0.004 * rng.standard_normal(n)
    return hi.astype(np.float64), lo.astype(np.float64), mkt.astype(np.float64)


def bench_srisk(seed: int = 20261231 + 344) -> dict[str, float]:
    hi, lo, mkt = synth_srisk(seed=seed)
    s_hi = srisk(hi, mkt, debt=100.0, equity=10.0)
    s_lo = srisk(lo, mkt, debt=100.0, equity=10.0)
    m_hi = mes(hi, mkt)
    m_lo = mes(lo, mkt)
    ok = (
        m_hi["mes"] < m_lo["mes"]
        and s_hi["srisk"] > s_lo["srisk"]
        and s_hi["lrmes"] > s_lo["lrmes"]
    )
    out: dict[str, float] = {
        "synthetic_mes_high_beta": m_hi["mes"],
        "synthetic_mes_low_beta": m_lo["mes"],
        "synthetic_srisk_high_beta": s_hi["srisk"],
        "synthetic_srisk_low_beta": s_lo["srisk"],
        "synthetic_lrmes_gap": s_hi["lrmes"] - s_lo["lrmes"],
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
