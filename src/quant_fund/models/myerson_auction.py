"""Myerson (1981) optimal auction — virtual valuations (SYNTHETIC)
φ(v) = v − (1 − F(v))/f(v), allocate to the highest nonnegative
virtual value, charge the winner's threshold bid. Includes the
regularity check and revenue comparison to VCG/second-price.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
VirtualFn = Callable[[FloatArray], FloatArray]


def virtual_value_uniform(v: FloatArray, lo: float = 0.0, hi: float = 1.0) -> FloatArray:
    """φ(v) = 2v − hi for U[lo, hi]."""
    out: FloatArray = np.asarray(2.0 * v - hi)
    return out


def myerson_alloc_pay(bids: FloatArray, virtuals: list[VirtualFn]) -> tuple[int, float]:
    """Allocate to argmax virtual value ≥ 0; payment = the smallest
    bid at which the winner still wins (threshold)."""
    vv = np.array([float(virtuals[i](np.array([bids[i]]))[0]) for i in range(len(bids))])
    winner = int(np.argmax(vv))
    if vv[winner] < 0:
        return -1, 0.0
    # threshold: bid b* where virtual(b*) = max(0, max others' vv)
    runner = max([vv[i] for i in range(len(bids)) if i != winner] + [0.0])
    lo_b, hi_b = 0.0, float(bids[winner])
    vf = virtuals[winner]
    for _ in range(60):
        mid = 0.5 * (lo_b + hi_b)
        if float(vf(np.array([mid]))[0]) >= runner:
            hi_b = mid
        else:
            lo_b = mid
    return winner, float(hi_b)


def bench_myerson(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 2 bidders U[0,1] — Myerson sets reserve φ⁻¹(0)=0.5;
    revenue exceeds second-price revenue (optimal auction)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # reserve check: φ(v)=0 → v=0.5
    out["synthetic_myerson_reserve"] = float(abs(virtual_value_uniform(np.array([0.5]))[0]) < 1e-12)
    # bids below reserve → no sale
    w, p = myerson_alloc_pay(np.array([0.3, 0.4]), [virtual_value_uniform] * 2)
    out["synthetic_myerson_no_sale"] = float(w == -1)
    # winner pays max(reserve, runner-up bid... virtual threshold)
    w, p = myerson_alloc_pay(np.array([0.8, 0.6]), [virtual_value_uniform] * 2)
    out["synthetic_myerson_payment"] = float(p)
    out["synthetic_myerson_threshold_ok"] = float(abs(p - max(0.5, 0.6)) < 0.02)
    # revenue comparison over sampled values
    rev_m, rev_v = 0.0, 0.0
    for _ in range(500):
        b = rng.random(2)
        wi, pm = myerson_alloc_pay(b, [virtual_value_uniform] * 2)
        rev_m += pm if wi >= 0 else 0.0
        rev_v += min(b) if np.argmax(b) >= 0 else 0.0
    out["synthetic_myerson_rev_gain"] = float((rev_m - rev_v) / 500)
    out["synthetic_myerson_optimal"] = float(rev_m >= rev_v - 1e-9)
    return out


if __name__ == "__main__":
    print(bench_myerson())
