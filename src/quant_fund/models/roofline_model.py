"""Roofline bandwidth model — SYNTHETIC.

attainable = min(peak_flops, bw * operational_intensity). Verified at
both regimes + ridge point + arithmetic-intensity monotonicity.
"""

from __future__ import annotations


def attainable(peak: float, bw: float, oi: float) -> float:
    return min(peak, bw * oi)


def bench_roofline_model(seed: int = 20261231 + 335) -> dict[str, float]:
    _ = seed
    peak, bw = 1000.0, 100.0  # GFLOP/s, GB/s
    ridge = peak / bw  # 10 flop/byte
    lo = attainable(peak, bw, ridge / 10)
    hi = attainable(peak, bw, ridge * 10)
    at = attainable(peak, bw, ridge)
    bound_ok = all(
        attainable(peak, bw, oi) <= peak + 1e-9 and attainable(peak, bw, oi) <= bw * oi + 1e-9
        for oi in (0.1, 1.0, 5.0, 10.0, 50.0, 500.0)
    )
    mono = all(
        attainable(peak, bw, a) <= attainable(peak, bw, b) + 1e-9
        for a, b in [(0.5, 1.0), (1.0, 8.0), (8.0, 9.0), (10.0, 20.0)]
    )
    return {
        "synthetic_mem_bound": float(lo == bw * ridge / 10),
        "synthetic_compute_bound": float(hi == peak),
        "synthetic_ridge_exact": float(at == peak),
        "synthetic_never_exceeds": float(bound_ok),
        "synthetic_monotone_oi": float(mono),
    }
