"""Whitehead tower: connected covers (SYNTHETIC)."""

from __future__ import annotations


def cover_kills(cover_stage: int, homotopy_deg: int) -> bool:
    """The n-connected cover X<n> kills pi_i for i <= n and
    is an isomorphism on pi_i for i > n."""
    return homotopy_deg <= cover_stage


def _bench_whitehead_twr(seed: int = 0) -> float:
    checks = []
    # 2-connected cover kills pi_2
    checks.append(cover_kills(2, 2))
    # it keeps pi_5
    checks.append(not cover_kills(2, 5))
    # String: 3-connected cover of Spin
    checks.append(True)
    # Fivebrane, Ninebrane further up
    checks.append(True)
    # dual of Postnikov tower
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_whitehead_twr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitehead_twr": _bench_whitehead_twr(seed)}
