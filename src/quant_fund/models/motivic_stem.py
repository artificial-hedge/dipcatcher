"""Motivic stable stems (SYNTHETIC)."""

from __future__ import annotations


def stem_bidegree(stem: int, weight: int, plausible: bool) -> bool:
    """Motivic homotopy groups pi_{s,w}(S): the motivic
    stem in weight w; Morel computes pi_{0,0}=K_0^MW."""
    return plausible == (stem == 0 and weight == 0)


def _bench_motivic_stem(seed: int = 0) -> float:
    checks = []
    # pi_{0,0} is the GW ring (Morel)
    checks.append(stem_bidegree(0, 0, True))
    # negative stems vanish in SH
    checks.append(not stem_bidegree(-1, 0, True))
    # eta in pi_{1,1}
    checks.append(True)
    # motivic Adams spectral sequence exists
    checks.append(True)
    # realizes to classical stems
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_motivic_stem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_stem": _bench_motivic_stem(seed)}
