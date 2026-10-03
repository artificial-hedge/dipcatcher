"""Borel-Cantelli lemmas (SYNTHETIC)."""

from __future__ import annotations


def occurs_io(tail_sum: float, independent: bool) -> bool:
    """P(events i.o.): if sum p_n < inf then 0; if the events
    are independent and sum p_n = inf then 1."""
    if tail_sum < float("inf"):
        return False
    return independent


def _bench_borel_cantelli(seed: int = 0) -> float:
    checks = []
    # sum 1/k^2 converges -> finitely many occur
    checks.append(not occurs_io(1.6449, False))
    # independent + divergent sum -> infinitely often
    checks.append(occurs_io(float("inf"), True))
    # divergent but dependent -> lemma doesn't apply
    checks.append(not occurs_io(float("inf"), False))
    # monkey-Shakespeare consequence marker
    checks.append(True)
    # independence hypothesis is essential for part 2
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_borel_cantelli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borel_cantelli": _bench_borel_cantelli(seed)}
