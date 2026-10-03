"""Sheaf condition on discrete covers (SYNTHETIC)."""

from __future__ import annotations


def glues(locals_: dict[int, int], agree: bool) -> int | None:
    """Matching family on pairwise-disjoint cover glues uniquely;
    on overlapping cover requires agreement on overlaps."""
    if not agree:
        return None
    return sum(locals_.values())


def _bench_sheaf_cond(seed: int = 0) -> float:
    checks = []
    # compatible sections glue
    checks.append(glues({0: 1, 1: 2}, True) == 3)
    # incompatible on overlaps -> no glue
    checks.append(glues({0: 1}, False) is None)
    # empty cover: unique section
    checks.append(glues({}, True) == 0)
    # single section always glues
    checks.append(glues({5: 7}, True) == 7)
    # sheaf of constant functions: global = sum on disjoint union
    checks.append(glues({0: 1, 1: 1, 2: 1}, True) == 3)
    return float(sum(checks) / len(checks))


def bench_sheaf_cond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheaf_cond": _bench_sheaf_cond(seed)}
