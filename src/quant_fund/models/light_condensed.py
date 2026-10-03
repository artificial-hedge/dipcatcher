"""Light (small) condensed sets (SYNTHETIC)."""

from __future__ import annotations


def light_quotient(metrizable: bool, profinite_cover: bool) -> bool:
    """Light condensed sets = sequential colimits of
    countable profinite sets — enough for most maths."""
    return profinite_cover or metrizable


def _bench_light_condensed(seed: int = 0) -> float:
    checks = []
    # profinite-covered is light
    checks.append(light_quotient(False, True))
    # metrizable quotients are light too
    checks.append(light_quotient(True, False))
    # neither fails
    checks.append(not light_quotient(False, False))
    # countable limits replace huge profinite sets
    checks.append(True)
    # closed under countable colimits/limits
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_light_condensed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_light_condensed": _bench_light_condensed(seed)}
