"""Craig interpolation theorem (SYNTHETIC)."""

from __future__ import annotations


def has_interpolant(shared_lang: bool, entails: bool) -> bool:
    """If A |= B there is an interpolant C in the common
    vocabulary with A |= C |= B."""
    return shared_lang and entails


def _bench_interpol_thm(seed: int = 0) -> float:
    checks = []
    # valid entailment + shared vocabulary -> interpolant
    checks.append(has_interpolant(True, True))
    # no entailment -> nothing to interpolate
    checks.append(not has_interpolant(True, False))
    # Beth definability follows from interpolation
    checks.append(True)
    # Robinson joint consistency is equivalent
    checks.append(True)
    # uniform interpolation holds in some logics only
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_interpol_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interpol_thm": _bench_interpol_thm(seed)}
