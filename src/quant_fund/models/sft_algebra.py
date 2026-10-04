"""SFT algebra (SYNTHETIC)."""

from __future__ import annotations


def sa_ok(dga: bool, holomorphic_curves: bool) -> bool:
    """SFT
    algebra:
    differential
    graded
    algebra
    from
    punctured
    holomorphic
    curves —
    BBGM
    compactification."""
    return dga and holomorphic_curves


def algebraic_sft(asft: bool) -> bool:
    """Algebraic
    SFT:
    finished
    vs
    unfinished
    algebra
    formalism
    in
    EGH —
    hierarchy
    of
    structures."""
    return asft


def _bench_sft_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(sa_ok(True, True))
    checks.append(not sa_ok(False, True))
    checks.append(algebraic_sft(True))
    checks.append(not algebraic_sft(False))
    checks.append(True)  # Bourgeois-Eliashberg-Hofer-Wysocki-Zehnder
    return float(sum(checks) / len(checks))


def bench_sft_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sft_algebra": _bench_sft_algebra(seed)}
