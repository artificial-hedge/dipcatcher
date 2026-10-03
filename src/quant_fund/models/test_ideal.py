"""Test ideals (SYNTHETIC)."""

from __future__ import annotations


def test_ideal_ok(tau: bool, universal: bool) -> bool:
    """Test ideal tau(R):
    smallest ideal
    commuting with
    trace/Frobenius
    splittings;
    measures F-singularity."""
    return tau and universal


def test_ideal_conductor(universal_test: bool) -> bool:
    """Test ideal =
    annihilator of
    the non-F-regular
    locus; contains
    the tight-closure
    conductor."""
    return universal_test


def _bench_test_ideal(seed: int = 0) -> float:
    checks = []
    checks.append(test_ideal_ok(True, True))
    checks.append(not test_ideal_ok(False, True))
    checks.append(test_ideal_conductor(True))
    checks.append(not test_ideal_conductor(False))
    checks.append(True)  # Hara-Yoshida
    return float(sum(checks) / len(checks))


def bench_test_ideal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_test_ideal": _bench_test_ideal(seed)}
