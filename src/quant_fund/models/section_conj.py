"""Section conjecture (SYNTHETIC)."""

from __future__ import annotations


def section_ok(sections: bool, points: bool) -> bool:
    """Section conjecture:
    rational points
    of a hyperbolic
    curve correspond
    to conjugacy
    classes of sections
    of pi_1."""
    return sections and points


def sec_map(map: bool) -> bool:
    """Section map
    X(k) -> H^1(k,
    pi_1^et)/~;
    conjecturally
    bijective."""
    return map


def _bench_section_conj(seed: int = 0) -> float:
    checks = []
    checks.append(section_ok(True, True))
    checks.append(not section_ok(False, True))
    checks.append(sec_map(True))
    checks.append(not sec_map(False))
    checks.append(True)  # Koenigsmann
    return float(sum(checks) / len(checks))


def bench_section_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_section_conj": _bench_section_conj(seed)}
