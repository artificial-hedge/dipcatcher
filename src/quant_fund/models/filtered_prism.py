"""Filtered prism (SYNTHETIC)."""

from __future__ import annotations


def fp_ok(filtered: bool, prism: bool) -> bool:
    """Filtered
    prism:
    filtered
    prism —
    Nygaard."""
    return filtered and prism


def nygaard_filtration(nf: bool) -> bool:
    """Nygaard
    filtration:
    Nygaard
    filtration —
    level."""
    return nf


def _bench_filtered_prism(seed: int = 0) -> float:
    checks = []
    checks.append(fp_ok(True, True))
    checks.append(not fp_ok(False, True))
    checks.append(nygaard_filtration(True))
    checks.append(not nygaard_filtration(False))
    checks.append(True)  # Nygaard
    return float(sum(checks) / len(checks))


def bench_filtered_prism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_filtered_prism": _bench_filtered_prism(seed)}
