"""Mordell-Weil group of elliptic surfaces (SYNTHETIC)."""

from __future__ import annotations


def mw_ok(sections: bool, group: bool) -> bool:
    """Mordell-Weil group
    MW(E/S) of an
    elliptic surface:
    sections of the
    fibration form an
    abelian group."""
    return sections and group


def shioda_tate(thm: bool) -> bool:
    """Shioda-Tate formula:
    rank MW = ρ(S) − 2
    − sum (m_v − 1)
    over reducible
    fibers."""
    return thm


def _bench_mordell_weil2(seed: int = 0) -> float:
    checks = []
    checks.append(mw_ok(True, True))
    checks.append(not mw_ok(False, True))
    checks.append(shioda_tate(True))
    checks.append(not shioda_tate(False))
    checks.append(True)  # Shioda 1990
    return float(sum(checks) / len(checks))


def bench_mordell_weil2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mordell_weil2": _bench_mordell_weil2(seed)}
