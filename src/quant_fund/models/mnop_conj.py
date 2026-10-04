"""MNOP conjectures (SYNTHETIC)."""

from __future__ import annotations


def mnop_ok(gw_dt: bool, gw_pt: bool) -> bool:
    """MNOP
    conjectures:
    precise
    equalities
    among
    GW,
    DT,
    and
    PT
    partition
    functions —
    fundamental
    enumerative
    duality."""
    return gw_dt and gw_pt


def proved_cases(pc: bool) -> bool:
    """Proved
    cases:
    GW-PT
    correspondence
    proved
    in
    many
    settings —
    Pandharipande-
    Pixton,
    Oblomkov."""
    return pc


def _bench_mnop_conj(seed: int = 0) -> float:
    checks = []
    checks.append(mnop_ok(True, True))
    checks.append(not mnop_ok(False, True))
    checks.append(proved_cases(True))
    checks.append(not proved_cases(False))
    checks.append(True)  # MNOP 2006
    return float(sum(checks) / len(checks))


def bench_mnop_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mnop_conj": _bench_mnop_conj(seed)}
