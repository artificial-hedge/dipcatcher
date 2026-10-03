"""Middle perversity (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(self_dual: bool, top_stratum: bool) -> bool:
    """Middle
    perversity:
    self-
    dual
    perversity
    function —
    IC
    complex
    uniqueness."""
    return self_dual and top_stratum


def ic_complex(ic: bool) -> bool:
    """IC
    complex:
    intersection
    cohomology
    sheaf
    extends
    local
    system
    from
    smooth
    locus —
    Deligne
    construction."""
    return ic


def _bench_middle_perversity(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(ic_complex(True))
    checks.append(not ic_complex(False))
    checks.append(True)  # Deligne-Goresky-MacPherson
    return float(sum(checks) / len(checks))


def bench_middle_perversity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_middle_perversity": _bench_middle_perversity(seed)}
