"""Decidability of first-order theories (SYNTHETIC)."""

from __future__ import annotations


def is_decidable(theory: str) -> bool:
    """Presburger arithmetic, DLO, RCF, ACF_p are decidable;
    full arithmetic (PA) is not."""
    return theory in {"presburger", "dlo", "rcf", "acf"}


def _bench_decidable_theory(seed: int = 0) -> float:
    checks = []
    checks.append(is_decidable("presburger"))
    checks.append(is_decidable("dlo"))
    checks.append(is_decidable("rcf"))
    # Peano arithmetic undecidable (Godel/Rosser)
    checks.append(not is_decidable("pa"))
    # decidability via quantifier elimination
    checks.append(is_decidable("acf"))
    return float(sum(checks) / len(checks))


def bench_decidable_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decidable_theory": _bench_decidable_theory(seed)}
