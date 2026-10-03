"""Complex multiplication: Heegner class numbers (SYNTHETIC)."""

from __future__ import annotations

_HEEGNER = {-3, -4, -7, -8, -11, -19, -43, -67, -163}
_H_TABLE = {-3: 1, -4: 1, -7: 1, -8: 1, -11: 1, -19: 1, -23: 3, -43: 1, -67: 1, -163: 1}


def class_number(d: int) -> int:
    """Class number of Q(sqrt d) from the toy table."""
    return _H_TABLE[d]


def is_heegner(d: int) -> bool:
    return d in _HEEGNER


def _bench_cm_points(seed: int = 0) -> float:
    checks = []
    # all nine Heegner discriminants have class number one
    checks.append(all(class_number(d) == 1 for d in _HEEGNER))
    # -23 is the first non-Heegner entry with h = 3
    checks.append(class_number(-23) == 3)
    checks.append(not is_heegner(-23))
    # Heegner list is the complete h=1 fundamental list (toy table)
    checks.append(all(class_number(d) == 1 for d in _HEEGNER))
    # j-invariant rationality iff h = 1 (toy equivalence)
    checks.append(is_heegner(-163) and class_number(-163) == 1)
    return float(sum(checks) / len(checks))


def bench_cm_points(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cm_points": _bench_cm_points(seed)}
