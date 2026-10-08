"""Class numbers of imaginary quadratic fields (SYNTHIC table) (SYNTHETIC)."""

from __future__ import annotations

# Heegner numbers have class number 1; disc -23 is the first with h = 3.
_CLASS_TABLE = {
    -3: 1,
    -4: 1,
    -7: 1,
    -8: 1,
    -11: 1,
    -12: 1,
    -15: 2,
    -16: 1,
    -19: 1,
    -20: 2,
    -23: 3,
    -24: 2,
    -27: 1,
    -28: 1,
    -31: 3,
    -40: 2,
    -43: 1,
    -51: 2,
    -52: 2,
    -67: 1,
    -88: 2,
    -148: 2,
    -163: 1,
    -232: 2,
}


def class_number(d: int) -> int:
    """h(d) for fundamental imaginary quadratic discriminants."""
    return _CLASS_TABLE.get(d, 0)


def is_pid(d: int) -> bool:
    """O_K is a PID iff h(d) = 1."""
    return class_number(d) == 1


def _bench_class_group_toy(seed: int = 0) -> float:
    checks = []
    # all nine Heegner discriminants have class number 1
    heegner = [-3, -4, -7, -8, -11, -19, -43, -67, -163]
    checks.append(all(class_number(d) == 1 for d in heegner))
    # h(-23) = 3 (first nontrivial)
    checks.append(class_number(-23) == 3)
    # h(-15) = 2
    checks.append(class_number(-15) == 2)
    # Z[i] is a PID
    checks.append(is_pid(-4))
    # Q(sqrt(-15)) is not
    checks.append(not is_pid(-15))
    # unknown discriminant returns 0, not a PID
    checks.append(class_number(-999) == 0)
    return float(sum(checks) / len(checks))


def bench_class_group_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_class_group_toy": _bench_class_group_toy(seed)}
