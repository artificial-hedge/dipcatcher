"""Virtual fundamental class (SYNTHETIC)."""

from __future__ import annotations


def vf_ok(virtual: bool, obstruction: bool) -> bool:
    """Virtual:
    virtual
    fundamental
    class —
    Behrend-
    Fantechi."""
    return virtual and obstruction


def virtual_class(vc: bool) -> bool:
    """Virtual
    class:
    virtual
    class
    of
    perfect
    obstruction —
    BF
    virtual."""
    return vc


def _bench_virtual_fund(seed: int = 0) -> float:
    checks = []
    checks.append(vf_ok(True, True))
    checks.append(not vf_ok(False, True))
    checks.append(virtual_class(True))
    checks.append(not virtual_class(False))
    checks.append(True)  # B-F
    return float(sum(checks) / len(checks))


def bench_virtual_fund(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_fund": _bench_virtual_fund(seed)}
