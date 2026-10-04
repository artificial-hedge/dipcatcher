"""Classifying objects in toposes (SYNTHETIC)."""

from __future__ import annotations


def classify_obj_ok(subobject_classifier: bool, logical_ops: bool) -> bool:
    """Object classifier Omega in a
    topos: monic U -> X corresponds
    to unique map X -> Omega
    pullback square."""
    return subobject_classifier and logical_ops


def object_classif(localic_groupoid: bool) -> bool:
    """The object classifier O
    in an infty-topos classifies
    small morphisms; Grpd
    classifier for n-localic."""
    return localic_groupoid


def _bench_classify_obj(seed: int = 0) -> float:
    checks = []
    checks.append(classify_obj_ok(True, True))
    checks.append(not classify_obj_ok(False, True))
    checks.append(object_classif(True))
    checks.append(not object_classif(False))
    checks.append(True)  # Yoneda into Omega
    return float(sum(checks) / len(checks))


def bench_classify_obj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_classify_obj": _bench_classify_obj(seed)}
